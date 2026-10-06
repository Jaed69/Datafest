"""Hypothesis evaluation on the existing rolling folds (Sep/Oct/Nov).

Reuses the rolling feature builders, model wrappers, frozen structural
parameters and iteration counts, the Gini metric and a paired client-level
bootstrap. Nothing is early-stopped or tuned on a validation month: every
setting below is fixed a priori. Candidates that build on "the best of" earlier
candidates pick it by mean rolling Gini, which is a (disclosed) selection on the
validation months, so those downstream comparisons are exploratory.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import platform
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from datafest import __version__
from datafest.ablation import ROLLING_CUTS, _feature_matrix
from datafest.data import validate_competition_data
from datafest.diagnostics import _weighted_gini_preparation
from datafest.features import CATEGORICAL_COLUMNS, _month_ordinal
from datafest.final_fit import FINAL_ITERATIONS
from datafest.lineage import file_record, repository_revision, sha256_file, write_json
from datafest.metrics import gini_score
from datafest.models import MODEL_CONFIGS, SEED, fit_model

MONTHS = tuple(valid for _, valid in ROLLING_CUTS)
BASELINE_REFERENCE_GINI = 0.25104430182490023  # LightGBM D, ablation run 20261001T051935820299Z_7e1c7547
BAND_ORDER = {"low": 0, "medium": 1, "high": 2}
MONOTONE_DIRECTIONS = {"banda_riesgo_ord": -1, "numero_productos": 1, "dias_ultima_transaccion": -1}
BAG_SEEDS = (42, 43, 44, 45, 46)
BAG_PARAMS = {"subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}
LOGISTIC_C = 0.1
LOGISTIC_BINS = 10


# ----------------------------------------------------------------- pure helpers
def time_decay_weights(months: pd.Series, half_life: float) -> np.ndarray:
    """Weight 1 for the newest month; halves every ``half_life`` calendar months."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    ordinal = _month_ordinal(pd.Series(months).reset_index(drop=True)).to_numpy()
    return 0.5 ** ((ordinal.max() - ordinal) / half_life)


def within_month_rank(values: pd.Series, months: pd.Series) -> np.ndarray:
    """Percentile rank in (0, 1] computed inside each month only."""
    frame = pd.DataFrame({"v": np.asarray(values, dtype=float), "m": np.asarray(months)})
    return frame.groupby("m")["v"].rank(pct=True, method="average").to_numpy()


def rank_average(predictions: list[np.ndarray], months: pd.Series) -> np.ndarray:
    """Equal-weight mean of per-month percentile ranks (monotone-invariant)."""
    if not predictions:
        raise ValueError("at least one member is required")
    ranks = [within_month_rank(pd.Series(p), months) for p in predictions]
    return np.mean(ranks, axis=0)


def monotone_vector(columns: list[str], directions: dict[str, int]) -> list[int]:
    """LightGBM monotone-constraint vector aligned with the feature order."""
    missing = set(directions) - set(columns)
    if missing:
        raise ValueError(f"constrained features missing from matrix: {sorted(missing)}")
    return [int(directions.get(column, 0)) for column in columns]


def recent_window_mask(months: pd.Series, train_through: int, n_months: int) -> np.ndarray:
    ordinal = _month_ordinal(pd.Series(months).reset_index(drop=True)).to_numpy()
    cut = int(_month_ordinal(pd.Series([train_through])).iloc[0])
    return (ordinal <= cut) & (ordinal > cut - n_months)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


# ------------------------------------------------------------------ candidates
@dataclass(frozen=True)
class Spec:
    name: str
    hypothesis: str
    description: str
    model: str  # lightgbm | catboost | logistic
    features: str  # D_absolute | A
    transform: str = "none"  # none | drop_interaccion | rank_interaccion | drop_both | ordinal_band
    half_life: float | None = None
    window: int | None = None
    config_extra: dict = field(default_factory=dict)
    monotone: bool = False
    seeds: tuple[int, ...] = (SEED,)


def _transform(matrix: pd.DataFrame, frame: pd.DataFrame, how: str) -> pd.DataFrame:
    out = matrix.copy()
    if how == "none":
        return out
    if how == "drop_interaccion":
        return out.drop(columns=["dias_ultima_interaccion"])
    if how == "drop_both":
        return out.drop(columns=["dias_ultima_interaccion", "dias_ultima_transaccion"])
    if how == "rank_interaccion":
        out["dias_ultima_interaccion"] = within_month_rank(out["dias_ultima_interaccion"], frame["mes"])
        return out
    if how == "ordinal_band":
        position = out.columns.get_loc("banda_riesgo")
        out.insert(position, "banda_riesgo_ord", out.pop("banda_riesgo").map(BAND_ORDER).astype(int))
        return out
    raise ValueError(f"unknown transform {how}")


def _logistic_pipeline(matrix: pd.DataFrame) -> Pipeline:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in matrix.columns]
    numeric = [c for c in matrix.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("bins", KBinsDiscretizer(n_bins=LOGISTIC_BINS, encode="onehot", strategy="quantile")),
    ])
    columns = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("features", columns), ("model", LogisticRegression(C=LOGISTIC_C, max_iter=2000))])


def _prepare_logistic(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in out.columns:
            out[column] = out[column].fillna("__UNKNOWN__").astype(str)
    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype(int)
    return out


def predict_folds(spec: Spec, matrix: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold predictions for Sep/Oct/Nov with fixed structural settings."""
    matrix = _transform(matrix, frame, spec.transform)
    pieces = []
    for train_through, valid_month in ROLLING_CUTS:
        in_window = frame["mes"].le(train_through).to_numpy()
        if spec.window is not None:
            in_window = in_window & recent_window_mask(frame["mes"], train_through, spec.window)
        valid_mask = frame["mes"].eq(valid_month).to_numpy()
        x_train = matrix.loc[in_window].reset_index(drop=True)
        y_train = frame.loc[in_window, "objetivo"].astype(int).to_numpy()
        x_valid = matrix.loc[valid_mask].reset_index(drop=True)
        weights = None
        if spec.half_life is not None:
            weights = time_decay_weights(frame.loc[in_window, "mes"], spec.half_life)
        if spec.model == "logistic":
            pipeline = _logistic_pipeline(matrix)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(_prepare_logistic(x_train), y_train)
                prediction = pipeline.predict_proba(_prepare_logistic(x_valid))[:, 1]
        else:
            config = dict(MODEL_CONFIGS[spec.model][0])
            config.update(spec.config_extra)
            if spec.monotone:
                config["monotone_constraints"] = monotone_vector(list(matrix.columns), MONOTONE_DIRECTIONS)
            runs = []
            for seed in spec.seeds:
                fitted = fit_model(
                    spec.model, x_train, y_train, config=config, seed=seed,
                    iterations=FINAL_ITERATIONS[spec.model], early_stopping_rounds=None,
                    sample_weight=weights,
                )
                runs.append(fitted.predict_proba(x_valid))
            prediction = np.mean(runs, axis=0)
        piece = frame.loc[valid_mask, ["id_cliente", "mes", "objetivo"]].reset_index(drop=True)
        piece["prediccion"] = prediction
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


# ------------------------------------------------------------------- evaluation
def fold_ginis(frame: pd.DataFrame) -> list[float]:
    return [
        gini_score(part["objetivo"], part["prediccion"])
        for _, part in frame.groupby("mes", sort=True)
    ]


def paired_bootstrap(frames: dict[str, pd.DataFrame], baseline: str, replicates: int, seed: int = SEED) -> pd.DataFrame:
    """Paired client bootstrap (one shared client draw per replicate across months and models)."""
    reference = frames[baseline]
    ids = pd.Index(reference["id_cliente"].unique())
    customer = ids.get_indexer(reference["id_cliente"])
    names = list(frames)
    month_masks = [reference["mes"].eq(m).to_numpy() for m in MONTHS]
    functions = []
    for name in names:
        f = frames[name]
        functions.append([
            _weighted_gini_preparation(
                f.loc[mask, "objetivo"].to_numpy(), f.loc[mask, "prediccion"].to_numpy(), customer[mask]
            )
            for mask in month_masks
        ])
    draws = np.empty((replicates, len(names), len(MONTHS)))
    rng = np.random.default_rng(seed)
    for r in range(replicates):
        counts = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        for i, funcs in enumerate(functions):
            draws[r, i] = [func(counts) for func in funcs]
    if not np.isfinite(draws).all():
        raise ValueError("Bootstrap replicate lacks a class; cannot silently discard draws")
    base_index = names.index(baseline)
    point = {n: np.array(fold_ginis(frames[n])) for n in names}
    rows = []
    for i, name in enumerate(names):
        if i == base_index:
            continue
        delta = draws[:, i] - draws[:, base_index]
        rolling = delta.mean(axis=1)
        observed = point[name] - point[baseline]
        observed_mean = float(observed.mean())
        row = {"candidate": name}
        for j, label in enumerate(("sep", "oct", "nov")):
            lo, hi = np.quantile(delta[:, j], [0.025, 0.975])
            row.update({f"delta_{label}": float(observed[j]), f"ci_low_{label}": float(lo), f"ci_high_{label}": float(hi)})
        lo, hi = np.quantile(rolling, [0.025, 0.975])
        row.update({
            "delta_mean": observed_mean, "ci_low_mean": float(lo), "ci_high_mean": float(hi),
            "p_two_sided": float((1 + np.sum(np.abs(rolling - observed_mean) >= abs(observed_mean))) / (replicates + 1)),
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_two_sided"].tolist())
    return result


# --------------------------------------------------------------------- the run
def _specs_stage_one() -> list[Spec]:
    lgb = dict(model="lightgbm", features="D_absolute")
    cat = dict(model="catboost", features="A")
    specs = [
        Spec("B0_lgbm_D", "baseline", "LightGBM D (history + month)", **lgb),
        Spec("B1_catboost_A", "baseline", "CatBoost A (base features)", **cat),
    ]
    for tag, how, text in (
        ("H1a", "drop_interaccion", "drop dias_ultima_interaccion"),
        ("H1b", "rank_interaccion", "dias_ultima_interaccion -> within-month percentile rank"),
        ("H1c", "drop_both", "drop both dias_ultima_* columns"),
    ):
        specs.append(Spec(f"{tag}_lgbm", "H1 drift", f"LightGBM D: {text}", transform=how, **lgb))
        specs.append(Spec(f"{tag}_catboost", "H1 drift", f"CatBoost A: {text}", transform=how, **cat))
    return specs


def _best(summary: dict[str, float], names: list[str]) -> str:
    return max(names, key=lambda n: summary[n])


def run_hypotheses(
    data_dir: str | Path = "data",
    experiment_root: str | Path = "experiments/hypotheses",
    seed: int = SEED,
    replicates: int = 2000,
) -> dict:
    data_dir, experiment_root = Path(data_dir), Path(experiment_root)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["objetivo"] = train["objetivo"].astype(int)
    validate_competition_data(train, test)
    train["_source_row"] = np.arange(len(train), dtype=np.int64)
    matrices = {key: _feature_matrix(train, key) for key in ("D_absolute", "A")}

    results: dict[str, pd.DataFrame] = {}
    specs: dict[str, Spec] = {}
    means: dict[str, float] = {}

    def evaluate(spec: Spec) -> None:
        print(f"[hypotheses] {spec.name}", flush=True)
        frame = predict_folds(spec, matrices[spec.features], train)
        results[spec.name], specs[spec.name] = frame, spec
        means[spec.name] = float(np.mean(fold_ginis(frame)))
        print(f"    mean Gini {means[spec.name]:.5f}", flush=True)

    for spec in _specs_stage_one():
        evaluate(spec)
    eligible = [n for n in specs if specs[n].hypothesis in ("baseline", "H1 drift")]
    h2_base_name = _best(means, eligible)
    h2_base = specs[h2_base_name]
    for tag, half_life, window in (("H2a", 3, None), ("H2b", 6, None), ("H2c", None, 6)):
        text = (f"time-decay weights, half-life {half_life} months" if half_life
                else f"train window = last {window} months")
        evaluate(replace(h2_base, name=f"{tag}_{h2_base_name}", hypothesis="H2 recency",
                         description=f"{h2_base_name} + {text}", half_life=half_life, window=window))

    evaluate(Spec("H3a_logistic", "H3 robustness",
                  f"Logistic C={LOGISTIC_C}, {LOGISTIC_BINS} quantile bins + one-hot (features A)",
                  model="logistic", features="A"))
    evaluate(Spec("H3b_lgbm_monotone", "H3 robustness",
                  "LightGBM D with monotone constraints (banda_riesgo_ord -, numero_productos +, dias_ultima_transaccion -)",
                  model="lightgbm", features="D_absolute", transform="ordinal_band", monotone=True))

    best_lgb = _best(means, [n for n in specs if specs[n].model == "lightgbm"])
    best_cat = _best(means, [n for n in specs if specs[n].model == "catboost"])
    for tag, name in (("lgbm", best_lgb), ("catboost", best_cat)):
        base = specs[name]
        extra = {**base.config_extra, **(BAG_PARAMS if base.model == "lightgbm" else {})}
        evaluate(replace(base, name=f"H4a_bag5_{tag}", hypothesis="H4 ensembles",
                         description=f"{name}, 5 seeds averaged" + (" (subsample 0.8 / colsample 0.8)" if tag == "lgbm" else ""),
                         config_extra=extra, seeds=BAG_SEEDS))

    months = results["B0_lgbm_D"]["mes"]
    reference = results["B0_lgbm_D"]
    for name, members, text in (
        ("H4b2_rank_lgbm_catboost", [best_lgb, best_cat], f"rank-average {best_lgb} + {best_cat}"),
        ("H4b3_rank_lgbm_catboost_logistic", [best_lgb, best_cat, "H3a_logistic"],
         f"rank-average {best_lgb} + {best_cat} + H3a_logistic"),
        ("H4b2bag_rank_bagged", ["H4a_bag5_lgbm", "H4a_bag5_catboost"], "rank-average of the two 5-seed bags"),
    ):
        for member in members:
            if not results[member][["id_cliente", "mes"]].equals(reference[["id_cliente", "mes"]]):
                raise ValueError("candidate rows are not aligned")
        frame = reference[["id_cliente", "mes", "objetivo"]].copy()
        frame["prediccion"] = rank_average([results[m]["prediccion"].to_numpy() for m in members], months)
        results[name] = frame
        specs[name] = Spec(name, "H4 ensembles", text, model="blend", features="-")
        means[name] = float(np.mean(fold_ginis(frame)))
        print(f"[hypotheses] {name}\n    mean Gini {means[name]:.5f}", flush=True)

    bootstrap = paired_bootstrap(results, "B0_lgbm_D", replicates, seed)
    summary = _summary_table(results, specs, bootstrap)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{sha256_file(train_path)[:8]}"
    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    oof = reference[["id_cliente", "mes", "objetivo"]].copy()
    for name, frame in results.items():
        oof[name] = frame["prediccion"].to_numpy().round(7)
    oof_path = run_dir / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    prior = _direction_evidence(train)
    b0_gap = float(summary.loc[summary["candidate"] == "B0_lgbm_D", "mean_gini"].iloc[0] - BASELINE_REFERENCE_GINI)
    report_path = run_dir / "results.md"
    report_path.write_text(
        _report(run_id, summary, h2_base_name, best_lgb, best_cat, b0_gap, replicates, prior), encoding="utf-8")
    code_paths = [Path("src/datafest") / n for n in ("hypotheses.py", "models.py", "ablation.py", "diagnostics.py",
                                                      "features.py", "metrics.py", "final_fit.py", "lineage.py")]
    manifest = {
        "schema_version": 1, "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository_revision(),
        "protocol": {
            "rolling_cuts": [{"train_through": a, "valid_month": b} for a, b in ROLLING_CUTS],
            "tuning_or_early_stopping_on_validation": False,
            "iterations": FINAL_ITERATIONS, "structural_config": "MODEL_CONFIGS[model][0] (baseline)",
            "bootstrap": {"unit": "id_cliente", "resamples": replicates, "seed": seed,
                          "p_value": "two-sided centered, plus-one", "multiplicity": "Holm across all candidates vs B0"},
            "h2_base": h2_base_name, "h4_best_lgbm": best_lgb, "h4_best_catboost": best_cat,
            "selection_note": "'best of' choices use mean rolling Gini on Sep-Nov; downstream comparisons are exploratory",
            "monotone_directions": MONOTONE_DIRECTIONS, "monotone_direction_evidence": prior,
            "logistic": {"C": LOGISTIC_C, "bins": LOGISTIC_BINS}, "bag_seeds": BAG_SEEDS, "bag_params": BAG_PARAMS,
            "b0_minus_reference_mean_gini": b0_gap,
        },
        "candidates": {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.__dict__.items()}
                       for n, s in specs.items()},
        "inputs": [file_record(train_path), file_record(test_path)],
        "code": [file_record(p) for p in code_paths if p.is_file()],
        "outputs": [file_record(summary_path), file_record(report_path), file_record(oof_path)],
        "environment": {"python": platform.python_version(), "datafest": __version__},
    }
    write_json(run_dir / "manifest.json", manifest)
    return {"run_id": run_id, "run_dir": str(run_dir), "summary": str(summary_path),
            "b0_minus_reference": b0_gap, "best": summary.iloc[0]["candidate"]}


def _direction_evidence(train: pd.DataFrame) -> dict:
    """Spearman(feature, objetivo) on months <= Aug only: the a priori monotone direction check."""
    early = train[train["mes"].le(202608)]
    band = early["banda_riesgo"].map(BAND_ORDER)
    out = {"banda_riesgo_ord": band, "numero_productos": early["numero_productos"],
           "dias_ultima_transaccion": early["dias_ultima_transaccion"]}
    return {k: float(spearmanr(v, early["objetivo"]).statistic) for k, v in out.items()}


def _summary_table(results: dict, specs: dict, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in results.items():
        g = fold_ginis(frame)
        rows.append({"candidate": name, "hypothesis": specs[name].hypothesis, "description": specs[name].description,
                     "gini_sep": g[0], "gini_oct": g[1], "gini_nov": g[2],
                     "mean_gini": float(np.mean(g)), "std_gini": float(np.std(g))})
    table = pd.DataFrame(rows).merge(bootstrap, on="candidate", how="left")
    return table.sort_values("mean_gini", ascending=False).reset_index(drop=True)


def _report(run_id, summary, h2_base, best_lgb, best_cat, b0_gap, replicates, prior) -> str:
    lines = [
        f"# Hypothesis evaluation `{run_id}`", "",
        "Rolling folds: train <=Aug->Sep, <=Sep->Oct, <=Oct->Nov. Fixed structural params and frozen iterations "
        "(LightGBM 58, CatBoost 192); no early stopping or tuning on any validation month. "
        f"Paired client bootstrap ({replicates} resamples, shared client draw across months and models) vs B0 = LightGBM D; "
        "p-values are two-sided centered with Holm adjustment across all candidates.", "",
        f"B0 reproduction: mean Gini differs from the ablation reference 0.25104430 by {b0_gap:+.2e}.", "",
        "| Candidate | Sep | Oct | Nov | Mean | Std | dMean vs B0 [95% CI] | dSep [CI] | dOct [CI] | dNov [CI] | p Holm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |",
    ]

    def ci(row, label):
        if pd.isna(row[f"delta_{label}"]):
            return "-"
        return f"{row[f'delta_{label}']:+.4f} [{row[f'ci_low_{label}']:+.4f}, {row[f'ci_high_{label}']:+.4f}]"

    for _, r in summary.iterrows():
        holm = "-" if pd.isna(r["p_holm"]) else f"{r['p_holm']:.3f}"
        lines.append(
            f"| {r['candidate']} | {r['gini_sep']:.5f} | {r['gini_oct']:.5f} | {r['gini_nov']:.5f} | "
            f"{r['mean_gini']:.5f} | {r['std_gini']:.5f} | {ci(r, 'mean')} | {ci(r, 'sep')} | {ci(r, 'oct')} | "
            f"{ci(r, 'nov')} | {holm} |")
    significant = summary[(summary["p_holm"] < 0.05) & (summary["delta_mean"] > 0)]
    top = summary.iloc[0]
    lines += ["", "## Interpretation", "",
              f"- Best mean Gini: `{top['candidate']}` ({top['mean_gini']:.5f}).",
              ("- Candidates significantly better than B0 after Holm: "
               + ", ".join(f"`{c}`" for c in significant["candidate"]) + ".") if len(significant)
              else "- No candidate beats B0 significantly after Holm adjustment; differences are within noise.",
              f"- Selections: H2 applied to `{h2_base}`; H4 used best LightGBM `{best_lgb}` and best CatBoost `{best_cat}`. "
              "These picks use validation-month means, so H2/H4 comparisons are exploratory (optimistic).",
              "- Monotone directions were checked a priori on months <=Aug (Spearman with objetivo): "
              + ", ".join(f"{k} {v:+.3f}" for k, v in prior.items()) + ".",
              "- LightGBM bagging adds subsample 0.8 / colsample 0.8 (the baseline config has no randomness, so seeds would "
              "otherwise be identical)."]
    return "\n".join(lines) + "\n"
