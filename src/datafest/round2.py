"""Round-2 hypothesis run: feature families (F1-F5) and GBDT upgrades (XGBoost, Optuna).

Reuses the round-1 harness unchanged in spirit: ``hypotheses.predict_folds`` (rolling
Sep/Oct/Nov folds, frozen iterations, no early stopping on a validation month),
``hypotheses.paired_bootstrap`` (shared client draws) and Holm adjustment. Every
candidate is compared against BOTH B0 (LightGBM D) and H4b2 (the champion blend).

Selection discipline
--------------------
* Every selection (best feature set per model, which families "helped", XGBoost
  iteration count, Optuna objective) uses only inner temporal folds whose validation
  months are <= August 2026. Sep/Oct/Nov are only used to REPORT.
* Even so, ~50 candidates are scored on Sep-Nov and the best of them is reported, so
  the maximum rolling mean is optimistic; read it with the bootstrap CIs and Holm p.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import pickle
import platform
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable
import warnings

import lightgbm
import catboost
import numpy as np
import optuna
import pandas as pd
import xgboost

from datafest import __version__
from datafest.ablation import ROLLING_CUTS, _feature_matrix
from datafest.calendar_features import PERU_HOLIDAYS_2026
from datafest.data import validate_competition_data
from datafest.final_fit import FINAL_ITERATIONS
from datafest.hypotheses import (
    BASELINE_REFERENCE_GINI, Spec, fold_ginis, paired_bootstrap, predict_folds, rank_average,
)
from datafest.lineage import file_record, repository_revision, sha256_file, write_json
from datafest.models import MODEL_CONFIGS, SEED, fit_model, xgboost_device
from datafest.round2_features import FAMILY_NAMES, family_extras

INNER_END = 202608
B0 = "B0_lgbm_D"
CAT_BASE = "H1b_catboost"
H4B2 = "H4b2_rank_lgbm_catboost"
XGB_EARLY_STOPPING = 50
XGB_MAX_ROUNDS = 1500
WORKERS = 3  # concurrent fits; every model keeps thread_count/n_jobs = 4 (CatBoost results depend on thread count)
UNION_EXCLUDED = ("F3r_rank_replace",)  # a column-replacing family cannot be unioned with others
FAMILY_LABELS = {
    "F1_calendar": "F1 calendar", "F1g_calendar_flags": "F1 calendar+flags", "F2_ratios": "F2 ratios",
    "F3_rank_all": "F3 ranks (add)", "F3r_rank_replace": "F3 ranks (replace)",
    "F4_target_enc": "F4 causal target enc", "F5_cohort": "F5 cohort/hazard",
}
MODEL_BASES = {  # prefix -> (model, base feature key, transform on the base)
    "lgbm_D": ("lightgbm", "D_absolute", "none"),
    "cat_A_H1b": ("catboost", "A", "rank_interaccion"),
    "xgb_D": ("xgboost", "D_absolute", "none"),
}


# ------------------------------------------------------------------ pure helpers
def inner_cuts(months: Iterable[int], end_month: int = INNER_END) -> tuple[tuple[int, int], ...]:
    """(train_through, valid_month) pairs of the inner walk-forward folds; validation months <= end_month."""
    usable = sorted({int(m) for m in months if int(m) <= end_month})
    if len(usable) < 5:
        raise ValueError("at least five months are required for inner folds")
    return tuple((max(m for m in usable if m < valid), valid) for valid in usable[-4:])


def select_by_inner(inner_means: dict[str, float], names: list[str]) -> str:
    return max(names, key=lambda n: inner_means[n])


def union_of_helpers(inner_means: dict[str, float], base: str, families: list[str]) -> list[str]:
    """Families whose inner-fold mean Gini beats the base (strictly), in input order."""
    return [f for f in families if inner_means[f] > inner_means[base]]


def suggest_params(model: str, trial: optuna.Trial) -> tuple[dict, int]:
    """Search space (config overrides, boosting iterations); iterations are searched, never early-stopped."""
    if model == "lightgbm":
        config = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        return config, trial.suggest_int("iterations", 20, 400, log=True)
    if model == "catboost":
        config = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        }
        return config, trial.suggest_int("iterations", 100, 500, log=True)
    raise ValueError(f"no search space for {model}")


@dataclass
class TuneResult:
    config_extra: dict
    iterations: int
    best_value: float
    trials: pd.DataFrame


def tune_model(model: str, base_spec: Spec, matrix: pd.DataFrame, frame: pd.DataFrame,
               n_trials: int, seed: int, cuts) -> TuneResult:
    """TPE search maximising the mean Gini over ``cuts`` (callers pass inner folds <= August only)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        config, iterations = suggest_params(model, trial)
        spec = replace(base_spec, config_extra={**base_spec.config_extra, **config}, iterations=iterations)
        value = float(np.mean(fold_ginis(predict_folds(spec, matrix, frame, cuts=cuts))))
        rows.append({"trial": trial.number, "value": value, **trial.params})
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_params = dict(study.best_trial.params)
    iterations = int(best_params.pop("iterations"))
    config = {**suggest_params(model, optuna.trial.FixedTrial({**best_params, "iterations": iterations}))[0]}
    return TuneResult(config, iterations, float(study.best_value), pd.DataFrame(rows))


def candidate_specs(xgb_iterations: int = 100) -> list[Spec]:
    """B0, the H1b CatBoost control, and every family add-on for LightGBM D, CatBoost A-H1b and XGBoost D."""
    specs = [
        Spec(B0, "baseline", "LightGBM D (history + month)", model="lightgbm", features="D_absolute"),
        Spec(CAT_BASE, "baseline", "CatBoost A + within-month rank of dias_ultima_interaccion",
             model="catboost", features="A", transform="rank_interaccion"),
        Spec("xgb_D", "G1 xgboost", "XGBoost hist on feature set D", model="xgboost", features="D_absolute",
             iterations=xgb_iterations),
    ]
    for family in FAMILY_NAMES:
        label = FAMILY_LABELS[family]
        for prefix, (model, base_key, transform) in MODEL_BASES.items():
            if family == "F3r_rank_replace" and transform == "rank_interaccion":
                transform = "none"  # raw dias_ultima_interaccion is replaced by its rank already
            group = "G1 xgboost" if model == "xgboost" else label
            specs.append(Spec(
                f"{prefix}+{family}", group, f"{model} on {base_key} + {label}", model=model,
                features=f"{base_key}+{family}", transform=transform,
                iterations=xgb_iterations if model == "xgboost" else None))
    return specs


def apply_families(names: list[str], frame: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Append several families' columns to ``matrix`` (later duplicates of a column are skipped)."""
    out = matrix.reset_index(drop=True)
    data = frame.reset_index(drop=True)
    for name in names:
        extra, drop = family_extras(name, data, out)
        out = out.drop(columns=drop)
        extra = extra.loc[:, [c for c in extra.columns if c not in out.columns]]
        out = pd.concat([out, extra.reset_index(drop=True)], axis=1)
    return out


def select_xgb_iterations(matrix: pd.DataFrame, frame: pd.DataFrame, cuts) -> tuple[int, list[int]]:
    """Median early-stopped round count over the inner folds (validation months <= August)."""
    best = []
    for train_through, valid_month in cuts:
        train_mask = frame["mes"].le(train_through).to_numpy()
        valid_mask = frame["mes"].eq(valid_month).to_numpy()
        fitted = fit_model(
            "xgboost", matrix.loc[train_mask].reset_index(drop=True), frame.loc[train_mask, "objetivo"].astype(int),
            valid_x=matrix.loc[valid_mask].reset_index(drop=True), valid_y=frame.loc[valid_mask, "objetivo"].astype(int),
            iterations=XGB_MAX_ROUNDS, early_stopping_rounds=XGB_EARLY_STOPPING,
        )
        best.append(fitted.best_iteration)
    return max(1, int(statistics.median(best))), best


# ------------------------------------------------------------------------- run
def _blend(results: dict[str, pd.DataFrame], members: list[str]) -> pd.DataFrame:
    reference = results[B0]
    for member in members:
        if not results[member][["id_cliente", "mes"]].equals(reference[["id_cliente", "mes"]]):
            raise ValueError("candidate rows are not aligned")
    frame = reference[["id_cliente", "mes", "objetivo"]].copy()
    frame["prediccion"] = rank_average([results[m]["prediccion"].to_numpy() for m in members], reference["mes"])
    return frame


def build_matrices(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = {key: _feature_matrix(train, key) for key in ("D_absolute", "A")}
    matrices = dict(base)
    for family in FAMILY_NAMES:
        for key in base:
            matrices[f"{key}+{family}"] = apply_families([family], train, base[key])
    return matrices


def compare_to_baselines(results: dict[str, pd.DataFrame], baselines: tuple[str, str], replicates: int,
                         seed: int) -> pd.DataFrame:
    """Paired bootstrap against each baseline; columns suffixed ``_b0`` / ``_h4``."""
    table = pd.DataFrame({"candidate": list(results)})
    for suffix, baseline in zip(("b0", "h4"), baselines):
        boot = paired_bootstrap(results, baseline, replicates, seed)
        keep = ["candidate", "delta_mean", "ci_low_mean", "ci_high_mean", "p_holm",
                "delta_sep", "delta_oct", "delta_nov"]
        renamed = boot[keep].rename(columns={c: f"{c}_{suffix}" for c in keep if c != "candidate"})
        table = table.merge(renamed, on="candidate", how="left")
    return table


def run_round2(
    data_dir: str | Path = "data",
    experiment_root: str | Path = "experiments/hypotheses",
    seed: int = SEED,
    replicates: int = 2000,
    trials_lgbm: int = 60,
    trials_catboost: int = 40,
    cache_dir: str | Path | None = None,
) -> dict:
    warnings.filterwarnings("ignore", message=".*mismatched devices.*")
    data_dir, experiment_root = Path(data_dir), Path(experiment_root)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["objetivo"] = train["objetivo"].astype(int)
    validate_competition_data(train, test)
    train["_source_row"] = np.arange(len(train), dtype=np.int64)
    matrices = build_matrices(train)
    months = sorted(int(m) for m in train["mes"].unique())
    inner = inner_cuts(months)
    tune_cuts = inner[1:]  # validation months Jun, Jul, Aug
    assert max(v for _, v in inner) <= INNER_END

    xgb_iterations, xgb_rounds = select_xgb_iterations(matrices["D_absolute"], train, inner)
    print(f"[round2] XGBoost device={xgboost_device()} iterations={xgb_iterations} (inner folds {xgb_rounds})", flush=True)

    results: dict[str, pd.DataFrame] = {}
    specs: dict[str, Spec] = {}
    rolling_means: dict[str, float] = {}
    inner_means: dict[str, float] = {}

    cache_root = Path(cache_dir) if cache_dir else None
    train_hash = sha256_file(train_path)

    def score(spec: Spec, with_inner: bool = True) -> tuple[float | None, pd.DataFrame]:
        """Inner mean (optional) and rolling OOF; optionally checkpointed so a crashed run can resume.

        The checkpoint key hashes the full spec, data hash and code of the harness, so a stale
        result is never reused after any of them changes.
        """
        path = None
        if cache_root is not None:
            code = "".join(sha256_file(Path("src/datafest") / n) for n in (
                "round2.py", "round2_features.py", "calendar_features.py", "hypotheses.py", "models.py"))
            key = hashlib.sha256(f"{spec!r}|{with_inner}|{train_hash}|{code}".encode()).hexdigest()[:20]
            path = cache_root / f"{spec.name.replace('|', '_').replace('+', '_')}_{key}.pkl"
            if path.is_file():
                print(f"[round2] {spec.name}: loaded from checkpoint", flush=True)
                return pickle.loads(path.read_bytes())
        matrix = matrices[spec.features]
        inner_mean = None
        if with_inner:
            inner_mean = float(np.mean(fold_ginis(predict_folds(spec, matrix, train, cuts=inner))))
        frame = predict_folds(spec, matrix, train)
        print(f"[round2] {spec.name}: inner(<=Aug) {inner_mean if inner_mean is not None else float('nan'):.5f}  "
              f"rolling {np.mean(fold_ginis(frame)):.5f}", flush=True)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps((inner_mean, frame)))
        return inner_mean, frame

    def evaluate_many(batch: list[Spec], with_inner: bool = True) -> None:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            outcomes = list(pool.map(lambda sp: score(sp, with_inner), batch))
        for spec, (inner_mean, frame) in zip(batch, outcomes):  # insertion order stays deterministic
            if inner_mean is not None:
                inner_means[spec.name] = inner_mean
            results[spec.name], specs[spec.name] = frame, spec
            rolling_means[spec.name] = float(np.mean(fold_ginis(frame)))

    evaluate_many(candidate_specs(xgb_iterations))

    # H4b2 control (needs only B0 and H1b CatBoost, both evaluated above).
    results[H4B2] = _blend(results, [B0, CAT_BASE])
    specs[H4B2] = Spec(H4B2, "baseline", f"rank-average {B0} + {CAT_BASE} (round-1 champion)", model="blend", features="-")
    rolling_means[H4B2] = float(np.mean(fold_ginis(results[H4B2])))

    # G3: union of the families that individually helped on the inner folds (<= Aug).
    selected: dict[str, str] = {}
    unions: dict[str, list[str]] = {}
    union_specs: list[Spec] = []
    for prefix, (model, base_key, transform) in MODEL_BASES.items():
        base_name = {"lgbm_D": B0, "cat_A_H1b": CAT_BASE, "xgb_D": "xgb_D"}[prefix]
        family_names = {f: f"{prefix}+{f}" for f in FAMILY_NAMES if f not in UNION_EXCLUDED}
        helpers = union_of_helpers(inner_means, base_name, list(family_names.values()))
        helper_families = [f for f, n in family_names.items() if n in helpers]
        if "F1_calendar" in helper_families and "F1g_calendar_flags" in helper_families:
            helper_families.remove("F1_calendar")  # F1g already contains F1
        unions[prefix] = helper_families
        if helper_families:
            key = f"{prefix}|{base_key}+UNION"
            matrices[key] = apply_families(helper_families, train, matrices[base_key])
            union_specs.append(Spec(f"{prefix}+UNION", "G3 union",
                                    f"{model} on {base_key} + union of inner-fold helpers {helper_families}",
                                    model=model, features=key, transform=transform,
                                    iterations=xgb_iterations if model == "xgboost" else None))
    evaluate_many(union_specs)
    for prefix, (model, base_key, transform) in MODEL_BASES.items():
        base_name = {"lgbm_D": B0, "cat_A_H1b": CAT_BASE, "xgb_D": "xgb_D"}[prefix]
        pool_names = [base_name] + [f"{prefix}+{f}" for f in FAMILY_NAMES if f not in UNION_EXCLUDED]
        pool_names += [f"{prefix}+UNION"] if unions[prefix] else []
        selected[prefix] = select_by_inner(inner_means, pool_names)
    print(f"[round2] inner-fold selection: {selected}; unions: {unions}", flush=True)

    # G2: Optuna on inner folds Jun/Jul/Aug only, for the base and the inner-selected feature set.
    # Studies run concurrently (one thread each); every study is internally sequential and seeded.
    tune_jobs = []
    for prefix, trials in (("lgbm_D", trials_lgbm), ("cat_A_H1b", trials_catboost)):
        base_name = B0 if prefix == "lgbm_D" else CAT_BASE
        for source in dict.fromkeys([base_name, selected[prefix]]):
            tune_jobs.append((MODEL_BASES[prefix][0], source, trials))

    def run_study(job):
        model, source, trials = job
        spec = specs[source]
        print(f"[round2] Optuna {model} on {source}: {trials} trials, objective = mean Gini over {tune_cuts}", flush=True)
        path = None
        if cache_root is not None:
            code = "".join(sha256_file(Path("src/datafest") / n) for n in (
                "round2.py", "round2_features.py", "calendar_features.py", "hypotheses.py", "models.py"))
            key = hashlib.sha256(f"{spec!r}|{trials}|{seed}|{tune_cuts}|{train_hash}|{code}".encode()).hexdigest()[:20]
            path = cache_root / f"optuna_{source.replace('|', '_').replace('+', '_')}_{key}.pkl"
            if path.is_file():
                print(f"[round2] Optuna {model} on {source}: loaded from checkpoint", flush=True)
                return pickle.loads(path.read_bytes())
        tuned = tune_model(model, spec, matrices[spec.features], train, trials, seed, tune_cuts)
        base_objective = float(np.mean(fold_ginis(predict_folds(spec, matrices[spec.features], train, cuts=tune_cuts))))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps((tuned, base_objective)))
        print(f"[round2] Optuna {model} on {source} done: best {tuned.best_value:.5f} vs base config {base_objective:.5f}", flush=True)
        return tuned, base_objective

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        study_outcomes = list(pool.map(run_study, tune_jobs))
    studies: dict[str, pd.DataFrame] = {}
    tuned_info: dict[str, dict] = {}
    tuned_specs = []
    for (model, source, trials), (tuned, base_objective) in zip(tune_jobs, study_outcomes):
        spec = specs[source]
        name = f"tuned_{source}"
        tuned_specs.append(replace(spec, name=name, hypothesis="G2 optuna",
                                   description=f"{source} with Optuna-tuned config ({trials} trials, inner folds <=Aug)",
                                   config_extra={**spec.config_extra, **tuned.config_extra}, iterations=tuned.iterations))
        studies[name] = tuned.trials
        tuned_info[name] = {"source": source, "config": tuned.config_extra, "iterations": tuned.iterations,
                            "objective_jun_aug": tuned.best_value, "trials": trials,
                            "base_objective_jun_aug": base_objective}
    evaluate_many(tuned_specs, with_inner=False)
    for spec in tuned_specs:
        inner_means[spec.name] = tuned_info[spec.name]["objective_jun_aug"]  # search target: optimistic

    # Blends of the inner-selected / tuned members.
    lgb_pick, cat_pick, xgb_pick = selected["lgbm_D"], selected["cat_A_H1b"], selected["xgb_D"]
    tuned_lgb, tuned_cat = f"tuned_{lgb_pick}", f"tuned_{cat_pick}"
    for name, members, text in (
        ("E1_rank_inner_selected", [lgb_pick, cat_pick], f"rank-average {lgb_pick} + {cat_pick}"),
        ("E2_rank_tuned", [tuned_lgb, tuned_cat], f"rank-average {tuned_lgb} + {tuned_cat}"),
        ("E3_rank_tuned_plus_xgb", [tuned_lgb, tuned_cat, xgb_pick], f"rank-average {tuned_lgb} + {tuned_cat} + {xgb_pick}"),
    ):
        results[name] = _blend(results, members)
        specs[name] = Spec(name, "E blend", text, model="blend", features="-")
        rolling_means[name] = float(np.mean(fold_ginis(results[name])))
        print(f"[round2] {name}\n    rolling {rolling_means[name]:.5f}", flush=True)

    comparison = compare_to_baselines(results, (B0, H4B2), replicates, seed)
    summary = _summary(results, specs, inner_means, comparison)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{sha256_file(train_path)[:8]}_r2"
    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    oof = results[B0][["id_cliente", "mes", "objetivo"]].copy()
    for name, frame in results.items():
        oof[name] = frame["prediccion"].to_numpy().round(7)
    oof_path = run_dir / "oof_predictions.csv"  # gitignored (> 5 MB)
    oof.to_csv(oof_path, index=False)
    trials_paths = []
    for name, trials_frame in studies.items():
        path = run_dir / f"optuna_{name}.csv"
        trials_frame.to_csv(path, index=False)
        trials_paths.append(path)
    b0_gap = float(summary.loc[summary["candidate"] == B0, "mean_gini"].iloc[0] - BASELINE_REFERENCE_GINI)
    report_path = run_dir / "results.md"
    report_path.write_text(_report(run_id, summary, selected, unions, tuned_info, xgb_iterations, xgb_rounds,
                                   replicates, b0_gap), encoding="utf-8")
    code_paths = [Path("src/datafest") / n for n in (
        "round2.py", "round2_features.py", "calendar_features.py", "hypotheses.py", "models.py", "ablation.py",
        "features.py", "metrics.py", "final_fit.py", "lineage.py", "diagnostics.py")] + [Path("pyproject.toml"), Path("uv.lock")]
    manifest = {
        "schema_version": 1, "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository_revision(),
        "protocol": {
            "rolling_cuts": [{"train_through": a, "valid_month": b} for a, b in ROLLING_CUTS],
            "inner_cuts_selection": [{"train_through": a, "valid_month": b} for a, b in inner],
            "optuna_cuts": [{"train_through": a, "valid_month": b} for a, b in tune_cuts],
            "selection_data": "inner folds with validation months <= 202608 only (feature-set pick, helper union, "
                              "XGBoost iterations, Optuna objective)",
            "validation_months_used_for_search_or_early_stopping": False,
            "iterations_frozen": {**FINAL_ITERATIONS, "xgboost": xgb_iterations},
            "xgboost": {"device": xgboost_device(), "config": MODEL_CONFIGS["xgboost"][0],
                        "inner_best_rounds": xgb_rounds, "early_stopping_rounds": XGB_EARLY_STOPPING},
            "optuna": {"sampler": "TPE", "seed": seed, "trials": {"lightgbm": trials_lgbm, "catboost": trials_catboost}},
            "bootstrap": {"unit": "id_cliente", "resamples": replicates, "seed": seed,
                          "p_value": "two-sided centered, plus-one",
                          "multiplicity": "Holm across all candidates, separately per baseline (B0, H4b2)"},
            "holidays_2026": sorted(d.isoformat() for d in PERU_HOLIDAYS_2026),
            "holiday_source": "https://www.gob.pe/feriados (non-working bridge days not included)",
            "selected_by_inner": selected, "union_families": unions, "tuned": tuned_info,
            "b0_minus_reference_mean_gini": b0_gap,
        },
        "candidates": {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.__dict__.items()}
                       for n, s in specs.items()},
        "inputs": [file_record(train_path), file_record(test_path)],
        "code": [file_record(p) for p in code_paths if p.is_file()],
        "outputs": [file_record(p) for p in [summary_path, report_path, oof_path, *trials_paths]],
        "environment": {"python": platform.python_version(), "datafest": __version__,
                        "xgboost": xgboost.__version__, "optuna": optuna.__version__,
                        "lightgbm": lightgbm.__version__, "catboost": catboost.__version__},
    }
    write_json(run_dir / "manifest.json", manifest)
    return {"run_id": run_id, "run_dir": str(run_dir), "summary": str(summary_path), "b0_minus_reference": b0_gap,
            "best": summary.iloc[0]["candidate"], "selected_by_inner": selected}


def _summary(results, specs, inner_means, comparison) -> pd.DataFrame:
    rows = []
    for name, frame in results.items():
        g = fold_ginis(frame)
        rows.append({"candidate": name, "hypothesis": specs[name].hypothesis, "description": specs[name].description,
                     "gini_sep": g[0], "gini_oct": g[1], "gini_nov": g[2], "mean_gini": float(np.mean(g)),
                     "std_gini": float(np.std(g)), "inner_mean_gini": inner_means.get(name, np.nan)})
    table = pd.DataFrame(rows).merge(comparison, on="candidate", how="left")
    return table.sort_values("mean_gini", ascending=False).reset_index(drop=True)


def _cell(row, suffix: str) -> str:
    if pd.isna(row[f"delta_mean_{suffix}"]):
        return "-"
    return (f"{row[f'delta_mean_{suffix}']:+.4f} [{row[f'ci_low_mean_{suffix}']:+.4f}, "
            f"{row[f'ci_high_mean_{suffix}']:+.4f}] p={row[f'p_holm_{suffix}']:.3f}")


def _report(run_id, summary, selected, unions, tuned_info, xgb_iterations, xgb_rounds, replicates, b0_gap) -> str:
    n = len(summary)
    lines = [
        f"# Round-2 hypothesis evaluation `{run_id}`", "",
        "Rolling folds: train <=Aug->Sep, <=Sep->Oct, <=Oct->Nov (reported). Selection used only inner folds with "
        "validation months <= August (train <=Apr->May ... <=Jul->Aug). LightGBM 58 / CatBoost 192 iterations frozen for "
        f"feature add-ons; XGBoost {xgb_iterations} rounds (median early-stopped rounds on inner folds {xgb_rounds}). "
        f"Paired client bootstrap ({replicates} resamples, shared draw across months and models) against B0 (LightGBM D) "
        "and H4b2 (rank-average champion); p = Holm-adjusted two-sided p over all candidates, per baseline.", "",
        f"B0 reproduction: mean Gini differs from the ablation reference 0.25104430 by {b0_gap:+.2e}.", "",
        "| Candidate | Sep | Oct | Nov | Mean | Std | Inner<=Aug | dMean vs B0 [95% CI] p Holm | dMean vs H4b2 [95% CI] p Holm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, r in summary.iterrows():
        inner_cell = "-" if pd.isna(r["inner_mean_gini"]) else f"{r['inner_mean_gini']:.5f}"
        lines.append(f"| {r['candidate']} | {r['gini_sep']:.5f} | {r['gini_oct']:.5f} | {r['gini_nov']:.5f} | "
                     f"{r['mean_gini']:.5f} | {r['std_gini']:.5f} | {inner_cell} | {_cell(r, 'b0')} | {_cell(r, 'h4')} |")
    sig_b0 = summary[(summary["p_holm_b0"] < 0.05) & (summary["delta_mean_b0"] > 0)]["candidate"].tolist()
    sig_h4 = summary[(summary["p_holm_h4"] < 0.05) & (summary["delta_mean_h4"] > 0)]["candidate"].tolist()
    top = summary.iloc[0]
    lines += ["", "## Interpretation", "",
              f"- Best rolling mean: `{top['candidate']}` ({top['mean_gini']:.5f}).",
              "- Significantly better than B0 after Holm: " + (", ".join(f"`{c}`" for c in sig_b0) or "none") + ".",
              "- Significantly better than H4b2 after Holm: " + (", ".join(f"`{c}`" for c in sig_h4) or "none") + ".",
              f"- Inner-fold (<=Aug) picks: {selected}. Helper unions: {unions}.",
              "- Optuna: " + "; ".join(
                  f"`{k}` objective(Jun-Aug) {v['objective_jun_aug']:.5f} vs base config {v['base_objective_jun_aug']:.5f}"
                  for k, v in tuned_info.items()) + ". The Optuna objective is the search target itself, so it is optimistic.",
              "- `gratificacion_month` (Jul, Dec) cannot be validated: no gratificacion month exists in Sep-Nov. `cts_month` "
              "is exercised only by November, with May as the sole training example. Compare F1 with F1g.",
              f"- Selection bias: {n} candidates were scored on Sep-Nov and ranked by their mean; the top rolling mean is the "
              "maximum of many noisy estimates and is optimistic. Selections themselves (which feature set, which helpers, "
              "XGBoost rounds, Optuna configs) used inner folds <= August only. Holm over the whole family is the honest "
              "significance read. Blend members are picked by inner-fold means, not by Sep-Nov."]
    return "\n".join(lines) + "\n"
