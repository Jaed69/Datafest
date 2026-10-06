"""Final fit of the selected ensemble H4b2 and its pre-submission checks.

H4b2 = equal-weight rank-average of LightGBM D (B0) and CatBoost A whose
``dias_ultima_interaccion`` is replaced by its within-month percentile rank
(H1b). Both members use the frozen structural configs and iteration counts and
no early stopping. December test ranks are computed inside December only.

The blend is a rank score, not a probability. Because the metric (Gini) only
depends on order, the submitted values are obtained by quantile mapping the
blend onto the sorted LightGBM December probabilities: the k-th smallest blend
receives the k-th smallest LightGBM probability. Order is preserved exactly.
"""
from __future__ import annotations

from datetime import datetime, timezone
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from datafest import __version__
from datafest.data import validate_competition_data, validate_submission
from datafest.final_fit import FINAL_ITERATIONS, build_final_matrices, validate_final_submission
from datafest.hypotheses import within_month_rank
from datafest.lineage import build_run_manifest, sha256_file, write_json
from datafest.models import MODEL_CONFIGS, SEED, fit_model

MODEL_NAME = "ensemble-h4b2"
INTERACTION = "dias_ultima_interaccion"


def quantile_map_to_reference(blend: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Give the k-th smallest blend value the k-th smallest reference value.

    Tied blend values share the reference value at the first position of the
    tie group (rank method ``min``), so ties stay ties and the result is
    deterministic and independent of row order.
    """
    blend, reference = np.asarray(blend, dtype=float), np.asarray(reference, dtype=float)
    if blend.shape != reference.shape:
        raise ValueError("blend and reference must have the same length")
    positions = rankdata(blend, method="min").astype(int) - 1
    return np.sort(reference)[positions]


def rank_interaction_matrices(
    train_x: pd.DataFrame, train_months: pd.Series, test_x: pd.DataFrame, test_months: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace ``dias_ultima_interaccion`` by its within-month percentile rank.

    Train ranks are per training month; test ranks use the test rows only.
    """
    train_out, test_out = train_x.copy(), test_x.copy()
    train_out[INTERACTION] = within_month_rank(train_x[INTERACTION], train_months)
    test_out[INTERACTION] = within_month_rank(test_x[INTERACTION], test_months)
    return train_out, test_out


def _describe(values: np.ndarray) -> dict:
    return {"rows": int(len(values)), "mean": float(np.mean(values)),
            **{f"p{int(q * 100)}": float(np.quantile(values, q)) for q in (0.1, 0.5, 0.9)}}


def pre_submission_checks(submission: pd.DataFrame, train: pd.DataFrame,
                          compare_with: str | Path | None) -> dict:
    prediction = submission["prediccion"].to_numpy()
    survivor = submission["id_cliente"].isin(train["id_cliente"]).to_numpy()
    new = (submission["id_cliente"] > train["id_cliente"].max()).to_numpy()
    checks = {
        "rows": int(len(submission)), "finite": bool(np.isfinite(prediction).all()),
        "in_unit_interval": bool(((prediction >= 0) & (prediction <= 1)).all()),
        "survivors": _describe(prediction[survivor]), "new_clients": _describe(prediction[new]),
        "n_survivors": int(survivor.sum()), "n_new": int(new.sum()), "n_other": int((~survivor & ~new).sum()),
        "spearman": None,
    }
    if compare_with is not None:
        other = pd.read_csv(compare_with)["prediccion"].to_numpy()

        def rho(mask: np.ndarray) -> float:
            return float(spearmanr(prediction[mask], other[mask]).statistic)

        checks["spearman"] = {"other": str(compare_with), "overall": rho(np.ones(len(prediction), bool)),
                              "survivors": rho(survivor), "new_clients": rho(new)}
    return checks


def _results_md(run_id: str, iterations: dict, checks: dict, overall: dict) -> str:
    lines = [
        f"# Final ensemble {run_id}", "",
        "- Model: H4b2 = equal-weight rank-average of LightGBM D (B0) and CatBoost A with "
        "`dias_ultima_interaccion` -> within-month percentile rank (H1b).",
        f"- Iterations: {iterations} (frozen, no early stopping); seed 42; trained on all Jan-Nov 2026 rows, "
        "scored December 2026.",
        "- CatBoost rank feature: per month in train, within December only for test.",
        "- Selection: rolling mean Gini 0.25245 vs B0 0.25104 (+0.0014, CI [-0.0032, +0.0059]); "
        "not significant, chosen by blind judges.", "",
        "## Output mapping", "",
        "The blend is a mean of percentile ranks over the whole December test set. To submit probabilities, the "
        "blend is quantile-mapped onto the sorted LightGBM December probabilities (k-th smallest blend gets the k-th "
        "smallest LightGBM probability; ties share the first value of the tie group). Order, hence AUC/Gini, equals the "
        "blend's exactly; the marginal distribution equals LightGBM's.", "",
        "## Pre-submission checks", "",
        f"- rows: {checks['rows']} (sample_submission order and columns validated); finite: {checks['finite']}; "
        f"within [0,1]: {checks['in_unit_interval']}",
        f"- overall: mean {overall['mean']:.5f}, p10 {overall['p10']:.5f}, p50 {overall['p50']:.5f}, "
        f"p90 {overall['p90']:.5f}",
        f"- groups: {checks['n_survivors']} survivors, {checks['n_new']} new clients (id > max train id), "
        f"{checks['n_other']} other", "",
        "| Group | Rows | Mean | p10 | p50 | p90 |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("Nov survivors", "survivors"), ("New clients", "new_clients")):
        d = checks[key]
        lines.append(f"| {label} | {d['rows']} | {d['mean']:.5f} | {d['p10']:.5f} | {d['p50']:.5f} | {d['p90']:.5f} |")
    if checks["spearman"]:
        s = checks["spearman"]
        lines += ["", f"## Spearman vs B0 submission (`{s['other']}`)", "",
                  f"- overall: {s['overall']:.5f}", f"- Nov survivors: {s['survivors']:.5f}",
                  f"- new clients: {s['new_clients']:.5f}"]
    return "\n".join(lines) + "\n"


def run_final_ensemble(
    data_dir: str | Path = "data",
    experiment_root: str | Path = "experiments/final",
    seed: int = SEED,
    iterations: dict | None = None,
    compare_with: str | Path | None = None,
) -> dict:
    iterations = dict(iterations or FINAL_ITERATIONS)
    data_dir, experiment_root = Path(data_dir), Path(experiment_root)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    train, test, sample = pd.read_csv(train_path), pd.read_csv(test_path), pd.read_csv(sample_path)
    train["objetivo"] = train["objetivo"].astype(int)
    validate_competition_data(train, test)
    validate_final_submission(sample, sample)
    validate_submission(sample, test)
    y = train["objetivo"]

    lgb_train, lgb_test = build_final_matrices(train, test, "D_absolute")
    lgb = fit_model("lightgbm", lgb_train, y, config=MODEL_CONFIGS["lightgbm"][0], seed=seed,
                    iterations=iterations["lightgbm"], early_stopping_rounds=None)
    lgb_prob = lgb.predict_proba(lgb_test)

    cat_train, cat_test = build_final_matrices(train, test, "A")
    cat_train, cat_test = rank_interaction_matrices(cat_train, train["mes"], cat_test, test["mes"])
    cat = fit_model("catboost", cat_train, y, config=MODEL_CONFIGS["catboost"][0], seed=seed,
                    iterations=iterations["catboost"], early_stopping_rounds=None)
    cat_prob = cat.predict_proba(cat_test)

    december = test["mes"]
    rank_lgb, rank_cat = within_month_rank(lgb_prob, december), within_month_rank(cat_prob, december)
    blend = (rank_lgb + rank_cat) / 2
    prediction = quantile_map_to_reference(blend, lgb_prob)
    submission = pd.DataFrame({"id_cliente": test["id_cliente"].to_numpy(), "prediccion": prediction})
    validate_final_submission(submission, sample)

    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
              f"_{sha256_file(train_path)[:8]}_{MODEL_NAME}")
    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    submission_path = run_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    lgb_path, cat_path = run_dir / "model_lightgbm.txt", run_dir / "model_catboost.cbm"
    lgb.save(str(lgb_path))
    cat.save(str(cat_path))
    members_path = run_dir / "member_predictions.csv"
    pd.DataFrame({"id_cliente": test["id_cliente"].to_numpy(), "prob_lightgbm": lgb_prob,
                  "prob_catboost": cat_prob, "rank_lightgbm": rank_lgb, "rank_catboost": rank_cat,
                  "blend": blend}).to_csv(members_path, index=False)

    checks = pre_submission_checks(submission, train, compare_with)
    results_path = run_dir / "results.md"
    results_path.write_text(_results_md(run_id, iterations, checks, _describe(prediction)), encoding="utf-8")

    code_paths = [Path("src/datafest") / n for n in ("final_ensemble.py", "final_fit.py", "hypotheses.py",
                  "ablation.py", "features.py", "models.py", "data.py", "lineage.py")]
    code_paths += [Path("pyproject.toml"), Path("uv.lock")]
    manifest = build_run_manifest(
        run_id, MODEL_NAME, "D_absolute+A(rank_interaccion)",
        input_paths=[train_path, test_path, sample_path], model_path=lgb_path,
        output_paths=[submission_path, results_path, members_path, cat_path],
        parameters={
            "members": {"lightgbm": {"features": "D_absolute", "config": MODEL_CONFIGS["lightgbm"][0]},
                        "catboost": {"features": "A", "transform": "rank_interaccion (per month; December-only for test)",
                                     "config": MODEL_CONFIGS["catboost"][0]}},
            "blend": "equal-weight mean of December percentile ranks",
            "output_mapping": "quantile map onto sorted LightGBM December probabilities",
            "iterations": iterations, "seed": seed, "early_stopping": False,
            "train_months": sorted(int(m) for m in train["mes"].unique()),
            "score_months": sorted(int(m) for m in test["mes"].unique()),
            "hypotheses_run": "20261006T100749205414Z_a714ebcd (H4b2_rank_lgbm_catboost)",
        },
        metrics={"checks": checks}, code_paths=[p for p in code_paths if p.is_file()],
        environment={"python": platform.python_version(), "datafest": __version__},
    )
    write_json(run_dir / "manifest.json", manifest)
    return {"run_id": run_id, "run_dir": str(run_dir), "submission": str(submission_path),
            "rows": len(submission), "checks": checks}
