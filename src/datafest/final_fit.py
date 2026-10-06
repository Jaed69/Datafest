"""Final fit: train on every labelled month and score the unlabelled test month.

Reuses the rolling-evaluation feature builders (``ablation._feature_matrix``),
model wrappers and lineage helpers. Nothing is tuned or early-stopped on a
held-out month: the structural hyperparameters are the project ``baseline``
configuration and the boosting iteration counts are the ones frozen by the
rolling ablation (selected on internal folds ending in August).
"""
from __future__ import annotations

from datetime import datetime, timezone
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from datafest import __version__
from datafest.ablation import VARIANT_CONFIGS, _feature_matrix
from datafest.data import validate_competition_data, validate_submission
from datafest.lineage import build_run_manifest, sha256_file, write_json
from datafest.models import MODEL_CONFIGS, SEED, fit_model

# Iteration counts frozen by the rolling ablation (manifest of run
# 20261001T051935820299Z_7e1c7547). They are kept as-is for the final fit:
# the training window grows by one month, but no held-out month exists to
# justify rescaling, and rescaling would be an untested change.
FINAL_ITERATIONS = {"lightgbm": 58, "catboost": 192}
VARIANT_ALIASES = {"B": "B_absolute", "D": "D_absolute"}
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def resolve_variant(variant: str) -> str:
    key = VARIANT_ALIASES.get(variant, variant)
    if key not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown feature variant: {variant}")
    return key


def validate_final_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Fail loudly unless ``submission`` mirrors ``sample_submission.csv``."""
    if list(sample.columns) != ["id_cliente", "prediccion"]:
        raise ValueError("sample_submission must have id_cliente,prediccion")
    validate_submission(submission, sample)


def build_final_matrices(
    train: pd.DataFrame, test: pd.DataFrame, key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature matrices for train and test from one causal pass over both.

    History features only look at the current and previous months, so scoring
    rows get history from train (seen clients) and none (new clients), and the
    train rows are identical to a train-only computation. The target never
    enters the concatenated frame.
    """
    predictors = train.drop(columns=["objetivo", "_source_row"], errors="ignore")
    combined = pd.concat([predictors, test[predictors.columns]], ignore_index=True)
    matrix = _feature_matrix(combined, key)
    return (
        matrix.iloc[: len(train)].reset_index(drop=True),
        matrix.iloc[len(train):].reset_index(drop=True),
    )


def _distribution(prediction: np.ndarray) -> dict:
    return {
        "rows": int(len(prediction)),
        "mean": float(np.mean(prediction)),
        "std": float(np.std(prediction)),
        "min": float(np.min(prediction)),
        "max": float(np.max(prediction)),
        "quantiles": {str(q): float(np.quantile(prediction, q)) for q in QUANTILES},
    }


def _write_results(path: Path, run_id: str, model: str, key: str, iterations: int,
                   stats: dict, spearman: dict | None) -> None:
    lines = [
        f"# Final fit {run_id}", "",
        f"- Model: {model}; features: {key}; iterations: {iterations} (no early stopping).",
        "- Trained on all labelled months (Jan-Nov 2026), scored December 2026.",
        "", "## Prediction distribution", "",
        f"- rows: {stats['rows']}", f"- mean: {stats['mean']:.5f}",
        f"- std: {stats['std']:.5f}", f"- min: {stats['min']:.5f}",
        f"- max: {stats['max']:.5f}", "", "| Quantile | Value |", "| ---: | ---: |",
    ]
    lines += [f"| {q} | {v:.5f} |" for q, v in stats["quantiles"].items()]
    if spearman:
        lines += ["", "## Agreement with another submission", "",
                  f"- compared with: `{spearman['other']}`",
                  f"- Spearman correlation: {spearman['spearman']:.5f}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_fit(
    model_name: str,
    variant: str,
    data_dir: str | Path = "data",
    experiment_root: str | Path = "experiments/final",
    seed: int = SEED,
    iterations: int | None = None,
    compare_with: str | Path | None = None,
) -> dict:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    key = resolve_variant(variant)
    data_dir, experiment_root = Path(data_dir), Path(experiment_root)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    train, test, sample = pd.read_csv(train_path), pd.read_csv(test_path), pd.read_csv(sample_path)
    train["objetivo"] = train["objetivo"].astype(int)
    validate_competition_data(train, test)
    validate_final_submission(sample, sample)
    validate_submission(sample, test)  # sample order must equal test order

    train_x, test_x = build_final_matrices(train, test, key)
    config = MODEL_CONFIGS[model_name][0]
    n_iterations = iterations if iterations is not None else FINAL_ITERATIONS[model_name]
    fitted = fit_model(
        model_name, train_x, train["objetivo"], config=config, seed=seed,
        iterations=n_iterations, early_stopping_rounds=None,
    )
    prediction = fitted.predict_proba(test_x)
    submission = pd.DataFrame({"id_cliente": test["id_cliente"].to_numpy(), "prediccion": prediction})
    validate_final_submission(submission, sample)

    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
              f"_{sha256_file(train_path)[:8]}_{model_name}_{key}")
    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    submission_path = run_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    model_path = run_dir / f"model.{'txt' if model_name == 'lightgbm' else 'cbm'}"
    fitted.save(str(model_path))

    stats = _distribution(prediction)
    spearman = None
    if compare_with is not None:
        other = pd.read_csv(compare_with)
        validate_final_submission(other, sample)
        spearman = {"other": str(compare_with),
                    "spearman": float(spearmanr(prediction, other["prediccion"]).statistic)}
    results_path = run_dir / "results.md"
    _write_results(results_path, run_id, model_name, key, n_iterations, stats, spearman)

    code_paths = [Path("src/datafest") / name for name in
                  ("final_fit.py", "ablation.py", "features.py", "models.py", "data.py", "lineage.py")]
    code_paths += [Path("pyproject.toml"), Path("uv.lock")]
    manifest = build_run_manifest(
        run_id, model_name, key,
        input_paths=[train_path, test_path, sample_path],
        model_path=model_path,
        output_paths=[submission_path, results_path],
        parameters={
            "config": config, "iterations": n_iterations, "seed": seed,
            "early_stopping": False,
            "train_months": sorted(int(m) for m in train["mes"].unique()),
            "score_months": sorted(int(m) for m in test["mes"].unique()),
            "feature_columns": list(train_x.columns),
            "iterations_source": "frozen by rolling ablation; not rescaled",
        },
        metrics={"prediction_distribution": stats, "spearman_vs_other": spearman},
        code_paths=[p for p in code_paths if p.is_file()],
        environment={"python": platform.python_version(), "datafest": __version__},
    )
    write_json(run_dir / "manifest.json", manifest)
    return {"run_id": run_id, "run_dir": str(run_dir), "submission": str(submission_path),
            "rows": len(submission), "distribution": stats, "spearman": spearman}
