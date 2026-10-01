from __future__ import annotations

from datetime import datetime, timezone
import platform
from pathlib import Path
import statistics

import numpy as np
import pandas as pd

from datafest import __version__
from datafest.data import validate_competition_data
from datafest.features import HISTORY_FEATURES, add_history_features, _month_ordinal
from datafest.lineage import file_record, sha256_file, write_json
from datafest.metrics import compute_metrics
from datafest.models import MODEL_CONFIGS, SEED, fit_model
from datafest.splitting import make_inner_temporal_folds


VARIANT_CONFIGS = {
    "A": ("base", "none"),
    "B_absolute": ("base_month", "absolute"),
    "B_month_index": ("base_month", "month_index"),
    "C": ("history", "none"),
    "D_absolute": ("base_history_month", "absolute"),
    "D_month_index": ("base_history_month", "month_index"),
}
ROLLING_CUTS = ((202608, 202609), (202609, 202610), (202610, 202611))
MODELS = ("lightgbm", "catboost")


def _feature_matrix(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    variant, encoding = VARIANT_CONFIGS[key]
    source = frame.drop(columns=["objetivo", "_source_row"], errors="ignore")
    include_history = variant in {"history", "base_history_month"}
    transformed = add_history_features(source) if include_history else source.copy()
    transformed = transformed.drop(columns=["id_cliente"], errors="ignore")
    if encoding == "none":
        transformed = transformed.drop(columns=["mes"], errors="ignore")
    elif encoding == "month_index":
        jan_ordinal = int(_month_ordinal(pd.Series([202601])).iloc[0])
        transformed["month_index"] = (
            _month_ordinal(transformed["mes"]) - jan_ordinal
        ).astype("int16")
        transformed = transformed.drop(columns=["mes"])
    return transformed.reset_index(drop=True)


def _gini(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    return compute_metrics(frame["objetivo"].astype(int), prediction)["gini"]


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_report(summary: pd.DataFrame, path: Path, batch_id: str, frozen: dict) -> None:
    fixed_iterations = ", ".join(
        f"{name}: {value['iterations']}" for name, value in frozen.items()
    )
    lines = [
        "# Ablación rolling de mes e historial", "", f"Lote: `{batch_id}`", "",
        "## Protocolo", "",
        "- A y C se ejecutan sin mes; B y D usan AAAAMM o `month_index` (enero=0 a noviembre=10). Todas excluyen `id_cliente`.",
        "- Rolling principal: ≤Ago→Sep, ≤Sep→Oct y ≤Oct→Nov. Se informa Gini mensual, media simple y desviación estándar poblacional (`ddof=0`).",
        "- Configuración baseline común por algoritmo; iteraciones elegidas en folds internos hasta agosto con A y C. Sin selección ni early stopping en septiembre-noviembre.",
        f"- Iteraciones congeladas: {fixed_iterations}.",
        "- Frozen: modelo entrenado hasta septiembre para octubre y noviembre. Pooled es secundaria.",
        "", "## Rolling principal", "",
        "| Modelo | Variante | Mes | Gini Sep | Gini Oct | Gini Nov | Media | Std. |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values(["model", "variant"]).iterrows():
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['month_encoding']} | "
            f"{row['gini_sep']:.5f} | {row['gini_oct']:.5f} | {row['gini_nov']:.5f} | "
            f"{row['mean_gini']:.5f} | {row['std_gini']:.5f} |"
        )
    lines.extend([
        "", "## Frozen hasta septiembre", "",
        "| Modelo | Variante | Gini Oct | Gini Nov | Media | Std. | Pooled secundario |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in summary.sort_values(["model", "variant"]).iterrows():
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['frozen_gini_oct']:.5f} | "
            f"{row['frozen_gini_nov']:.5f} | {row['frozen_mean_gini']:.5f} | "
            f"{row['frozen_std_gini']:.5f} | {row['frozen_pooled_secondary']:.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(
    data_dir: str | Path = "data",
    experiment_root: str | Path = "experiments/ablation",
    seed: int = SEED,
) -> dict:
    data_dir, experiment_root = Path(data_dir), Path(experiment_root)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["objetivo"] = train["objetivo"].astype(int)
    validate_competition_data(train, test)
    train["_source_row"] = np.arange(len(train), dtype=np.int64)
    train_months = sorted(int(month) for month in train["mes"].unique())
    feature_matrices = {key: _feature_matrix(train, key) for key in VARIANT_CONFIGS}

    # Freeze a common configuration per algorithm on the no-month A/C controls,
    # with internal validation months ending no later than August.
    inner = make_inner_temporal_folds(train.drop(columns=["_source_row"]), end_month=202608)
    frozen: dict[str, dict] = {}
    tuning_log: dict[str, list[dict]] = {}
    for model_name in MODELS:
        candidates = []
        # Keep tree-shape and regularization parameters fixed across every
        # feature ablation; only choose a common boosting iteration count.
        for config in MODEL_CONFIGS[model_name][:1]:
            fold_scores, iteration_counts = [], []
            for key in ("A", "C"):
                matrix = feature_matrices[key]
                for fold in inner:
                    print(
                        f"[iteration selection] {model_name}/{key}/valid={fold.valid_month}",
                        flush=True,
                    )
                    train_mask = train["mes"].le(fold.train_through).to_numpy()
                    valid_mask = train["mes"].eq(fold.valid_month).to_numpy()
                    fitted = fit_model(
                        model_name, matrix.loc[train_mask].reset_index(drop=True),
                        train.loc[train_mask, "objetivo"].astype(int),
                        valid_x=matrix.loc[valid_mask].reset_index(drop=True),
                        valid_y=train.loc[valid_mask, "objetivo"].astype(int),
                        config=config, seed=seed,
                    )
                    prediction = fitted.predict_proba(matrix.loc[valid_mask].reset_index(drop=True))
                    fold_scores.append(_gini(train.loc[valid_mask], prediction))
                    iteration_counts.append(fitted.best_iteration)
            candidates.append({
                "config": config,
                "mean_gini": float(np.mean(fold_scores)),
                "iterations": iteration_counts,
            })
        selected = max(candidates, key=lambda value: value["mean_gini"])
        frozen[model_name] = {
            "config": selected["config"],
            "iterations": max(1, int(statistics.median(selected["iterations"]))),
            "selection_months": [fold.valid_month for fold in inner],
            "selection_variants": ["A", "C"],
        }
        tuning_log[model_name] = [
            {"config": item["config"], "mean_gini": item["mean_gini"],
             "iterations_median": int(statistics.median(item["iterations"]))}
            for item in sorted(candidates, key=lambda value: value["mean_gini"], reverse=True)
        ]

    batch_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{sha256_file(train_path)[:8]}"
    batch_dir = experiment_root / "runs" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    schema_path = batch_dir / "feature_schemas.json"
    write_json(schema_path, {
        key: {"feature_columns": list(matrix.columns),
              "dtypes": {column: str(dtype) for column, dtype in matrix.dtypes.items()},
              "feature_group": VARIANT_CONFIGS[key][0],
              "month_encoding": VARIANT_CONFIGS[key][1]}
        for key, matrix in feature_matrices.items()
    })
    assignment_rows = []
    for train_through, valid_month in ROLLING_CUTS:
        for role, mask in (
            ("train", train["mes"].le(train_through)),
            ("validation", train["mes"].eq(valid_month)),
        ):
            assignment_rows.extend({
                "split": f"rolling_{valid_month}", "role": role,
                "source_row": int(row),
            } for row in train.loc[mask, "_source_row"])
    for role, mask in (
        ("train", train["mes"].le(202609)),
        ("validation", train["mes"].isin([202610, 202611])),
    ):
        assignment_rows.extend({
            "split": "frozen_sep_oct_nov", "role": role,
            "source_row": int(row),
        } for row in train.loc[mask, "_source_row"])
    assignments_path = batch_dir / "split_assignments.csv"
    _save_csv(pd.DataFrame(assignment_rows), assignments_path)
    split_manifest_path = batch_dir / "split_manifest.json"
    write_json(split_manifest_path, {
        "split_id": "rolling_aug_sep_oct_nov_plus_frozen_sep_oct_nov",
        "source": file_record(train_path),
        "assignments": file_record(assignments_path),
        "rolling_cuts": [{"train_through": a, "valid_month": b} for a, b in ROLLING_CUTS],
        "frozen_train_through": 202609,
        "frozen_valid_months": [202610, 202611],
    })
    summary_rows = []
    run_records = []

    for model_name in MODELS:
        choice = frozen[model_name]
        for key, (variant_name, encoding) in VARIANT_CONFIGS.items():
            print(f"[rolling + frozen] {model_name}/{key}", flush=True)
            matrix = feature_matrices[key]
            predictions = []
            scores = {}
            model_records = []
            for train_through, valid_month in ROLLING_CUTS:
                train_mask = train["mes"].le(train_through).to_numpy()
                valid_mask = train["mes"].eq(valid_month).to_numpy()
                fitted = fit_model(
                    model_name, matrix.loc[train_mask].reset_index(drop=True),
                    train.loc[train_mask, "objetivo"].astype(int),
                    config=choice["config"], seed=seed, iterations=choice["iterations"],
                    early_stopping_rounds=None,
                )
                valid_frame = train.loc[valid_mask, ["id_cliente", "mes", "objetivo"]].copy().reset_index(drop=True)
                prediction = fitted.predict_proba(matrix.loc[valid_mask].reset_index(drop=True))
                valid_frame["prediccion"] = prediction
                scores[str(valid_month)] = _gini(valid_frame, prediction)
                predictions.append(valid_frame)
                model_path = batch_dir / model_name / key / f"rolling_{valid_month}.{'txt' if model_name == 'lightgbm' else 'cbm'}"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                fitted.save(str(model_path))
                model_records.append(file_record(model_path))

            # Frozen model: train through September, predict October and November once.
            frozen_train_mask = train["mes"].le(202609).to_numpy()
            future_mask = train["mes"].isin([202610, 202611]).to_numpy()
            frozen_model = fit_model(
                model_name, matrix.loc[frozen_train_mask].reset_index(drop=True),
                train.loc[frozen_train_mask, "objetivo"].astype(int),
                config=choice["config"], seed=seed, iterations=choice["iterations"],
                early_stopping_rounds=None,
            )
            frozen_prediction = frozen_model.predict_proba(matrix.loc[future_mask].reset_index(drop=True))
            frozen_frame = train.loc[future_mask, ["id_cliente", "mes", "objetivo"]].copy().reset_index(drop=True)
            frozen_frame["prediccion"] = frozen_prediction
            frozen_scores = {
                str(month): _gini(part, part["prediccion"].to_numpy())
                for month, part in frozen_frame.groupby("mes", sort=True)
            }
            pooled_gini = _gini(frozen_frame, frozen_prediction)
            frozen_path = batch_dir / model_name / key / "frozen_oct_nov.csv"
            _save_csv(frozen_frame, frozen_path)
            model_path = batch_dir / model_name / key / f"frozen_sep.{ 'txt' if model_name == 'lightgbm' else 'cbm'}"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            frozen_model.save(str(model_path))
            model_records.append(file_record(model_path))
            rolling_frame = pd.concat(predictions, ignore_index=True)
            rolling_path = batch_dir / model_name / key / "rolling_predictions.csv"
            _save_csv(rolling_frame, rolling_path)
            rolling_mean, rolling_std = _mean_std([scores[str(month)] for month in (202609, 202610, 202611)])
            frozen_mean, frozen_std = _mean_std([frozen_scores[str(month)] for month in (202610, 202611)])
            metrics = {
                "rolling_gini_by_month": scores,
                "rolling_mean_gini": rolling_mean,
                "rolling_std_gini": rolling_std,
                "frozen_gini_by_month": frozen_scores,
                "frozen_mean_gini": frozen_mean,
                "frozen_std_gini": frozen_std,
                "frozen_pooled_oct_nov_gini_secondary": pooled_gini,
            }
            metrics_path = batch_dir / model_name / key / "metrics.json"
            write_json(metrics_path, metrics)
            run_id = f"{batch_id}_{model_name}_{key}"
            run_records.append({
                "run_id": run_id,
                "model_name": model_name,
                "variant": key,
                "variant_group": variant_name,
                "month_encoding": encoding,
                "parameters": choice,
                "metrics": metrics,
                "models": model_records,
                "predictions": [file_record(rolling_path), file_record(frozen_path)],
                "metrics_file": file_record(metrics_path),
            })
            summary_rows.append({
                "model": model_name, "variant": key, "features": variant_name,
                "month_encoding": encoding,
                "gini_sep": scores["202609"], "gini_oct": scores["202610"],
                "gini_nov": scores["202611"], "mean_gini": rolling_mean,
                "std_gini": rolling_std,
                "frozen_gini_oct": frozen_scores["202610"],
                "frozen_gini_nov": frozen_scores["202611"],
                "frozen_mean_gini": frozen_mean, "frozen_std_gini": frozen_std,
                "frozen_pooled_secondary": pooled_gini,
            })

    summary = pd.DataFrame(summary_rows).sort_values(["model", "variant"]).reset_index(drop=True)
    summary_csv = batch_dir / "summary.csv"
    _save_csv(summary, summary_csv)
    report_path = batch_dir / "results.md"
    _write_report(summary, report_path, batch_id, frozen)
    code_paths = [Path("src/datafest/ablation.py"), Path("src/datafest/cli.py"),
                  Path("src/datafest/models.py"), Path("src/datafest/features.py"),
                  Path("src/datafest/data.py"), Path("src/datafest/metrics.py"),
                  Path("src/datafest/splitting.py"), Path("pyproject.toml"), Path("uv.lock")]
    manifest = {
        "schema_version": 1, "run_id": batch_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "rolling_cuts": [{"train_through": a, "valid_month": b} for a, b in ROLLING_CUTS],
            "hyperparameter_selection_valid_through": 202608,
            "frozen_train_through": 202609,
            "frozen_valid_months": [202610, 202611],
            "primary_metric": "per-month Gini; report mean and population std",
            "pooled_metric_role": "secondary only",
        },
        "frozen_configurations": frozen, "tuning_cv": tuning_log,
        "runs": run_records, "summary": file_record(summary_csv),
        "report": file_record(report_path),
        "inputs": [file_record(train_path), file_record(test_path)],
        "split": file_record(split_manifest_path),
        "split_artifacts": [file_record(assignments_path)],
        "features": [file_record(schema_path)],
        "code": [file_record(path) for path in code_paths],
        "environment": {"python": platform.python_version(), "datafest": __version__},
    }
    manifest_path = batch_dir / "manifest.json"
    write_json(manifest_path, manifest)
    index = {
        "schema_version": 1, "batch_id": batch_id,
        "manifest": file_record(manifest_path), "summary": file_record(summary_csv),
        "report": file_record(report_path),
        "runs": [{"run_id": run["run_id"], "model": run["model_name"],
                  "variant": run["variant"], "metrics": run["metrics"]}
                 for run in run_records],
    }
    write_json(experiment_root / "index.json", index)
    return index
