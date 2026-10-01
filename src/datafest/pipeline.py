from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import statistics

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datafest import __version__
from datafest.data import validate_competition_data, validate_submission
from datafest.features import HISTORY_FEATURES, add_history_features, features_for_variant
from datafest.lineage import (
    build_run_manifest,
    file_record,
    sha256_file,
    verify_file_records,
    verify_manifest_integrity,
    write_json,
)
from datafest.metrics import compute_metrics, paired_group_bootstrap_gini_delta
from datafest.models import (
    EARLY_STOPPING_ROUNDS,
    MAX_ROUNDS,
    MODEL_CONFIGS,
    SEED,
    FittedModel,
    fit_model,
)
from datafest.splitting import (
    TemporalFold,
    make_group_diagnostic_split,
    make_inner_temporal_folds,
    make_outer_temporal_split,
)


VARIANTS = ("clean", "history", "id_raw")
MODELS = ("lightgbm", "catboost")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _save_csv(frame: pd.DataFrame, path: Path, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"index": False}
    if compressed:
        options["compression"] = {"method": "gzip", "mtime": 0}
    frame.to_csv(path, **options)


def _feature_schema(model: FittedModel, variant: str) -> dict:
    return {
        "feature_columns": model.feature_columns,
        "categorical_columns": model.categorical_columns,
        "category_maps": model.category_maps,
        "unknown_category_code": "__UNKNOWN__ for CatBoost; next category code for LightGBM",
        "history_features": HISTORY_FEATURES if variant == "history" else [],
        "feature_variant": variant,
    }


def _write_split_artifacts(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    source_path: Path,
    split_root: Path,
    seed: int,
) -> tuple[Path, Path, Path, dict]:
    train_months = sorted(int(value) for value in train["mes"].unique())
    valid_months = sorted(int(value) for value in valid["mes"].unique())
    split_id = f"temporal_{train_months[0]}-{train_months[-1]}__{valid_months[0]}-{valid_months[-1]}"
    split_path = split_root / split_id
    train_path = split_path / "train.csv"
    valid_path = split_path / "validation.csv"
    assignment_path = split_path / "inner_fold_assignments.csv"
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    _save_csv(train.drop(columns=["_source_row"], errors="ignore"), train_path)
    _save_csv(valid.drop(columns=["_source_row"], errors="ignore"), valid_path)

    folds = make_inner_temporal_folds(train)
    assignment_rows = []
    for fold_index, fold in enumerate(folds, start=1):
        assignment_rows.extend(
            {"fold": fold_index, "role": "train", "source_row": int(row)}
            for row in fold.train["_source_row"].tolist()
        )
        assignment_rows.extend(
            {"fold": fold_index, "role": "validation", "source_row": int(row)}
            for row in fold.valid["_source_row"].tolist()
        )
    assignments = pd.DataFrame(assignment_rows)
    _save_csv(assignments, assignment_path)

    group_train, group_valid = make_group_diagnostic_split(train, holdout_month=train_months[-1])
    group_assignment_path = split_path / "group_diagnostic_assignments.csv"
    group_assignments = pd.concat(
        [
            pd.DataFrame({"role": "train", "source_row": group_train["_source_row"].astype(int)}),
            pd.DataFrame({"role": "validation", "source_row": group_valid["_source_row"].astype(int)}),
        ],
        ignore_index=True,
    )
    _save_csv(group_assignments, group_assignment_path)

    manifest_path = split_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "split_id": split_id,
        "source": file_record(source_path),
        "seed": seed,
        "outer_train": file_record(train_path),
        "outer_validation": file_record(valid_path),
        "inner_fold_assignments": file_record(assignment_path),
        "inner_folds": [
            {
                "fold": index,
                "train_through": fold.train_through,
                "validation_month": fold.valid_month,
                "train_rows": len(fold.train),
                "validation_rows": len(fold.valid),
            }
            for index, fold in enumerate(folds, start=1)
        ],
        "group_diagnostic_assignments": file_record(group_assignment_path),
        "group_diagnostic": {
            "train_months": sorted(int(value) for value in group_train["mes"].unique()),
            "validation_month": train_months[-1],
            "train_rows": len(group_train),
            "validation_rows": len(group_valid),
            "holdout_clients": 0.2,
            "train_validation_client_overlap": 0,
        },
    }
    write_json(manifest_path, manifest)
    split_records = [
        manifest["source"],
        manifest["outer_train"],
        manifest["outer_validation"],
        manifest["inner_fold_assignments"],
        manifest["group_diagnostic_assignments"],
    ]
    errors = verify_file_records(split_records)
    if errors:
        raise RuntimeError(f"Integridad del split fallida: {errors}")
    return split_path, train_path, valid_path, manifest


def _take_rows(matrix: pd.DataFrame, full_frame: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    source_rows = set(subset["_source_row"].astype(int))
    selected_positions = np.flatnonzero(full_frame["_source_row"].astype(int).isin(source_rows).to_numpy())
    return matrix.iloc[selected_positions].reset_index(drop=True)


def _fold_input(
    all_features: pd.DataFrame,
    raw_frame: pd.DataFrame,
    fold: TemporalFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = raw_frame["mes"].le(fold.train_through)
    valid_mask = raw_frame["mes"].eq(fold.valid_month)
    return (
        all_features.loc[train_mask.to_numpy()].reset_index(drop=True),
        all_features.loc[valid_mask.to_numpy()].reset_index(drop=True),
    )


def _cohort_lookup(history_context: pd.DataFrame) -> dict[tuple[int, int], bool]:
    context = add_history_features(history_context.drop(columns=["objetivo"], errors="ignore"))
    lookup = {
        (int(client), int(month)): bool(recurrent)
        for client, month, recurrent in zip(
            context["id_cliente"], context["mes"], context["cliente_recurrente"]
        )
    }
    return lookup


def _cohort_labels(frame: pd.DataFrame, lookup: dict[tuple[int, int], bool]) -> pd.Series:
    return pd.Series(
        [lookup[(int(client), int(month))] for client, month in zip(frame["id_cliente"], frame["mes"])],
        index=frame.index,
    ).reset_index(drop=True)


def _score_slices(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    scored = frame[["id_cliente", "mes", "objetivo"]].copy().reset_index(drop=True)
    scored["prediccion"] = prediction
    if "cliente_recurrente" in frame:
        scored["cliente_recurrente"] = frame["cliente_recurrente"].to_numpy()
    result = {"overall": compute_metrics(scored["objetivo"], scored["prediccion"])}
    result["by_month"] = {}
    for month, part in scored.groupby("mes", sort=True):
        result["by_month"][str(int(month))] = compute_metrics(part["objetivo"], part["prediccion"])
    result["by_cohort"] = {}
    if "cliente_recurrente" in scored:
        for recurrent, part in scored.groupby("cliente_recurrente", sort=True):
            key = "recurrente" if recurrent else "nuevo"
            result["by_cohort"][key] = compute_metrics(part["objetivo"], part["prediccion"])
    return result


def _evaluate_config(
    model_name: str,
    variant: str,
    config: dict,
    raw_frame: pd.DataFrame,
    all_features: pd.DataFrame,
    folds: list[TemporalFold],
    seed: int,
    cohort_lookup: dict[tuple[int, int], bool],
) -> dict:
    fold_metrics = []
    best_iterations = []
    oof_parts = []
    for fold in folds:
        train_x, valid_x = _fold_input(all_features, raw_frame, fold)
        train_y = fold.train["objetivo"].astype(int).to_numpy()
        valid_y = fold.valid["objetivo"].astype(int).to_numpy()
        model = fit_model(
            model_name,
            train_x,
            train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            config=config,
            seed=seed,
        )
        prediction = model.predict_proba(valid_x)
        metrics = compute_metrics(valid_y, prediction)
        fold_metrics.append({"valid_month": fold.valid_month, **metrics})
        best_iterations.append(model.best_iteration)
        oof = fold.valid[["id_cliente", "mes", "objetivo"]].copy().reset_index(drop=True)
        oof["prediccion"] = prediction
        oof["cliente_recurrente"] = _cohort_labels(fold.valid, cohort_lookup).to_numpy()
        oof_parts.append(oof)
    return {
        "config": config,
        "fold_metrics": fold_metrics,
        "mean_gini": float(np.mean([value["gini"] for value in fold_metrics])),
        "mean_roc_auc": float(np.mean([value["roc_auc"] for value in fold_metrics])),
        "mean_pr_auc": float(np.mean([value["pr_auc"] for value in fold_metrics])),
        "best_iterations": best_iterations,
        "oof_predictions": pd.concat(oof_parts, ignore_index=True),
    }


def _group_diagnostic(
    model_name: str,
    variant: str,
    config: dict,
    raw_frame: pd.DataFrame,
    all_features: pd.DataFrame,
    group_train: pd.DataFrame,
    group_valid: pd.DataFrame,
    iterations: int,
    seed: int,
    cohort_lookup: dict[tuple[int, int], bool],
) -> tuple[dict, pd.DataFrame]:
    train_x = _take_rows(all_features, raw_frame, group_train)
    valid_x = _take_rows(all_features, raw_frame, group_valid)
    model = fit_model(
        model_name,
        train_x,
        group_train["objetivo"].astype(int),
        config=config,
        seed=seed,
        iterations=iterations,
        early_stopping_rounds=None,
    )
    prediction = model.predict_proba(valid_x)
    scored = group_valid[["id_cliente", "mes", "objetivo"]].copy().reset_index(drop=True)
    scored["prediccion"] = prediction
    scored["cliente_recurrente"] = _cohort_labels(group_valid, cohort_lookup).to_numpy()
    return compute_metrics(scored["objetivo"], prediction), scored


def _gate_id_experiment(
    raw_oof: pd.DataFrame,
    history_oof: pd.DataFrame,
    raw_group: pd.DataFrame,
    history_group: pd.DataFrame,
    seed: int,
) -> dict:
    raw_oof = raw_oof.sort_values(["mes", "id_cliente"]).reset_index(drop=True)
    history_oof = history_oof.sort_values(["mes", "id_cliente"]).reset_index(drop=True)
    if not np.array_equal(raw_oof[["id_cliente", "mes"]], history_oof[["id_cliente", "mes"]]):
        raise ValueError("Las predicciones OOF de ID e historial no están alineadas")
    temporal = paired_group_bootstrap_gini_delta(
        raw_oof["objetivo"],
        raw_oof["prediccion"],
        history_oof["prediccion"],
        raw_oof["id_cliente"],
        n_resamples=1000,
        seed=seed,
    )
    group = paired_group_bootstrap_gini_delta(
        raw_group["objetivo"],
        raw_group["prediccion"],
        history_group["prediccion"],
        raw_group["id_cliente"],
        n_resamples=1000,
        seed=seed,
    )
    cohort_deltas = {}
    for recurrent, name in [(False, "nuevo"), (True, "recurrente")]:
        mask = raw_oof["cliente_recurrente"].eq(recurrent).to_numpy()
        if mask.sum() and raw_oof.loc[mask, "objetivo"].nunique() == 2:
            cohort_deltas[name] = float(
                (2 * roc_auc_score(raw_oof.loc[mask, "objetivo"], raw_oof.loc[mask, "prediccion"]) - 1)
                - (2 * roc_auc_score(history_oof.loc[mask, "objetivo"], history_oof.loc[mask, "prediccion"]) - 1)
            )
        else:
            cohort_deltas[name] = None
    passed = (
        temporal["ci95_low"] > 0
        and group["ci95_low"] > 0
        and all(value is None or value >= 0 for value in cohort_deltas.values())
    )
    return {
        "passed": bool(passed),
        "criteria": {
            "temporal_delta_ci95_low_gt_zero": temporal["ci95_low"] > 0,
            "unseen_client_group_delta_ci95_low_gt_zero": group["ci95_low"] > 0,
            "no_negative_delta_for_new_or_recurrent": all(
                value is None or value >= 0 for value in cohort_deltas.values()
            ),
        },
        "temporal_raw_id_vs_history": temporal,
        "unseen_client_group_raw_id_vs_history": group,
        "temporal_cohort_delta_gini": cohort_deltas,
        "bootstrap": {"unit": "id_cliente", "resamples": 1000, "seed": seed},
    }


def _outer_winner_status(candidates: list[dict], seed: int) -> dict:
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    eligible.sort(key=lambda value: (-value["inner_mean_gini"], value["model_name"], value["variant"]))
    if len(eligible) < 2:
        return {
            "status": "no_conclusive_winner",
            "reason": "Se necesitan al menos dos candidatos elegibles para una comparación pareada.",
            "champion": eligible[0]["key"] if eligible else None,
        }
    champion, runner_up = eligible[:2]
    first = champion["validation_predictions"].sort_values(["mes", "id_cliente"]).reset_index(drop=True)
    second = runner_up["validation_predictions"].sort_values(["mes", "id_cliente"]).reset_index(drop=True)
    if not np.array_equal(first[["id_cliente", "mes"]], second[["id_cliente", "mes"]]):
        raise ValueError("Las predicciones del holdout no están alineadas entre candidatos")
    delta = paired_group_bootstrap_gini_delta(
        first["objetivo"], first["prediccion"], second["prediccion"], first["id_cliente"],
        n_resamples=2000, seed=seed
    )
    month_scores = champion["metrics"]["by_month"]
    positive_months = all(value["gini"] > 0 for value in month_scores.values())
    cohort_scores = champion["metrics"]["by_cohort"]
    nonnegative_cohorts = all(value["gini"] >= 0 for value in cohort_scores.values())
    passed = delta["ci95_low"] > 0 and positive_months and nonnegative_cohorts
    return {
        "status": "conclusive_winner" if passed else "no_conclusive_winner",
        "champion": champion["key"],
        "runner_up": runner_up["key"],
        "selection_basis": "mean inner-fold Gini, before the outer holdout was scored",
        "champion_vs_runner_up": delta,
        "champion_month_gini_positive": positive_months,
        "champion_new_and_recurrent_gini_nonnegative": nonnegative_cohorts,
        "holdout_used_for_hyperparameter_selection": False,
    }


def run_pipeline(
    data_dir: str | Path = "data",
    experiment_root: str | Path = "experiments",
    split_root: str | Path = "data/splits",
    processed_root: str | Path = "data/processed",
    seed: int = SEED,
) -> dict:
    data_dir = Path(data_dir)
    experiment_root = Path(experiment_root)
    split_root = Path(split_root)
    processed_root = Path(processed_root)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    metadata_path = data_dir / "metaData.csv"

    train = pd.read_csv(train_path).reset_index(drop=True)
    test = pd.read_csv(test_path).reset_index(drop=True)
    train["objetivo"] = train["objetivo"].astype(int)
    report = validate_competition_data(train, test)
    sample = pd.read_csv(sample_path)
    validate_submission(sample, test)
    train["_source_row"] = np.arange(len(train), dtype=np.int64)

    outer_train, outer_valid = make_outer_temporal_split(train)
    split_path, outer_train_path, outer_valid_path, split_manifest = _write_split_artifacts(
        outer_train,
        outer_valid,
        train_path,
        split_root,
        seed,
    )
    folds = make_inner_temporal_folds(outer_train)
    group_train, group_valid = make_group_diagnostic_split(
        outer_train, holdout_month=int(outer_train["mes"].max())
    )

    # All transforms are causal by month. Targets are never inputs to feature construction.
    fold_results: dict[tuple[str, str, str], dict] = {}
    group_predictions: dict[tuple[str, str], tuple[dict, pd.DataFrame]] = {}
    all_inner_features = {
        variant: features_for_variant(outer_train.drop(columns=["_source_row"]), variant)
        for variant in VARIANTS
    }
    inner_cohort_lookup = _cohort_lookup(outer_train)

    # Run each fixed baseline first; these are also the anti-memorization gate controls.
    for model_name in MODELS:
        baseline = MODEL_CONFIGS[model_name][0]
        for variant in VARIANTS:
            print(f"[baseline CV] {model_name}/{variant}", flush=True)
            feature_source = all_inner_features[variant]
            raw_for_folds = outer_train.reset_index(drop=True)
            result = _evaluate_config(
                model_name,
                variant,
                baseline,
                raw_for_folds.drop(columns=["_source_row"], errors="ignore"),
                feature_source,
                folds,
                seed,
                inner_cohort_lookup,
            )
            fold_results[(model_name, variant, baseline["name"])] = result

        for variant in ("history", "id_raw"):
            print(f"[group diagnostic] {model_name}/{variant}", flush=True)
            diagnostic, prediction = _group_diagnostic(
                model_name,
                variant,
                baseline,
                outer_train,
                all_inner_features[variant],
                group_train,
                group_valid,
                iterations=max(1, int(statistics.median(fold_results[(model_name, variant, baseline["name"])]["best_iterations"]))),
                seed=seed,
                cohort_lookup=inner_cohort_lookup,
            )
            group_predictions[(model_name, variant)] = (diagnostic, prediction)

    id_gates = {}
    for model_name in MODELS:
        id_baseline = fold_results[(model_name, "id_raw", "baseline")]
        history_baseline = fold_results[(model_name, "history", "baseline")]
        gate = _gate_id_experiment(
            id_baseline["oof_predictions"],
            history_baseline["oof_predictions"],
            group_predictions[(model_name, "id_raw")][1],
            group_predictions[(model_name, "history")][1],
            seed,
        )
        id_gates[model_name] = gate
        print(f"[ID gate] {model_name}: {'passed' if gate['passed'] else 'failed'}", flush=True)

    # Search five fixed candidates on each clean feature set. Raw ID search is gated.
    for model_name in MODELS:
        for variant in VARIANTS:
            configs = MODEL_CONFIGS[model_name]
            if variant == "id_raw" and not id_gates[model_name]["passed"]:
                configs = configs[:1]
            for config in configs:
                key = (model_name, variant, config["name"])
                if key in fold_results:
                    continue
                print(f"[search CV] {model_name}/{variant}/{config['name']}", flush=True)
                fold_results[key] = _evaluate_config(
                    model_name,
                    variant,
                    config,
                    outer_train.drop(columns=["_source_row"], errors="ignore").reset_index(drop=True),
                    all_inner_features[variant],
                    folds,
                    seed,
                    inner_cohort_lookup,
                )

    selected: dict[tuple[str, str], dict] = {}
    search_tables = {}
    for model_name in MODELS:
        for variant in VARIANTS:
            candidates = [
                value
                for (candidate_model, candidate_variant, _), value in fold_results.items()
                if candidate_model == model_name and candidate_variant == variant
            ]
            best = max(candidates, key=lambda value: (value["mean_gini"], value["mean_pr_auc"]))
            selected[(model_name, variant)] = best
            search_tables[(model_name, variant)] = [
                {
                    "config": value["config"],
                    "fold_metrics": value["fold_metrics"],
                    "mean_gini": value["mean_gini"],
                    "mean_roc_auc": value["mean_roc_auc"],
                    "mean_pr_auc": value["mean_pr_auc"],
                    "best_iterations": value["best_iterations"],
                }
                for value in sorted(candidates, key=lambda row: row["mean_gini"], reverse=True)
            ]

    full_train_no_target = train.drop(columns=["objetivo", "_source_row"])
    test_only = test.copy()
    combined_features = pd.concat([full_train_no_target, test_only], ignore_index=True)
    split_id = split_manifest["split_id"]
    processed_paths: dict[tuple[str, str], dict[str, Path]] = {}
    final_feature_matrices = {}
    outer_feature_matrices = {}
    for variant in VARIANTS:
        outer_all = features_for_variant(
            train.drop(columns=["_source_row"]), variant
        )
        outer_train_mask = train["mes"].lt(min(outer_valid["mes"]))
        outer_valid_mask = train["mes"].ge(min(outer_valid["mes"]))
        outer_feature_matrices[variant] = outer_all
        final_all = features_for_variant(combined_features, variant)
        final_train_matrix = final_all.iloc[: len(train)].reset_index(drop=True)
        final_test_matrix = final_all.iloc[len(train) :].reset_index(drop=True)
        variant_path = processed_root / split_id / variant
        paths = {
            "outer_train": variant_path / "outer_train.csv.gz",
            "outer_valid": variant_path / "outer_validation.csv.gz",
            "final_train": variant_path / "final_train.csv.gz",
            "test": variant_path / "competition_test.csv.gz",
        }
        _save_csv(outer_all.loc[outer_train_mask.to_numpy()].reset_index(drop=True), paths["outer_train"], True)
        _save_csv(outer_all.loc[outer_valid_mask.to_numpy()].reset_index(drop=True), paths["outer_valid"], True)
        _save_csv(final_train_matrix, paths["final_train"], True)
        _save_csv(final_test_matrix, paths["test"], True)
        final_feature_matrices[variant] = (final_train_matrix, final_test_matrix)
        processed_paths[("all", variant)] = paths

    valid_cohort_lookup = _cohort_lookup(train)
    for model_name in MODELS:
        for variant in VARIANTS:
            print(f"[outer evaluation + final fit] {model_name}/{variant}", flush=True)
            best = selected[(model_name, variant)]
            config = best["config"]
            best_iterations = max(1, int(statistics.median(best["best_iterations"])))
            outer_x_train = features_for_variant(
                outer_train.drop(columns=["_source_row"]), variant
            )
            # Causal history for November may use October predictors, never October labels.
            outer_x_valid = outer_feature_matrices[variant].loc[
                outer_valid_mask.to_numpy()
            ].reset_index(drop=True)
            fitted_outer = fit_model(
                model_name,
                outer_x_train,
                outer_train["objetivo"].astype(int),
                config=config,
                seed=seed,
                iterations=best_iterations,
                early_stopping_rounds=None,
            )
            valid_prediction = fitted_outer.predict_proba(outer_x_valid)
            valid_scored = outer_valid[["id_cliente", "mes", "objetivo"]].copy().reset_index(drop=True)
            valid_scored["prediccion"] = valid_prediction
            valid_scored["cliente_recurrente"] = _cohort_labels(outer_valid, valid_cohort_lookup).to_numpy()
            metrics = _score_slices(valid_scored, valid_prediction)

            # Refit on all labeled months and score December without accessing labels.
            final_train_matrix, final_test_matrix = final_feature_matrices[variant]
            fitted_final = fit_model(
                model_name,
                final_train_matrix,
                train["objetivo"].astype(int),
                config=config,
                seed=seed,
                iterations=best_iterations,
                early_stopping_rounds=None,
            )
            test_prediction = fitted_final.predict_proba(final_test_matrix)
            submission = pd.DataFrame(
                {"id_cliente": test["id_cliente"].to_numpy(), "prediccion": test_prediction}
            )
            validate_submission(submission, test)
            selected[(model_name, variant)]["metrics"] = metrics
            selected[(model_name, variant)]["validation_predictions"] = valid_scored
            selected[(model_name, variant)]["best_iterations_median"] = best_iterations
            selected[(model_name, variant)]["validation_model"] = fitted_outer
            selected[(model_name, variant)]["model"] = fitted_final
            selected[(model_name, variant)]["submission"] = submission
            selected[(model_name, variant)]["validation_x"] = outer_x_valid

    candidates_for_winner = []
    for model_name in MODELS:
        for variant in VARIANTS:
            chosen = selected[(model_name, variant)]
            eligible = variant != "id_raw" or id_gates[model_name]["passed"]
            candidates_for_winner.append(
                {
                    "key": f"{model_name}/{variant}",
                    "model_name": model_name,
                    "variant": variant,
                    "eligible": eligible,
                    "inner_mean_gini": chosen["mean_gini"],
                    "metrics": chosen["metrics"],
                    "validation_predictions": chosen["validation_predictions"],
                }
            )
    winner = _outer_winner_status(candidates_for_winner, seed)

    run_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_batch_id = f"{run_time}_{sha256_file(train_path)[:8]}"
    code_paths = sorted(Path("src/datafest").glob("*.py")) + [
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    split_artifact_paths = [
        split_path / "train.csv",
        split_path / "validation.csv",
        split_path / "inner_fold_assignments.csv",
        split_path / "group_diagnostic_assignments.csv",
    ]
    all_runs = []
    for model_name in MODELS:
        for variant in VARIANTS:
            chosen = selected[(model_name, variant)]
            run_id = f"{run_batch_id}_{model_name}_{variant}"
            run_dir = experiment_root / model_name / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            ext = "txt" if model_name == "lightgbm" else "cbm"
            model_path = run_dir / f"model.{ext}"
            chosen["model"].save(str(model_path))
            validation_model_path = run_dir / f"validation_model.{ext}"
            chosen["validation_model"].save(str(validation_model_path))
            validation_path = run_dir / "validation_predictions.csv"
            submission_path = run_dir / "submission.csv"
            metrics_path = run_dir / "metrics.json"
            schema_path = run_dir / "feature_schema.json"
            outer_schema_path = run_dir / "validation_feature_schema.json"
            _save_csv(chosen["validation_predictions"], validation_path)
            _save_csv(chosen["submission"], submission_path)

            write_json(schema_path, _feature_schema(chosen["model"], variant))
            write_json(outer_schema_path, _feature_schema(chosen["validation_model"], variant))
            gate = id_gates[model_name]
            variant_eligible = variant != "id_raw" or gate["passed"]
            report_metrics = {
                "inner_search": search_tables[(model_name, variant)],
                "selected_inner_mean_gini": chosen["mean_gini"],
                "selected_inner_mean_roc_auc": chosen["mean_roc_auc"],
                "selected_inner_mean_pr_auc": chosen["mean_pr_auc"],
                "outer_holdout": chosen["metrics"],
                "id_memorization_gate": gate if variant == "id_raw" else None,
                "winner_assessment": winner,
                "test_rows": len(test),
                "submission_columns": list(chosen["submission"].columns),
            }
            write_json(metrics_path, _jsonable(report_metrics))
            source_paths = [train_path, test_path, sample_path, metadata_path]
            feature_files = list(processed_paths[("all", variant)].values()) + [schema_path]
            output_paths = [
                validation_path,
                submission_path,
                metrics_path,
                schema_path,
                outer_schema_path,
            ]
            extra = {
                "split_id": split_id,
                "split_manifest": str(split_path / "manifest.json"),
                "selected_config": chosen["config"],
                "best_iterations_median": chosen["best_iterations_median"],
                "seed": seed,
                "id_gate_passed": gate["passed"],
                "winner_status": winner["status"],
                "variant_eligible_for_winner": variant_eligible,
                "package_version": __version__,
            }
            manifest = build_run_manifest(
                run_id,
                model_name,
                variant,
                source_paths,
                model_path,
                output_paths,
                parameters={
                    "selected_config": chosen["config"],
                    "seed": seed,
                    "iterations": chosen["best_iterations_median"],
                    "max_rounds_during_search": MAX_ROUNDS,
                    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                },
                metrics=report_metrics,
                split_path=split_path / "manifest.json",
                split_artifacts=split_artifact_paths,
                feature_paths=feature_files,
                code_paths=code_paths,
                validation_model_path=validation_model_path,
                environment={"python": platform.python_version(), "datafest": __version__},
                extra=extra,
            )
            manifest_path = run_dir / "manifest.json"
            write_json(manifest_path, _jsonable(manifest))
            integrity_errors = verify_manifest_integrity(manifest)
            if integrity_errors:
                raise RuntimeError(f"Integridad de corrida fallida: {integrity_errors}")
            all_runs.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "feature_variant": variant,
                    "run_dir": str(run_dir),
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "metrics": chosen["metrics"],
                    "inner_mean_gini": chosen["mean_gini"],
                    "id_gate_passed": gate["passed"],
                    "winner_status": winner["status"],
                    "eligible_for_winner": variant_eligible,
                    "submission": str(submission_path),
                }
            )

    index = {
        "schema_version": 1,
        "batch_id": run_batch_id,
        "split_id": split_id,
        "data_report": report.__dict__,
        "id_gates": id_gates,
        "winner_assessment": winner,
        "runs": all_runs,
    }
    write_json(experiment_root / "index.json", _jsonable(index))
    return index
