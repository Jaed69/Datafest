"""Evaluate official NVIDIA Kumo Tabular with Datafest's rolling protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import sdm

from datafest.features import add_history_features, _month_ordinal
from datafest.metrics import compute_metrics


SEED = 42
CUTS = ((202608, 202609), (202609, 202610), (202610, 202611))
BASELINE_GINI = 0.25104
MONTH_NAMES = {202609: "Sep", 202610: "Oct", 202611: "Nov"}
CHUNK_SIZE = 256
SIZES = ("small", "medium", "large")


def features(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    source = frame.drop(columns=["objetivo", "_source_row"], errors="ignore")
    if variant == "A":
        return source.drop(columns=["id_cliente", "mes"], errors="ignore").reset_index(drop=True)
    transformed = add_history_features(source)
    transformed["month_index"] = (
        _month_ordinal(transformed["mes"]) - int(_month_ordinal(pd.Series([202601])).iloc[0])
    ).astype("int16")
    return transformed.drop(columns=["id_cliente", "mes"], errors="ignore").reset_index(drop=True)


def stratified_indices(months: np.ndarray, n: int, method: str, seed: int) -> np.ndarray:
    """Deterministic train-only selection; never sees a validation target."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(months))
    if n >= len(months):
        return all_idx
    if method == "uniform":
        return np.sort(rng.choice(all_idx, size=n, replace=False))
    unique = np.array(sorted(np.unique(months)))
    groups = [np.flatnonzero(months == month) for month in unique]
    # Largest-remainder allocation, with every train month represented.
    weights = np.ones(len(unique), dtype=float)
    if method == "recency-aware":
        weights += np.linspace(0.0, 1.0, len(unique))
    elif method != "balanced-by-month":
        raise ValueError(method)
    quotas = weights / weights.sum() * n
    counts = np.maximum(1, np.floor(quotas).astype(int))
    while counts.sum() > n:
        candidates = np.flatnonzero(counts > 1)
        j = candidates[np.argmin(quotas[candidates] - counts[candidates])]
        counts[j] -= 1
    while counts.sum() < n:
        j = int(np.argmax(quotas - counts))
        counts[j] += 1
    chosen = [rng.choice(group, min(count, len(group)), replace=False) for group, count in zip(groups, counts)]
    out = np.concatenate(chosen)
    if len(out) < n:
        remaining = np.setdiff1d(all_idx, out, assume_unique=False)
        out = np.concatenate([out, rng.choice(remaining, n - len(out), replace=False)])
    return np.sort(out[:n])


def make_tensor(frame: pd.DataFrame, stypes: dict, device: torch.device) -> sdm.TableTensor:
    return sdm.TableTensor.from_pandas(df=frame, stypes=stypes, device=device)


def probabilities(model, x_query, chunk_size: int) -> np.ndarray:
    out = []
    for start in range(0, x_query.shape[0], chunk_size):
        prediction = model.predict(x_query[start:start + chunk_size])
        pdf = prediction.to_pandas()
        positive = "1" if "1" in pdf.columns else 1
        out.append(pdf[positive].to_numpy(dtype=float))
    return np.concatenate(out)


def new_model(size: str, device: torch.device):
    return sdm.models.KumoTabular(task="classification", size=size, device=device)


def metrics_for(y, pred):
    return compute_metrics(np.asarray(y, dtype=int), np.asarray(pred, dtype=float))


def max_gpu_memory(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / 1024**3)


def cuda_memory_gib() -> tuple[float, float] | None:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


def hardware() -> dict:
    result = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os_cpu_count()}
    if torch.cuda.is_available():
        result.update({"gpu": torch.cuda.get_device_name(0), "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)})
    else:
        result["gpu"] = None
    return result


def os_cpu_count() -> int | None:
    return __import__("os").cpu_count()


def official_version() -> str:
    return str(getattr(sdm, "__version__", "unknown"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_month: np.ndarray,
    query_x: pd.DataFrame,
    *,
    size: str,
    estimators: int,
    context_limit: int,
    sampling: str,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    chosen_contexts = []
    for estimator in range(estimators):
        ix = stratified_indices(train_month, min(context_limit, len(train_x)), sampling, seed + estimator * 1009)
        chosen_contexts.append(ix)
    lengths = {len(ix) for ix in chosen_contexts}
    if len(lengths) != 1:
        raise RuntimeError("Estimator contexts must have identical size")
    context_n = lengths.pop()
    context = train_x.iloc[chosen_contexts[0]].reset_index(drop=True)
    stypes = sdm.infer_stypes(context)
    x_context = make_tensor(context, stypes, device)
    y_context = make_tensor(pd.DataFrame({"target": train_y.iloc[chosen_contexts[0]].astype("int64").reset_index(drop=True)}), {"target": "categorical"}, device)
    query = make_tensor(query_x.reset_index(drop=True), stypes, device)
    if estimators > 1:
        # NVIDIA's documented estimator dimension: distinct train-only contexts.
        context_ids = torch.as_tensor(np.stack(chosen_contexts), device=device)
        x_context = make_tensor(train_x.reset_index(drop=True), stypes, device)[context_ids]
        y_full = make_tensor(pd.DataFrame({"target": train_y.astype("int64").reset_index(drop=True)}), {"target": "categorical"}, device)
        y_context = y_full[context_ids]
        query = query.expand(estimators, *query.shape)
    seed_generator = torch.Generator(device=device).manual_seed(seed)
    model = new_model(size, device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    try:
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            model.fit(x_context, y_context, num_estimators=None if estimators > 1 else 1, estimator_batch_size=1, generator=seed_generator)
            pred_parts = []
            query_n = query.shape[-2]
            for start in range(0, query_n, CHUNK_SIZE):
                pred_parts.append(model.predict(query[..., start:start + CHUNK_SIZE, :]))
            # The official recipe averages estimator outputs before exposing
            # its probability columns, including for distinct context tensors.
            pred = pd.concat([part.to_pandas() for part in pred_parts], ignore_index=True)
            pos = "1" if "1" in pred.columns else 1
            prediction = pred[pos].to_numpy(dtype=float)
    finally:
        del model, x_context, y_context, query
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return prediction, {
        "context_rows_per_estimator": context_n,
        "estimator_context_rows": context_n * estimators,
        "query_rows": len(query_x),
        "batch_size": CHUNK_SIZE,
        "estimators": estimators,
        "sampling": sampling,
        "elapsed_seconds": time.perf_counter() - started,
        "vram_peak_gib": max_gpu_memory(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", default="experiments/kumo")
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--context-limit", type=int, default=None, help="Optional hardware-limited cap; None means all fold training rows")
    parser.add_argument("--max-context-limit", type=int, default=32768, help="Context cap for the max-context A/D and Medium/Large runs; fixed sampling/ensemble plans remain at 8,192")
    parser.add_argument("--sizes", nargs="*", choices=SIZES, default=None, help="Optional additional size arena")
    parser.add_argument("--skip-ensembles", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.output_root)
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{official_version().replace('+','-')}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    data = pd.read_csv(Path(args.data_dir) / "train.csv")
    data["_source_row"] = np.arange(len(data), dtype=np.int64)
    feature_sets = {name: features(data, name) for name in ("A", "D")}
    all_predictions: dict[tuple[str, str], list[pd.DataFrame]] = {}
    record_rows: list[dict] = []
    prediction_path = run_dir / "predictions.csv"
    prediction_frames = []
    completed = set()
    if args.resume and (run_dir / "fold_results.csv").exists():
        previous_records = pd.read_csv(run_dir / "fold_results.csv")
        record_rows = previous_records.to_dict("records")
        if prediction_path.exists():
            previous_predictions = pd.read_csv(prediction_path)
            prediction_frames = []
            cursor = 0
            # Persisted predictions are concatenated in the same order as fold_results.
            # Slice them by each config's recorded query row count to distinguish
            # the E=1 max-context and fixed 8,192-row plans.
            for record in previous_records.itertuples():
                n_query = int(record.query_rows)
                fold = previous_predictions.iloc[cursor:cursor + n_query].copy().reset_index(drop=True)
                cursor += n_query
                fold["context_rows_per_estimator"] = int(record.context_rows_per_estimator)
                prediction_frames.append(fold)
                key = (str(record.model_size), f"{record.variant}-E{int(record.estimators)}-C{int(record.context_rows_per_estimator)}-{record.sampling}")
                all_predictions.setdefault(key, []).append(fold)
            if cursor != len(previous_predictions):
                raise ValueError("No se pudo alinear predictions.csv con fold_results.csv al reanudar")
        completed = {
            (str(r.model_size), str(r.variant), int(r.estimators), int(r.context_rows_per_estimator), str(r.sampling), int(r.valid_month))
            for r in previous_records.itertuples()
        } if args.resume and (run_dir / "fold_results.csv").exists() else set()
    model_sizes = [args.size] + [s for s in (args.sizes or []) if s != args.size]
    for model_size in model_sizes:
        # One estimator uses the largest stable measured context. Sampling and
        # estimator ensembles use a smaller context to keep iteration viable.
        plans = [("A", 1, args.max_context_limit, "uniform"), ("D", 1, args.max_context_limit, "uniform")]
        if not args.skip_ensembles:
            plans += [("D", 1, 8192, method) for method in ("uniform", "recency-aware", "balanced-by-month")]
            plans += [("D", n, 8192, "uniform") for n in (4, 8)]
        if model_size != args.size:
            plans = [("D", 1, args.max_context_limit, "uniform")]
        for variant, estimator_count, planned_cap, planned_sampling in plans:
                for train_through, valid_month in CUTS:
                    cap = args.context_limit if args.context_limit is not None else planned_cap
                    expected_context = min(cap, int(data["mes"].le(train_through).sum()))
                    expected_sampling = planned_sampling if cap < int(data["mes"].le(train_through).sum()) else "uniform"
                    if (model_size, variant, estimator_count, expected_context, expected_sampling, valid_month) in completed:
                        print(f"[resume skip] {model_size}/{variant} E={estimator_count} {planned_sampling} valid={valid_month}", flush=True)
                        continue
                    tr = data["mes"].le(train_through).to_numpy()
                    va = data["mes"].eq(valid_month).to_numpy()
                    train_x = feature_sets[variant].loc[tr].reset_index(drop=True)
                    query_x = feature_sets[variant].loc[va].reset_index(drop=True)
                    train_y = data.loc[tr, "objetivo"].astype(int).reset_index(drop=True)
                    cap = args.context_limit if args.context_limit is not None else planned_cap
                    sampling = planned_sampling if cap < len(train_x) else "uniform"
                    print(f"[Kumo {model_size}/{variant} E={estimator_count}] train<={train_through} valid={valid_month} context<={min(cap,len(train_x))} sampling={sampling}", flush=True)
                    pred, run_meta = fit_predict(
                        train_x, train_y, data.loc[tr, "mes"].to_numpy(), query_x,
                        size=model_size, estimators=estimator_count,
                        context_limit=cap, sampling=sampling, seed=SEED,
                        device=device,
                    )
                    train_months = data.loc[tr, "mes"].to_numpy()
                    context_client_ids = set()
                    for estimator_i in range(estimator_count):
                        context_ix = stratified_indices(
                            train_months, min(cap, len(train_x)), sampling,
                            SEED + estimator_i * 1009,
                        )
                        context_client_ids.update(data.loc[tr, "id_cliente"].iloc[context_ix].tolist())
                    query_client_ids = set(data.loc[va, "id_cliente"].tolist())
                    run_meta["context_unique_clients"] = len(context_client_ids)
                    run_meta["query_unique_clients"] = len(query_client_ids)
                    run_meta["context_query_client_overlap"] = len(context_client_ids & query_client_ids)
                    run_meta["train_rows"] = len(train_x)
                    valid = data.loc[va, ["id_cliente", "mes", "objetivo"]].reset_index(drop=True).copy()
                    valid["prediccion"] = pred
                    valid["model_size"] = model_size
                    valid["variant"] = variant
                    valid["estimators"] = estimator_count
                    valid["sampling"] = sampling
                    valid["context_rows_per_estimator"] = run_meta["context_rows_per_estimator"]
                    prediction_frames.append(valid)
                    all_predictions.setdefault((model_size, f"{variant}-E{estimator_count}-C{run_meta['context_rows_per_estimator']}-{sampling}"), []).append(valid)
                    score = metrics_for(valid["objetivo"], pred)
                    record_rows.append({
                        "model": "KumoTabular", "variant": variant,
                        "config": f"{model_size}; estimators={estimator_count}; context={run_meta['context_rows_per_estimator']}; sampling={sampling}",
                        "model_size": model_size, "estimators": estimator_count,
                        "sampling": sampling, "train_through": train_through,
                        "valid_month": valid_month, **score, **run_meta,
                        "checkpoint": f"nvidia/Kumo-Tabular@v1.0.0/{model_size}/classifier.pt",
                    })
                    # Persist every completed fold so a hardware failure never
                    # silently discards completed rolling work.
                    pd.DataFrame(record_rows).to_csv(run_dir / "fold_results.csv", index=False)
                    pd.concat(prediction_frames, ignore_index=True).to_csv(prediction_path, index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(prediction_path, index=False)
    records = pd.DataFrame(record_rows)
    for column in ("context_unique_clients", "query_unique_clients", "context_query_client_overlap"):
        if column not in records:
            records[column] = np.nan
    for i, row in records.iterrows():
        if pd.notna(row["context_query_client_overlap"]):
            continue
        tr = data["mes"].le(int(row["train_through"])).to_numpy()
        va = data["mes"].eq(int(row["valid_month"])).to_numpy()
        train_months = data.loc[tr, "mes"].to_numpy()
        context_clients = set()
        for estimator_i in range(int(row["estimators"])):
            ix = stratified_indices(
                train_months, int(row["context_rows_per_estimator"]),
                str(row["sampling"]), SEED + estimator_i * 1009,
            )
            context_clients.update(data.loc[tr, "id_cliente"].iloc[ix].tolist())
        query_clients = set(data.loc[va, "id_cliente"].tolist())
        records.loc[i, "context_unique_clients"] = len(context_clients)
        records.loc[i, "query_unique_clients"] = len(query_clients)
        records.loc[i, "context_query_client_overlap"] = len(context_clients & query_clients)
    records.to_csv(run_dir / "fold_results.csv", index=False)
    summary_rows = []
    for key, group in records.groupby(["model_size", "variant", "estimators", "context_rows_per_estimator", "sampling"], sort=False):
        by_month = {int(row.valid_month): row for row in group.itertuples()}
        values = [by_month[m].gini for m in (202609, 202610, 202611)]
        summary_rows.append({
            "model": "KumoTabular", "variant": key[1], "config": f"{key[0]}; E={key[2]}; context={key[3]}; {key[4]}",
            "size": key[0], "estimators": key[2], "context_rows_per_estimator": key[3], "sampling": key[4],
            "gini_sep": values[0], "gini_oct": values[1], "gini_nov": values[2],
            "mean_gini": float(np.mean(values)), "std_gini": float(np.std(values)),
            "delta_vs_lgbm_d": float(np.mean(values) - BASELINE_GINI),
            "roc_auc_sep": by_month[202609].roc_auc, "roc_auc_oct": by_month[202610].roc_auc, "roc_auc_nov": by_month[202611].roc_auc,
            "average_precision_sep": by_month[202609].pr_auc, "average_precision_oct": by_month[202610].pr_auc, "average_precision_nov": by_month[202611].pr_auc,
        })
    kumo_summary = pd.DataFrame(summary_rows)
    # Load the matching official LightGBM D month_index predictions for paired diagnostics and blends.
    lgb_path = Path("experiments/ablation/runs/20261001T051935820299Z_7e1c7547/lightgbm/D_month_index/rolling_predictions.csv")
    lgb = pd.read_csv(lgb_path)
    ensemble_rows = []
    diversity_rows = []
    for row in kumo_summary.sort_values("mean_gini", ascending=False).itertuples():
        config_values = row.config.split("; ")
        size_value = config_values[0]
        e_value = int(config_values[1].split("=")[1])
        context_value = int(config_values[2].split("=")[1])
        sample_value = config_values[3]
        key = (size_value, f"{row.variant}-E{e_value}-C{context_value}-{sample_value}")
        if key not in all_predictions:
            continue
        folds = all_predictions[key]
        if len(folds) != 3:
            continue
        corr_by_month = []
        for fold in folds:
            month = int(fold.mes.iloc[0])
            reference = lgb[lgb.mes.eq(month)].reset_index(drop=True)
            if not np.array_equal(fold.id_cliente.to_numpy(), reference.id_cliente.to_numpy()):
                reference = reference.set_index("id_cliente").loc[fold.id_cliente].reset_index()
            a = reference["prediccion"].to_numpy()
            b = fold["prediccion"].to_numpy()
            diversity_rows.append({"variant": row.variant, "config": row.config, "month": month, "pearson": stats.pearsonr(a,b).statistic, "spearman": stats.spearmanr(a,b).statistic})
            corr_by_month.append((a,b,fold.objetivo.to_numpy(),month))
        for weight_lgb in (0.75, 0.50, 0.25):
            for method in ("probability", "rank"):
                scores = []
                for a,b,y,month in corr_by_month:
                    if method == "rank":
                        a = stats.rankdata(a, method="average") / len(a)
                        b = stats.rankdata(b, method="average") / len(b)
                    scores.append(metrics_for(y, weight_lgb*a + (1-weight_lgb)*b)["gini"])
                ensemble_rows.append({"variant": row.variant, "kumo_config": row.config, "method": method, "weight_lgbm": weight_lgb, "weight_kumo": 1-weight_lgb, "gini_sep": scores[0], "gini_oct": scores[1], "gini_nov": scores[2], "mean_gini": float(np.mean(scores)), "std_gini": float(np.std(scores))})
    ensemble = pd.DataFrame(ensemble_rows)
    diversity = pd.DataFrame(diversity_rows)
    ensemble.to_csv(run_dir / "ensembles.csv", index=False)
    diversity.to_csv(run_dir / "diversity.csv", index=False)
    lightgbm_summary = pd.read_csv("experiments/ablation/runs/20261001T051935820299Z_7e1c7547/summary.csv")
    benchmarks = [
        ("LightGBM", "D", .25104), ("CatBoost", "A", .24932),
        ("CatBoost", "B AAAAMM", .24948), ("LightGBM", "B", .24869),
    ]
    total_elapsed = float(records.elapsed_seconds.sum())
    gpu_peaks = records.vram_peak_gib.dropna()
    previous_manifest_path = run_dir / "manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8")) if args.resume and previous_manifest_path.exists() else {}
    run_hardware = previous_manifest.get("hardware", hardware())
    run_device = previous_manifest.get("device", str(device))
    run_dtype = previous_manifest.get("dtype", "float16 autocast on CUDA; default float32 on CPU")
    manifest = {
        "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_package": "structured-data-models", "package_version": official_version(),
        "official_source": "https://github.com/NVIDIA/structured-data-models",
        "official_source_commit": "b982fbf4188ea8db17249fe81338723106100b63",
        "checkpoint_revision": "v1.0.0", "sizes_offered_by_official_code": list(SIZES),
        "checkpoint": f"nvidia/Kumo-Tabular@v1.0.0/{args.size}/classifier.pt",
        "checkpoints_by_size": {size: f"nvidia/Kumo-Tabular@v1.0.0/{size}/classifier.pt" for size in sorted(records.model_size.unique())},
        "hardware": run_hardware, "device": run_device, "dtype": run_dtype,
        "seed": SEED, "inference": {"estimators": [1,4,8], "batch_size": CHUNK_SIZE, "context_limit_cli_override": args.context_limit, "max_context_limit": args.max_context_limit, "default_context_plan": {"A_E1": args.max_context_limit, "D_E1": args.max_context_limit, "D_sampling_E1": 8192, "D_estimators_4_8": 8192, "D_medium_large_E1": args.max_context_limit}, "cache": "model.fit + model.predict; official KV cache"},
        "context": f"NVIDIA source defines no universal hard inference row cap; its benchmark exposes optional max_context_size and the model article reports training contexts up to 60,000 rows. This run configured a max-context cap of {args.max_context_limit:,} rows; realized per-estimator contexts were {sorted(int(v) for v in records['context_rows_per_estimator'].dropna().unique())}. An earlier local 100,600-row probe on GTX 1650 exceeded practical VRAM; this is a hardware observation, not a model-imposed limit.",
        "total_inference_seconds": total_elapsed, "max_vram_gib": float(gpu_peaks.max()) if len(gpu_peaks) else None,
        "fold_protocol": [{"train_through": a, "valid_month": b} for a,b in CUTS],
        "feature_definitions": {"A": "original columns excluding id_cliente, mes, objetivo", "D": "A + month_index (baseline D encoding, Jan 2026=0) + causal history fields from add_history_features"},
        "leakage_audit": {"target_only_training": True, "validation_target_passed_to_model": False, "id_cliente_predictor": False, "target_history_features": False, "history_features": "covariate-only chronological counts/first occurrence/duration; no labels", "sampled_context_query_client_overlap_by_fold": records[["valid_month", "context_query_client_overlap"]].drop_duplicates().to_dict("records")},
        "warnings": ["Scores are in-context softmax probabilities; softmax scores are not claimed to be calibrated probabilities.", "All official variants were checked in NVIDIA source; only requested/runtime-selected sizes were run."],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Official published comparisons are inserted without relabeling other metrics.
    comp_rows = [{"model": model, "variant": variant, "config": "existing rolling benchmark", "gini_sep": np.nan, "gini_oct": np.nan, "gini_nov": np.nan, "mean_gini": score, "std_gini": np.nan, "delta_vs_lgbm_d": score - BASELINE_GINI} for model,variant,score in benchmarks]
    all_table = pd.concat([pd.DataFrame(comp_rows), kumo_summary[["model","variant","config","gini_sep","gini_oct","gini_nov","mean_gini","std_gini","delta_vs_lgbm_d"]]], ignore_index=True)
    all_table.to_csv(run_dir / "summary.csv", index=False)
    best = kumo_summary.sort_values("mean_gini", ascending=False).iloc[0]
    best_d = kumo_summary[kumo_summary.variant.eq("D")].sort_values("mean_gini", ascending=False).iloc[0]
    best_ens = ensemble.sort_values("mean_gini", ascending=False).iloc[0] if len(ensemble) else None
    corr_best = diversity[(diversity.variant.eq(best_d.variant)) & (diversity.config.eq(best_d.config))] if len(diversity) else pd.DataFrame()
    corr_overall = diversity[(diversity.variant.eq(best.variant)) & (diversity.config.eq(best.config))] if len(diversity) else pd.DataFrame()
    corr_text = ", ".join(f"{int(r.month)} Pearson {r.pearson:.3f}/Spearman {r.spearman:.3f}" for r in corr_best.itertuples()) or "No D diversity records"
    corr_overall_text = ", ".join(f"{int(r.month)} Pearson {r.pearson:.3f}/Spearman {r.spearman:.3f}" for r in corr_overall.itertuples()) or "No diversity records"
    conclusion = (
        "promote" if best.mean_gini > .253
        else "competitive; requires additional bootstrap check" if best.mean_gini > .251
        else "ensemble-only (best fixed blend is exploratory; paired bootstrap required)" if best_ens is not None and best_ens.mean_gini > BASELINE_GINI
        else "ensemble-only" if best_d.mean_gini >= .248 and len(corr_best) and corr_best.spearman.mean() < .95
        else "reject"
    )
    lines = [
        f"# NVIDIA Kumo Tabular — {run_id}", "", "## Executive summary", "",
        f"Mejor Kumo: **{best.config}, variante {best.variant}**, mean rolling Gini **{best.mean_gini:.5f}** (Δ vs LightGBM D: **{best.delta_vs_lgbm_d:+.5f}**). Sep/Oct/Nov: {best.gini_sep:.5f}/{best.gini_oct:.5f}/{best.gini_nov:.5f}; std temporal {best.std_gini:.5f}.",
        f"Mejor Kumo-D para diversidad: **{best_d.config}**, mean Gini {best_d.mean_gini:.5f}; correlaciones por fold: {corr_text}.",
        f"Correlaciones de la mejor variante global ({best.variant}, {best.config}) con LightGBM D: {corr_overall_text}.",
        f"Conclusión operativa: **{conclusion}**.",
        f"Costo: {total_elapsed/60:.1f} min de inferencia acumulada, pico VRAM observado {max(gpu_peaks) if len(gpu_peaks) else float('nan'):.2f} GiB; hardware {manifest['hardware']}.",
        f"Mejor blend evaluado: variante {best_ens.variant}, {best_ens.kumo_config} con {best_ens.method}, {best_ens.weight_lgbm:.0%} LGBM / {best_ens.weight_kumo:.0%} Kumo, mean Gini {best_ens.mean_gini:.5f} (Δ vs LGBM-D {best_ens.mean_gini-BASELINE_GINI:+.5f})." if best_ens is not None else "No se pudo construir un blend.",
        "", "## Protocolo y API", "",
        "Rolling: ≤202608→202609, ≤202609→202610, ≤202610→202611. Gini=2×ROC AUC−1; media simple y desviación poblacional. No se usaron splits aleatorios ni diciembre.",
        "Se usó `sdm.models.KumoTabular(task='classification', size=..., device=...)`, `fit` con etiquetas solo del train y `predict` por lotes. Salida softmax multiclase; la columna `1` es P(clase positiva) y se usa continua, sin threshold. No se garantiza calibración.",
        "Features categóricas y booleanas se pasan con el tipo inferido desde train; el recipe oficial las alinea, convierte a numéricas y aplica sus transformaciones. `mes` se adapta al `month_index` de D existente (enero=0). Los tipos/recipe se ajustan solo con contexto de entrenamiento.",
        "", "## Tabla principal", "",
        "| Modelo | Variante | Config | Sep | Oct | Nov | Mean | Std | Δ vs LGBM-D |", "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_table.itertuples(index=False):
        fields = [getattr(r, c) for c in all_table.columns]
        lines.append(f"| {fields[0]} | {fields[1]} | {fields[2]} | {fields[3]:.5f} | {fields[4]:.5f} | {fields[5]:.5f} | {fields[6]:.5f} | {fields[7]:.5f} | {fields[8]:+.5f} |")
    lines += ["", "## ROC AUC y Average Precision por mes", "", "| Config | ROC AUC Sep | ROC AUC Oct | ROC AUC Nov | AP Sep | AP Oct | AP Nov |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in kumo_summary.itertuples():
        lines.append(f"| {r.config} / {r.variant} | {r.roc_auc_sep:.5f} | {r.roc_auc_oct:.5f} | {r.roc_auc_nov:.5f} | {r.average_precision_sep:.5f} | {r.average_precision_oct:.5f} | {r.average_precision_nov:.5f} |")
    lines += ["", "## Comparación con benchmarks", "", "Los cuatro valores históricos son rolling y se copian sin recalcularlos. No se mezclan métricas frozen ni pooled.", "", "## Ensemble", "", "| Variante | Config Kumo | Método | LGBM | Kumo | Sep | Oct | Nov | Mean | Std |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ensemble.itertuples():
        lines.append(f"| {r.variant} | {r.kumo_config} | {r.method} | {r.weight_lgbm:.2f} | {r.weight_kumo:.2f} | {r.gini_sep:.5f} | {r.gini_oct:.5f} | {r.gini_nov:.5f} | {r.mean_gini:.5f} | {r.std_gini:.5f} |")
    lines += ["", "### Diversidad por fold", "", "| Variante | Config | Mes | Pearson | Spearman |", "|---|---|---:|---:|---:|"]
    for r in diversity.itertuples():
        lines.append(f"| {r.variant} | {r.config} | {r.month} | {r.pearson:.5f} | {r.spearman:.5f} |")
    actual_contexts = sorted(int(v) for v in records["context_rows_per_estimator"].dropna().unique())
    observed_context_max = max(actual_contexts) if actual_contexts else 0
    if "train_rows" in records:
        sampled = int((records["context_rows_per_estimator"] < records["train_rows"]).sum())
        context_counts = f"Se usaron contextos por estimator de {actual_contexts}; {sampled} de {len(records)} folds-config usaron sampling determinista al quedar el train por encima del contexto."
    else:
        context_counts = f"Se usaron contextos por estimator de {actual_contexts}; el artefacto previo no registró el número de filas de train por fold."
    runtime_by_size = records.groupby("model_size").elapsed_seconds.mean().to_dict()
    runtime_note = ", ".join(f"{size}: {seconds:.1f} s/fold-config en promedio" for size, seconds in runtime_by_size.items())
    context_analysis = (
        "NVIDIA no documenta un máximo duro de inferencia; su benchmark expone `max_context_size` y su artículo describe entrenamiento con contextos de hasta 60,000 filas. "
        f"El límite configurado para los experimentos de contexto máximo fue {args.max_context_limit:,} filas; el máximo observado fue {observed_context_max:,}. {context_counts} "
        f"Tiempos promedio por fold-config: {runtime_note}. El intento exploratorio local de 100,600 filas excedió la VRAM de la GTX 1650; no se presenta como límite del modelo. "
        "El código oficial soporta `fit`/`predict` con KV cache y contextos de estimator en batch; esta arena promedió 1, 4 u 8 estimators con contextos muestreados de forma determinista. `predict` procesa batches de 256."
    )
    lines += ["", "## Context-size analysis", "", context_analysis, "", "## Leakage audit", "", "- `model.fit` recibe únicamente etiquetas de `mes <= train_through`; nunca recibe `objetivo` del mes validado.", "- `objetivo` no entra a X y no se usa target histórico.", "- `id_cliente` no entra a X; sólo alinea predicciones y audita recurrencia. `context_query_client_overlap` por fold queda en `fold_results.csv` y `manifest.json`; la recurrencia de un cliente entre contexto pasado y query posterior está permitida.", "- Las features de historial son covariables cronológicas: observaciones previas, primera aparición y duración. No consultan etiquetas. Primera aparición y duraciones se calculan hasta el mes de cada fila.", "", "## Limitaciones", "", f"Se ejecutaron {', '.join(sorted(records.model_size.unique()))} en {manifest['hardware']['gpu'] or 'CPU'}; Medium/Large se limitaron a D, E=1 y el límite de contexto máximo configurado. Las mediciones por fold, contexto real, checkpoint y memoria están en `fold_results.csv`; hardware y versiones en `manifest.json`. Los estimators se ejecutan secuencialmente (`estimator_batch_size=1`).", "El softmax no se calibró y no se hizo búsqueda de hiperparámetros. El ganador entre mezclas usa únicamente pesos fijos 75/25, 50/50 y 25/75, pero sigue siendo la mejor fila observada entre configuraciones y folds compartidos; es exploratorio y debe validarse con bootstrap pareado/futuros meses antes de usarse.", ""]
    (run_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Results: {run_dir / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
