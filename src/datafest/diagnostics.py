"""Ordered rolling diagnostics; never selects parameters on Sep–Nov."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datafest.ablation import ROLLING_CUTS, _feature_matrix
from datafest.features import _month_ordinal
from datafest.lineage import file_record, repository_revision, verify_file_records, write_json
from datafest.models import fit_model

BASELINE = Path("experiments/ablation/runs/20261001T051935820299Z_7e1c7547")
PRIMARY_MODELS = {"LightGBM_D": "lightgbm/D_month_index", "LightGBM_B": "lightgbm/B_month_index", "CatBoost_A": "catboost/A"}
TRAJECTORY_COLUMNS = [f"interaction_{s}" for s in ("lag1", "lag2", "delta", "slope", "min", "max", "mean", "reset", "resets")]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as stream:
        temporary = stream.name
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    """Inclusive expanding history; slope uses elapsed calendar months, not row number."""
    if frame.duplicated(["id_cliente", "mes"]).any():
        raise ValueError("Duplicate customer-month")
    ordered = frame.assign(_position=np.arange(len(frame))).sort_values(["id_cliente", "mes"])
    g = ordered.groupby("id_cliente", sort=False)
    value = ordered["dias_ultima_interaccion"].astype(float)
    t = _month_ordinal(ordered["mes"])
    # Subtract each customer's origin to avoid cancellation in the slope sums.
    t = (t - t.groupby(ordered.id_cliente).transform("min")).astype(float)
    n = g.cumcount() + 1
    sum_y = value.groupby(ordered.id_cliente).cumsum()
    sum_t = t.groupby(ordered.id_cliente).cumsum()
    sum_tt = (t*t).groupby(ordered.id_cliente).cumsum()
    sum_ty = (t*value).groupby(ordered.id_cliente).cumsum()
    lag1 = g.dias_ultima_interaccion.shift(1)
    reset = value.lt(lag1)
    out = pd.DataFrame({
        "interaction_lag1": lag1, "interaction_lag2": g.dias_ultima_interaccion.shift(2),
        "interaction_delta": value-lag1,
        "interaction_slope": (n*sum_ty-sum_t*sum_y)/(n*sum_tt-sum_t**2).replace(0, np.nan),
        "interaction_min": g.dias_ultima_interaccion.cummin(),
        "interaction_max": g.dias_ultima_interaccion.cummax(),
        "interaction_mean": sum_y/n, "interaction_reset": reset.astype(int),
        "interaction_resets": reset.groupby(ordered.id_cliente).cumsum().astype(int),
        "_position": ordered._position,
    }).sort_values("_position").drop(columns="_position")
    return out.reset_index(drop=True)


def probability_frame(valid: pd.DataFrame, prediction) -> pd.DataFrame:
    p = np.asarray(prediction, dtype=float)
    if p.shape != (len(valid),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Expected exactly one finite probability per validation row")
    out = valid[["id_cliente", "mes", "objetivo"]].copy().reset_index(drop=True)
    out["prediccion"] = p
    return out


def metrics(frame: pd.DataFrame) -> dict:
    scores = {str(m): float(2*roc_auc_score(g.objetivo, g.prediccion)-1) for m,g in frame.groupby("mes")}
    return {"gini_by_month": scores, "mean_rolling_gini": float(np.mean(list(scores.values()))),
            "temporal_std_ddof0": float(np.std(list(scores.values())))}


def manifest(directory: Path, config: dict, inputs: list[Path], features: list[Path] = ()) -> None:
    write_json(directory / "config.json", config)
    files = [p for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"]
    write_json(directory / "manifest.json", {
        "schema_version": 1, "run_id": directory.name, "repository": repository_revision(),
        "protocol": {"rolling_cuts": ROLLING_CUTS, "metric": "monthly Gini and simple rolling mean",
                     "temporal_std": "description only; not statistical uncertainty", "tuning": False},
        "inputs": [file_record(p) for p in inputs], "features": [file_record(p) for p in features],
        "code": [file_record(p) for p in Path("src/datafest").glob("*.py")],
        "outputs": [file_record(p) for p in files], "parameters": config,
        "split_artifacts": [file_record(p) for p in (directory/"split_assignments.csv",) if p.is_file()],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    })


def split_assignments(train: pd.DataFrame, directory: Path):
    rows=[]
    for through,month in ROLLING_CUTS:
        for role,mask in (("train",train.mes.le(through)),("validation",train.mes.eq(month))):
            rows.append(pd.DataFrame({"fold":month,"role":role,"source_row":np.flatnonzero(mask)}))
    atomic_csv(pd.concat(rows,ignore_index=True),directory/"split_assignments.csv")


def load_baselines(baseline: Path, train: pd.DataFrame, train_path=Path("data/train.csv")) -> dict[str, pd.DataFrame]:
    expected = train.loc[train.mes.isin([b for _,b in ROLLING_CUTS]), ["id_cliente", "mes", "objetivo"]].reset_index(drop=True)
    parent = json.loads((baseline/"manifest.json").read_text())
    if file_record(train_path)["sha256"] != parent["inputs"][0]["sha256"]:
        raise ValueError("Baseline source hash differs")
    result = {}
    for name, location in PRIMARY_MODELS.items():
        path = baseline/location/"rolling_predictions.csv"
        record = next(r for run in parent["runs"] for r in run["predictions"] if r["path"] == str(path))
        if verify_file_records([record]):
            raise ValueError(f"Corrupt baseline: {path}")
        frame = pd.read_csv(path)
        pd.testing.assert_frame_equal(frame[expected.columns], expected, check_dtype=False)
        probability_frame(expected, frame.prediccion)
        result[name] = frame
    return result


def _weighted_gini_preparation(y, p, customer):
    order = np.argsort(p, kind="stable")
    p, y, customer = np.asarray(p)[order], np.asarray(y)[order], np.asarray(customer)[order]
    ties = np.cumsum(np.r_[True, p[1:] != p[:-1]])-1
    def score(counts):
        w = counts[customer]
        pos = np.bincount(ties, weights=w*y)
        neg = np.bincount(ties, weights=w*(1-y), minlength=len(pos))
        if pos.sum() == 0 or neg.sum() == 0:
            return np.nan
        auc = np.sum(pos*(np.cumsum(neg)-0.5*neg))/(pos.sum()*neg.sum())
        return 2*auc-1
    return score


def paired_bootstrap(baselines: dict, directory: Path, seed=42, replicates=2000, baseline=BASELINE):
    directory.mkdir(parents=True, exist_ok=True)
    reference = next(iter(baselines.values()))
    ids = pd.Index(reference.id_cliente.unique())
    customer = ids.get_indexer(reference.id_cliente)
    names = list(baselines)
    months = [b for _,b in ROLLING_CUTS]
    functions = [[_weighted_gini_preparation(f.loc[f.mes.eq(m), "objetivo"].to_numpy(),
                    f.loc[f.mes.eq(m), "prediccion"].to_numpy(), customer[f.mes.eq(m)]) for m in months]
                 for f in baselines.values()]
    draws = np.empty((replicates, len(names), len(months)))
    rng = np.random.default_rng(seed)
    for r in range(replicates):
        counts = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        draws[r] = [[func(counts) for func in funcs] for funcs in functions]
    if not np.isfinite(draws).all():
        raise ValueError("Bootstrap replicate lacks a class; cannot silently discard draws")
    np.savez_compressed(directory/"bootstrap_gini_draws.npz", gini=draws, models=names, months=months)
    rows = []
    for a,b in itertools.combinations(range(len(names)), 2):
        point = np.array(list(metrics(baselines[names[a]])["gini_by_month"].values())) - np.array(list(metrics(baselines[names[b]])["gini_by_month"].values()))
        delta = draws[:,a]-draws[:,b]
        rolling = delta.mean(axis=1)
        observed = float(point.mean())
        p = float((1+np.sum(np.abs(rolling-observed) >= abs(observed)))/(replicates+1))
        for j,m in enumerate(months+["rolling_mean"]):
            samples = delta[:,j] if j < len(months) else rolling
            lo,hi = np.quantile(samples, [0.025,0.975])
            rows.append({"pair": f"{names[a]} - {names[b]}", "month": str(m),
                         "delta_gini": float(point[j]) if j<len(months) else observed,
                         "ci95_low": float(lo), "ci95_high": float(hi),
                         "p_two_sided_centered_bootstrap": p if j==len(months) else None})
    rolling_rows = [r for r in rows if r["month"]=="rolling_mean"]
    running = 0.0
    for rank,row in enumerate(sorted(rolling_rows,key=lambda r:r["p_two_sided_centered_bootstrap"])):
        running = max(running,min(1., (3-rank)*row["p_two_sided_centered_bootstrap"]))
        row["p_holm_three_rolling_pairs"] = running
    atomic_csv(pd.DataFrame(rows), directory/"metrics.csv")
    atomic_csv(reference, directory/"validation_rows.csv")
    for name,frame in baselines.items():
        atomic_csv(frame,directory/f"predictions_{name}.csv")
    manifest(directory, {"seed": seed,"replicates":replicates,"unit":"id_cliente",
             "resampling":"one shared customer draw across all months and models",
             "ci":"unadjusted percentile 95%", "holm_family":"three rolling mean pair comparisons",
             "p_value":"two-sided centered bootstrap with plus-one correction"},
             [baseline/location/"rolling_predictions.csv" for location in PRIMARY_MODELS.values()])
    return rows


def run_trajectory(train: pd.DataFrame, baseline: Path, directory: Path, train_path=Path("data/train.csv")):
    x = pd.concat([_feature_matrix(train, "D_month_index"), trajectory(train)], axis=1)
    directory.mkdir(parents=True, exist_ok=True)
    split_assignments(train,directory)
    x.to_csv(directory/"features.csv.gz",index=False)
    choice = json.loads((baseline/"manifest.json").read_text())["frozen_configurations"]["lightgbm"]
    write_json(directory/"feature_schema.json", {"columns": list(x.columns),"dynamic":TRAJECTORY_COLUMNS})
    predictions = []
    for through,month in ROLLING_CUTS:
        print(f"[trajectory] {through} -> {month}", flush=True)
        tm,vm = train.mes.le(through), train.mes.eq(month)
        fitted = fit_model("lightgbm",x.loc[tm],train.loc[tm,"objetivo"], config=choice["config"],
                           iterations=choice["iterations"],early_stopping_rounds=None)
        frame = probability_frame(train.loc[vm],fitted.predict_proba(x.loc[vm]))
        atomic_csv(frame,directory/f"predictions_{month}.csv")
        fitted.save(str(directory/f"model_{month}.txt"))
        write_json(directory/f"preprocessing_{month}.json", {"category_maps": fitted.category_maps})
        predictions.append(frame)
    combined = pd.concat(predictions,ignore_index=True)
    atomic_csv(combined,directory/"rolling_predictions.csv")
    write_json(directory/"metrics.json", metrics(combined))
    manifest(directory, {"frozen_lightgbm":choice,"features":"D_month_index + interaction trajectory",
             "slope_units":"days of interaction per calendar month; absent with one observation",
             "reset":"current observed value < previous observed value"},
             [train_path, baseline/"manifest.json"], [directory/"feature_schema.json",directory/"features.csv.gz"])
    return metrics(combined)


def save_pickle(value, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        pickle.dump(value,stream)
        temporary = stream.name
    os.replace(temporary,path)


def run_diagnostics(data_dir="data", baseline=BASELINE, root="experiments/diagnostics"):
    from datafest.survival import run_survival
    baseline, root = Path(baseline), Path(root)
    train = pd.read_csv(Path(data_dir)/"train.csv")
    train.objetivo = train.objetivo.astype(int)
    train_path=Path(data_dir)/"train.csv"
    baselines = load_baselines(baseline,train,train_path)
    bootstrap = paired_bootstrap(baselines,root/"01_bootstrap",baseline=baseline)
    dynamic = run_trajectory(train,baseline,root/"02_trajectory",train_path)
    survival = run_survival(train,root/"03_survival",train_path)
    result = {"baseline": {n: metrics(f) for n,f in baselines.items()}, "bootstrap":bootstrap,
              "trajectory":dynamic,"survival":survival}
    write_json(root/"summary.json",result)
    return result
