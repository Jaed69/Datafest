"""Resumable full-validation foundation folds, explicitly pinned to TabPFN-3."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
import pandas as pd

from datafest.ablation import ROLLING_CUTS, _feature_matrix
from datafest.diagnostics import atomic_csv, manifest, metrics, probability_frame, save_pickle, split_assignments
from datafest.lineage import file_record, sha256_file, verify_file_records, write_json


class FoldTimeout(TimeoutError):
    pass


def _timeout(signum, frame):
    raise FoldTimeout("Fold exceeded 1800 seconds; resume on Colab")


def completed_fold(directory: Path, expected: pd.DataFrame, identity: dict):
    state_path = directory/"state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    if state.get("status") != "complete":
        return None
    if state["identity"] != identity:
        raise ValueError("Resume identity differs (data, features, code or model version)")
    errors = verify_file_records(state["artifacts"])
    if errors:
        raise ValueError(f"Completed fold artifact corruption: {errors}")
    result = pd.read_csv(directory/"predictions.csv")
    pd.testing.assert_frame_equal(result[["id_cliente","mes","objetivo"]],
                                  expected[["id_cliente","mes","objetivo"]].reset_index(drop=True),check_dtype=False)
    probability_frame(expected,result.prediccion)
    return result


def _checkpoint_records(estimator, model, directory):
    import torch
    checkpoint_dir = directory/"checkpoint"
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    if model == "tabpfn3":
        paths = estimator.model_path
        if not isinstance(paths,(list,tuple)):
            paths = [paths]
        for path in paths:
            path=Path(path)
            if not path.is_file() or "v3" not in path.name or "v3.5" in path.name:
                raise RuntimeError(f"Expected an explicit TabPFN-3 checkpoint, got {path}")
            shutil.copy2(path,checkpoint_dir/path.name)
    else:
        torch.save(estimator.model.state_dict(),checkpoint_dir/"tabfm_v1_0_0.pt")
        write_json(checkpoint_dir/"config.json",getattr(estimator.model,"config",{}))
    return [file_record(p) for p in checkpoint_dir.iterdir() if p.is_file()]


def _make_estimator(model, device):
    if model == "tabpfn3":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        estimator = TabPFNClassifier.create_default_for_version(ModelVersion.V3, device=device)
        if "v3" not in str(estimator.model_path) or "v3.5" in str(estimator.model_path):
            raise RuntimeError("TabPFN-3 version selector failed")
        context = {"version":"TabPFN-3","selection":"ModelVersion.V3",
                   "documented_max_training_rows":1000000,"external_training_subsample":False}
    elif model == "tabfm":
        from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch
        backbone=tabfm_v1_0_0_pytorch.load(device=device)
        # The plan fixes the published 100-row context. Current upstream's
        # constructor says None, in conflict with its own README; make that
        # documented setting explicit instead of silently using all 80k rows.
        from inspect import signature
        runtime_default=signature(TabFMClassifier).parameters["max_num_rows"].default
        estimator=TabFMClassifier(model=backbone,max_num_rows=100)
        context={"version":"TabFM v1.0.0","documented_default_max_context_rows":100,
                 "runtime_default_max_num_rows":runtime_default,"effective_max_num_rows":100,
                 "documentation":"https://github.com/google-research/tabfm",
                 "note":"Explicit max_num_rows=100 preserves the plan's published context default despite constructor drift."}
    else:
        raise ValueError(model)
    return estimator,context


def run_foundation(model: str, data_dir: Path, root: Path, device="cuda", fold=None):
    train=pd.read_csv(data_dir/"train.csv")
    train.objetivo=train.objetivo.astype(int)
    x=_feature_matrix(train,"D_month_index")
    target=root/model
    target.mkdir(parents=True,exist_ok=True)
    if not (target/"split_assignments.csv").exists():
        split_assignments(train,target)
    if not (target/"features.csv.gz").exists():
        x.to_csv(target/"features.csv.gz",index=False)
    # Feature prefix invariance on the actual dataset, not just toy fixtures.
    for through,_ in ROLLING_CUTS:
        mask=train.mes.le(through)
        pd.testing.assert_frame_equal(_feature_matrix(train.loc[mask],"D_month_index"),x.loc[mask].reset_index(drop=True))
    write_json(target/"feature_schema.json",{"columns":list(x.columns),"dtypes":{c:str(t) for c,t in x.dtypes.items()}})
    versions={name:importlib.metadata.version(name) for name in ("numpy","pandas","scikit-learn","torch", "tabpfn" if model=="tabpfn3" else "tabfm")}
    identity={"train_sha256":sha256_file(data_dir/"train.csv"),"model":model,
              "feature_columns":list(x.columns),"versions":versions,
              "features_sha256":sha256_file(target/"features.csv.gz"),
              "code_sha256":{p.name:sha256_file(p) for p in [Path(__file__),Path("src/datafest/features.py"),Path("src/datafest/ablation.py")]}}
    config={"model":model,"features":"exact D_month_index","defaults":True,"device":device,
            "timeout_seconds":1800,"versions":versions,"identity":identity}
    if (root/"source_snapshot.json").exists():
        config["source_snapshot"]=json.loads((root/"source_snapshot.json").read_text())
    completed=[]
    for through,month in ROLLING_CUTS:
        tm,vm=train.mes.le(through),train.mes.eq(month)
        directory=target/str(month)
        directory.mkdir(parents=True,exist_ok=True)
        existing=completed_fold(directory,train.loc[vm],identity)
        if existing is not None:
            completed.append(existing)
            continue
        if fold is not None and month!=fold:
            continue
        write_json(directory/"state.json",{"status":"running","identity":identity,"train_through":through,
                    "validation_month":month,"n_train":int(tm.sum()),"n_validation":int(vm.sum())})
        start=time.monotonic()
        previous=signal.signal(signal.SIGALRM,_timeout)
        signal.alarm(1800)
        try:
            print(f"[{model}] {through} -> {month} on {device}",flush=True)
            estimator,context=_make_estimator(model,device)
            parameters={k:v for k,v in estimator.get_params(deep=False).items() if k!="model"}
            write_json(directory/"parameters.json",{"defaults":parameters,"context":context})
            estimator.fit(x.loc[tm].reset_index(drop=True),train.loc[tm,"objetivo"].to_numpy(int))
            if model=="tabfm":
                patterns=estimator.ensemble_generator_.row_subsample_patterns_
                context["actual_context_rows_per_member"]=[len(rows) if rows is not None else int(tm.sum())
                    for rows_list in patterns.values() for rows in rows_list]
            else:
                context["resolved_n_estimators"]=getattr(estimator,"n_estimators_",None)
                context["inference_config"]=str(getattr(estimator,"inference_config_",None))
            write_json(directory/"parameters.json",{"defaults":parameters,"context":context})
            checkpoints=_checkpoint_records(estimator,model,directory)
            # Full valid set is passed once, allowing the estimator to control
            # its own inference batching without changing default context.
            positive=np.flatnonzero(np.asarray(estimator.classes_)==1)
            if len(positive)!=1:
                raise ValueError("Positive class missing")
            prediction=estimator.predict_proba(x.loc[vm].reset_index(drop=True))[:,positive[0]]
            frame=probability_frame(train.loc[vm],prediction)
            atomic_csv(frame,directory/"predictions.csv")
            write_json(directory/"metrics.json",metrics(frame))
            save_pickle(estimator,directory/"fitted_estimator.pkl")
            if model=="tabfm":
                generator=estimator.ensemble_generator_
                # Configurations include exact per-member row subsamples.
                save_pickle(generator,directory/"context_generator.pkl")
            artifacts=[file_record(p) for p in directory.rglob("*") if p.is_file() and p.name not in {"state.json","manifest.json"}]
            write_json(directory/"state.json",{"status":"complete","identity":identity,"train_through":through,
                       "validation_month":month,"elapsed_seconds":time.monotonic()-start,
                       "n_train":int(tm.sum()),"n_validation":len(frame),"context":context,"artifacts":artifacts})
            manifest(directory,config,[data_dir/"train.csv"],[target/"feature_schema.json",target/"features.csv.gz",target/"split_assignments.csv"])
            completed.append(frame)
        except Exception as error:
            write_json(directory/"state.json",{"status":"failed","identity":identity,"train_through":through,
                       "validation_month":month,"elapsed_seconds":time.monotonic()-start,
                       "error_type":type(error).__name__,"error":str(error),
                       "traceback":traceback.format_exc(),"next_action":"resume same fold on Colab for memory/timeout failures"})
            manifest(directory,config,[data_dir/"train.csv"],[target/"feature_schema.json",target/"features.csv.gz",target/"split_assignments.csv"])
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM,previous)
    if len(completed)==len(ROLLING_CUTS):
        predictions=pd.concat(completed,ignore_index=True)
        atomic_csv(predictions,target/"rolling_predictions.csv")
        result=metrics(predictions)
        write_json(target/"metrics.json",result)
        write_json(target/"state.json",{"status":"complete","folds":[b for _,b in ROLLING_CUTS]})
    else:
        result={"status":"partial","completed_months":[int(f.mes.iloc[0]) for f in completed]}
        write_json(target/"state.json",result)
    manifest(target,config,[data_dir/"train.csv"],[target/"feature_schema.json",target/"features.csv.gz"])
    return result
