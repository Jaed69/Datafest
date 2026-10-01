"""Monthly hazard and Breslow Cox counting-process likelihoods.

Each observed customer-month is (tenure-1, tenure]. Missing observation
months contribute no fabricated covariates or risk intervals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from datafest.ablation import ROLLING_CUTS, _feature_matrix
from datafest.diagnostics import atomic_csv, manifest, metrics, probability_frame, save_pickle, split_assignments
from datafest.features import _month_ordinal
from datafest.lineage import write_json


def risk_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.assign(_position=np.arange(len(frame))).sort_values(["id_cliente","mes"])
    previous_events = ordered.groupby("id_cliente").objetivo.cumsum()-ordered.objetivo
    if previous_events.gt(0).any():
        raise ValueError("Rows after first conversion violate the risk set")
    month = _month_ordinal(ordered.mes)
    origin = month.groupby(ordered.id_cliente).transform("min")
    result = ordered[["id_cliente","objetivo","_position"]].copy()
    result["start"] = (month-origin).astype(int)
    result["stop"] = result.start+1
    result["censored_last_observation"] = (~ordered.id_cliente.duplicated(keep="last")) & ordered.objetivo.eq(0)
    return result.sort_values("_position").drop(columns="_position").reset_index(drop=True)


class Design:
    """Fit encodings/scale only on the fold's training covariates."""
    def fit_transform(self, frame):
        encoded = pd.get_dummies(frame,drop_first=True,dtype=float).astype(float)
        self.columns = list(encoded.columns[encoded.nunique().gt(1)])
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(encoded[self.columns])

    def transform(self, frame):
        # Explicit training categorical levels prevent drop_first changing the
        # reference category when a validation month lacks a training category.
        encoded = pd.get_dummies(frame,drop_first=True,dtype=float).astype(float)
        return self.scaler.transform(encoded.reindex(columns=self.columns,fill_value=0))


def cloglog_objective(beta, x, y, ridge):
    eta = x@beta
    h = np.exp(np.clip(eta,-40,40))
    # P(event) = -expm1(-h); stable at either end of the link.
    p = -np.expm1(-h)
    loss = np.sum(np.where(y==1,-np.log(np.maximum(p,1e-300)),h))/len(y)
    factor = np.zeros_like(h)
    small = h < 50
    factor[small] = h[small]/np.expm1(h[small])
    derivative = np.where(y==1,-factor,h)
    penalty = beta.copy()
    penalty[0] = 0
    return loss + 0.5*ridge*np.dot(penalty,penalty), x.T@derivative/len(y)+ridge*penalty


def cox_objective(beta, x, events, starts, stops, ridge):
    """Breslow ties, with exact start < event_time <= stop membership."""
    eta = x@beta
    loss = -float(events@eta)
    gradient = -(x.T@events)
    for t in np.unique(stops[events.astype(bool)]):
        risk = (starts<t)&(stops>=t)
        d = int(events[stops==t].sum())
        risk_eta = eta[risk]
        shift = risk_eta.max()
        weight = np.exp(risk_eta-shift)
        denominator = weight.sum()
        loss += d*(shift+np.log(denominator))
        gradient += d*(x[risk].T@weight)/denominator
    n_events = events.sum()
    return loss/n_events+0.5*ridge*np.dot(beta,beta), gradient/n_events+ridge*beta


def _fit(objective, dimension, args):
    fit = minimize(objective,np.zeros(dimension),args=args,jac=True,method="L-BFGS-B",
                   options={"maxiter":1000,"ftol":1e-11,"gtol":1e-6,"maxls":50})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Survival convergence failure: {fit.message}")
    return fit


def run_survival(train: pd.DataFrame, directory: Path, train_path=Path("data/train.csv")):
    directory.mkdir(parents=True,exist_ok=True)
    x = _feature_matrix(train,"D_month_index")
    # Deterministic time aliases are handled by baseline time or month_index;
    # including them separately would make linear coefficients unidentified.
    x = x.drop(columns=["mes_primera_aparicion","meses_en_riesgo","meses_desde_entrada"])
    split_assignments(train,directory)
    x.to_csv(directory/"features.csv.gz",index=False)
    all_predictions = {"cloglog":[],"cox":[]}
    audit = []
    for through,month in ROLLING_CUTS:
        print(f"[survival] {through} -> {month}",flush=True)
        tm,vm = train.mes.le(through),train.mes.eq(month)
        past = train.loc[train.mes.le(month)].reset_index(drop=True)
        intervals = risk_intervals(past)
        training_intervals = risk_intervals(train.loc[tm].reset_index(drop=True))
        valid_intervals = intervals.loc[past.mes.eq(month)]
        xf,xv = x.loc[tm].copy(),x.loc[vm].copy()
        for col in xf.select_dtypes(include=["object","str"]).columns:
            levels = sorted(xf[col].unique())
            xf[col] = pd.Categorical(xf[col],categories=levels)
            xv[col] = pd.Categorical(xv[col],categories=levels)
        design = Design()
        xt = design.fit_transform(xf)
        xv = design.transform(xv)
        y = train.loc[tm,"objetivo"].to_numpy(float)
        start = training_intervals.start.to_numpy()
        stop = training_intervals.stop.to_numpy()
        age = stop.astype(int)
        max_age = age.max()
        # Base month-of-risk effect; extrapolate unseen tenure with the last
        # fitted tenure factor. Calendar trend remains the D month_index.
        baseline = np.eye(max_age)[age-1][:,1:]
        v_age = valid_intervals.stop.to_numpy(int)
        v_base = np.eye(max_age)[np.minimum(v_age,max_age)-1][:,1:]
        hazard_x = np.column_stack([np.ones(len(xt)),xt,baseline])
        hazard_v = np.column_stack([np.ones(len(xv)),xv,v_base])
        clog = _fit(cloglog_objective,hazard_x.shape[1],(hazard_x,y,1e-6))
        clog_probability = -np.expm1(-np.exp(np.clip(hazard_v@clog.x,-40,40)))
        cox = _fit(cox_objective,xt.shape[1],(xt,y,start,stop,1e-4))
        train_risk_score = np.exp(np.clip(xt@cox.x,-40,40))
        increments = {}
        for t in range(1,max_age+1):
            d = y[stop==t].sum()
            denom = train_risk_score[(start<t)&(stop>=t)].sum()
            increments[t] = float(d/denom) if denom else 0.
        v_increment = np.array([increments[min(t,max_age)] for t in v_age])
        cox_probability = -np.expm1(-v_increment*np.exp(np.clip(xv@cox.x,-40,40)))
        for model,p in (("cloglog",clog_probability),("cox",cox_probability)):
            frame = probability_frame(train.loc[vm],p)
            atomic_csv(frame,directory/model/f"predictions_{month}.csv")
            all_predictions[model].append(frame)
        save_pickle({"design":design,"category_levels":{c:list(xf[c].cat.categories) for c in xf.select_dtypes("category")},
                     "cloglog_beta":clog.x,"cox_beta":cox.x,"baseline_increments":increments,
                     "max_age":int(max_age),"feature_columns":list(x.columns)},directory/f"models_{month}.pkl")
        atomic_csv(training_intervals,directory/f"risk_intervals_{month}.csv")
        audit.append({"train_through":through,"validation_month":month,"risk_rows":len(xt),
                      "events":int(y.sum()),"right_censored_customers":int(training_intervals.censored_last_observation.sum()),
                      "cloglog_iterations":int(clog.nit),"cox_iterations":int(cox.nit),
                      "cloglog_gradient_max":float(np.abs(clog.jac).max()),"cox_gradient_max":float(np.abs(cox.jac).max()),
                      "validation_beyond_training_tenure":int((v_age>max_age).sum()),
                      "baseline_hazard_training_only":increments})
    results = {}
    for model,frames in all_predictions.items():
        combined = pd.concat(frames,ignore_index=True)
        atomic_csv(combined,directory/model/"rolling_predictions.csv")
        results[model] = metrics(combined)
        write_json(directory/model/"metrics.json",results[model])
    write_json(directory/"audit.json",audit)
    write_json(directory/"feature_schema.json",{"columns":list(x.columns)})
    manifest(directory,{"base_features":"D_month_index minus deterministic time aliases",
             "time":"months since first observation; delayed observations keep actual month spacing",
             "risk_interval":"(tenure-1, tenure] only for observed rows; exit at event; last non-event is right censored",
             "cloglog_baseline":"categorical month in risk; calendar trend via D month_index",
             "cox_ties":"Breslow", "cox_baseline":"training-only monthly Breslow increments",
             "unseen_tenure":"carry last training monthly baseline increment / tenure factor",
             "fixed_ridge":{"cloglog":1e-6,"cox":1e-4},"ridge_selection":"fixed numeric stabilization, no validation tuning"},
             [train_path],[directory/"feature_schema.json",directory/"features.csv.gz"])
    return results
