import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative
from sklearn.metrics import roc_auc_score

from datafest.diagnostics import _weighted_gini_preparation, probability_frame, trajectory
from datafest.survival import cloglog_objective, cox_objective, risk_intervals, Design
from datafest.foundation import completed_fold
from datafest.lineage import file_record, write_json


def sample():
    return pd.DataFrame({"id_cliente":[1,2,1,1,2],"mes":[202601,202601,202603,202604,202602],
                         "dias_ultima_interaccion":[10,50,30,5,40],"objetivo":[0,0,0,1,0]})


def test_trajectory_prefix_invariance_and_shuffled_order():
    data=sample()
    features=trajectory(data)
    for month in sorted(data.mes.unique()):
        mask=data.mes.le(month)
        pd.testing.assert_frame_equal(trajectory(data.loc[mask]),features.loc[mask].reset_index(drop=True))
    reordered=data.sample(frac=1,random_state=7)
    np.testing.assert_allclose(trajectory(reordered),features.iloc[reordered.index],equal_nan=True)
    changed=data.copy()
    changed.loc[changed.mes.eq(202604),"dias_ultima_interaccion"]=9999
    np.testing.assert_allclose(trajectory(changed).iloc[:3],features.iloc[:3],equal_nan=True)
    assert features.loc[2,"interaction_slope"] == 10 # 20 days / 2 calendar months
    assert features.loc[3,"interaction_reset"] == 1
    assert features.loc[3,"interaction_resets"] == 1
    assert features.loc[3,"interaction_lag2"] == 10
    assert np.isnan(features.loc[0,"interaction_lag1"])


def test_cluster_weighted_auc_matches_materialized_customer_draws_with_ties():
    y=np.array([0,1,0,1,1,0]); p=np.array([.1,.1,.6,.8,.6,.6])
    ids=np.array([0,1,0,2,1,2]); counts=np.array([2,3,1])
    rows=np.repeat(np.arange(len(y)),counts[ids])
    assert _weighted_gini_preparation(y,p,ids)(counts) == pytest.approx(2*roc_auc_score(y[rows],p[rows])-1)


def test_probability_coverage_and_risk_exit():
    data=sample()
    assert len(probability_frame(data,np.full(len(data),.2)))==len(data)
    for p in ([.2],np.full(len(data),np.nan),np.full(len(data),1.1)):
        with pytest.raises(ValueError): probability_frame(data,p)
    intervals=risk_intervals(data)
    assert intervals.loc[2,"start"] == 2
    assert intervals.loc[3,"stop"] == 4
    assert intervals.censored_last_observation.sum()==1
    bad=pd.concat([data,pd.DataFrame({"id_cliente":[1],"mes":[202605],"objetivo":[0]})])
    with pytest.raises(ValueError): risk_intervals(bad)


def test_likelihood_gradients_and_start_stop_membership():
    x=np.array([[1.,-.2],[1.,.5],[1.,.9],[1.,-.7]])
    beta=np.array([-.3,.4]); y=np.array([0.,1.,1.,0.])
    args=(x,y,1e-6)
    np.testing.assert_allclose(cloglog_objective(beta,*args)[1],
                              approx_derivative(lambda b:cloglog_objective(b,*args)[0],beta).ravel(),atol=1e-7)
    start=np.array([0,0,1,1]); stop=np.array([1,1,2,2])
    args=(x,y,start,stop,1e-4)
    np.testing.assert_allclose(cox_objective(beta,*args)[1],
                              approx_derivative(lambda b:cox_objective(b,*args)[0],beta).ravel(),atol=1e-7)
    assert cox_objective(np.zeros(2),*args)[0] == pytest.approx(np.log(2))


def test_design_keeps_training_category_reference():
    train=pd.DataFrame({"cat":pd.Categorical(["a","b","c"],categories=["a","b","c"]),"v":[1,2,3]})
    valid=train.iloc[1:].copy()
    design=Design(); xt=design.fit_transform(train)
    np.testing.assert_allclose(design.transform(valid),xt[1:])


def test_resume_accepts_only_verified_complete_fold(tmp_path):
    expected=sample()
    predictions=probability_frame(expected,np.full(len(expected),.2))
    predictions.to_csv(tmp_path/"predictions.csv",index=False)
    identity={"model":"tabpfn3","train_sha256":"source"}
    write_json(tmp_path/"state.json",{"status":"running"})
    assert completed_fold(tmp_path,expected,identity) is None
    write_json(tmp_path/"state.json",{"status":"complete","identity":identity,
              "artifacts":[file_record(tmp_path/"predictions.csv")]})
    pd.testing.assert_frame_equal(completed_fold(tmp_path,expected,identity),predictions)
    with pytest.raises(ValueError): completed_fold(tmp_path,expected,{"model":"tabpfn3"})
    (tmp_path/"predictions.csv").write_text("corrupted")
    with pytest.raises(ValueError): completed_fold(tmp_path,expected,identity)


def test_foundation_recovers_after_failure_without_repeating_completed_fold(tmp_path,monkeypatch):
    import datafest.foundation as foundation
    data=tmp_path/"data"; data.mkdir()
    frame=pd.DataFrame([{ "id_cliente":i,"mes":m,"dias_ultima_interaccion":i*10,"objetivo":i-1}
                        for m in (202601,202608,202609,202610,202611) for i in (1,2)])
    frame.to_csv(data/"train.csv",index=False)
    fits=[]; calls=[]
    class Fake:
        classes_=np.array([0,1])
        def get_params(self,deep=False): return {"model_path":"tabpfn-v3.ckpt"}
        def fit(self,x,y): fits.append(len(x)); return self
        def predict_proba(self,x):
            p=np.where(x.dias_ultima_interaccion.eq(20),.8,.2)
            return np.column_stack([1-p,p])
    def make(model,device):
        calls.append(model)
        if len(calls)==2: raise RuntimeError("simulated fold interruption")
        return Fake(),{"version":"TabPFN-3"}
    def checkpoint(estimator,model,directory):
        path=directory/"checkpoint.txt"; path.write_text("weights")
        return [file_record(path)]
    monkeypatch.setattr(foundation,"_make_estimator",make)
    monkeypatch.setattr(foundation,"_checkpoint_records",checkpoint)
    monkeypatch.setattr(foundation,"save_pickle",lambda value,path:path.write_text("estimator"))
    monkeypatch.setattr(foundation.importlib.metadata,"version",lambda name:"test-version")
    root=tmp_path/"foundation"
    with pytest.raises(RuntimeError): foundation.run_foundation("tabpfn3",data,root,"cpu")
    assert fits==[4]
    result=foundation.run_foundation("tabpfn3",data,root,"cpu")
    assert fits==[4,6,8]
    assert len(calls)==4 # first complete fold was reused
    assert result["mean_rolling_gini"]==1.
    assert len(pd.read_csv(root/"tabpfn3/rolling_predictions.csv"))==6
