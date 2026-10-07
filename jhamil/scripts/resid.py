import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from rulelib import rules
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"])
tr["k"]=tr.groupby("id_cliente").cumcount()+1
gini=lambda y,p:2*auc(y,p)-1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
for c in cats: tr[c]=tr[c].astype("category")
base=[c for c in tr.columns if c not in("id_cliente","mes","objetivo","dias_ultima_interaccion")]
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,n_estimators=400,verbose=-1)
def fit_rules(a): return LogisticRegression(max_iter=2000).fit(rules(a),a.objetivo)
res=[]
for m in [202606,202607,202608,202609,202610,202611]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]; lr=fit_rules(a)
    za=lr.decision_function(rules(a)); zv=lr.decision_function(rules(v))
    for n_est in [100,400]:
        p=np.mean([lgb.LGBMClassifier(**{**P,"n_estimators":n_est},random_state=s).fit(a[base],a.objetivo,init_score=za).predict_proba(v[base],raw_score=True)+zv for s in range(3)],0)
        res.append((m%100,f"RES{n_est}",gini(v.objetivo,p)))
    res.append((m%100,"RULES",gini(v.objetivo,zv)))
r=pd.DataFrame(res,columns=["mes","model","g"]).pivot(index="mes",columns="model",values="g")
print(r.round(4)); print("MEAN",r.mean().round(4).to_dict())
# residual importance on full Jan-Nov
lr=fit_rules(tr); z=lr.decision_function(rules(tr))
print("rule coefs:",dict(zip(rules(tr).columns,lr.coef_[0].round(3))))
imp=np.zeros(len(base))
for s in range(5):
    mdl=lgb.LGBMClassifier(**P,random_state=s,importance_type="gain").fit(tr[base],tr.objetivo,init_score=z); imp+=mdl.feature_importances_
print("residual gain importance:\n",pd.Series(imp/imp.sum(),index=base).sort_values(ascending=False).round(3).head(12))
# residual per band x feature: observed - expected by bins
tr["pexp"]=1/(1+np.exp(-z)); tr["res"]=tr.objetivo-tr.pexp
