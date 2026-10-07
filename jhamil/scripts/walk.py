import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from rulelib import rules2
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"]); tr["k"]=tr.groupby("id_cliente").cumcount()+1
gini=lambda y,p:2*auc(y,p)-1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
for c in cats: tr[c]=tr[c].astype("category")
base=[c for c in tr.columns if c not in("id_cliente","mes","objetivo","dias_ultima_interaccion")]
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,n_estimators=400,verbose=-1)
fit=lambda a,v,cols=base:np.mean([lgb.LGBMClassifier(**P,random_state=s).fit(a[cols],a.objetivo).predict_proba(v[cols])[:,1] for s in range(2)],0)
rows=[]
months=sorted(tr.mes.unique())
for m in months[1:]:
    v=tr[tr.mes==m]; exp=tr[tr.mes<m]; sl=tr[(tr.mes<m)&(tr.mes>=months[max(0,months.index(m)-3)])]
    lr=LogisticRegression(max_iter=2000).fit(rules2(exp),exp.objetivo)
    r=dict(mes=m,n_train=len(exp),LGB_acumulada=gini(v.objetivo,fit(exp,v)),LGB_movil3=gini(v.objetivo,fit(sl,v)),Reglas_acumulada=gini(v.objetivo,lr.decision_function(rules2(v))))
    rows.append(r); print({k:(round(x,4) if isinstance(x,float) else x) for k,x in r.items()},flush=True)
pd.DataFrame(rows).to_csv("walk_forward.csv",index=False)
# random KFold vs GroupKFold vs temporal
X=tr; res={}
for name,cv in [("KFold aleatorio",KFold(5,shuffle=True,random_state=0)),("GroupKFold por cliente",GroupKFold(5))]:
    g=[]
    for a_i,v_i in cv.split(X,groups=X.id_cliente):
        a,v=X.iloc[a_i],X.iloc[v_i]; g.append(gini(v.objetivo,fit(a,v)))
    res[name]=(np.mean(g),np.std(g)); print(name,np.round(g,4),flush=True)
# random KFold with id_cliente as feature (memorization)
g=[]
for a_i,v_i in KFold(5,shuffle=True,random_state=0).split(X):
    a,v=X.iloc[a_i],X.iloc[v_i]; g.append(gini(v.objetivo,fit(a,v,base+["id_cliente"])))
res["KFold aleatorio + id_cliente"]=(np.mean(g),np.std(g)); print("KFold+id",np.round(g,4))
w=pd.DataFrame(rows); t=w[w.mes>=202609].LGB_acumulada
res["Temporal sep-nov"]=(t.mean(),t.std())
pd.DataFrame(res,index=["mean","std"]).T.to_csv("cv_compare.csv"); print(pd.DataFrame(res,index=["mean","std"]).T.round(4))
