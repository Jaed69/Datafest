import pandas as pd, numpy as np, sys, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from rulelib import rules2
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"]); tr["k"]=tr.groupby("id_cliente").cumcount()+1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
for c in cats: tr[c]=tr[c].astype("category")
base=[c for c in tr.columns if c not in("id_cliente","mes","objetivo","dias_ultima_interaccion")]
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,verbose=-1)
def cat_fit(X,y,Xv,seed):
    Xc=X.copy(); Xvc=Xv.copy()
    for c in cats: Xc[c]=Xc[c].astype(str); Xvc[c]=Xvc[c].astype(str)
    return CatBoostClassifier(iterations=800,learning_rate=0.05,depth=6,l2_leaf_reg=5,random_seed=seed,verbose=0,thread_count=-1,cat_features=cats).fit(Xc,y).predict_proba(Xvc)[:,1]
def xgb_fit(X,y,Xv,seed):
    return xgb.XGBClassifier(n_estimators=500,learning_rate=0.03,max_depth=4,min_child_weight=50,subsample=0.8,colsample_bytree=0.8,reg_lambda=5,tree_method="hist",enable_categorical=True,random_state=seed).fit(X,y).predict_proba(Xv)[:,1]
out=[]
for m in [202609,202610,202611]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]; y=a.objetivo
    Ra,Rv=rules2(a),rules2(v)
    XaR=pd.concat([a[base],Ra.add_prefix("r_")],axis=1); XvR=pd.concat([v[base],Rv.add_prefix("r_")],axis=1)
    pr={}
    pr["LGB"]=np.mean([lgb.LGBMClassifier(**P,n_estimators=400,random_state=s).fit(a[base],y).predict_proba(v[base])[:,1] for s in range(3)],0)
    pr["LGB_R"]=np.mean([lgb.LGBMClassifier(**P,n_estimators=400,random_state=s).fit(XaR,y).predict_proba(XvR)[:,1] for s in range(3)],0)
    pr["CAT"]=np.mean([cat_fit(a[base],y,v[base],s) for s in range(2)],0)
    pr["CAT_R"]=np.mean([cat_fit(XaR,y,XvR,s) for s in range(2)],0)
    pr["XGB"]=np.mean([xgb_fit(a[base],y,v[base],s) for s in range(3)],0)
    pr["XGB_R"]=np.mean([xgb_fit(XaR,y,XvR,s) for s in range(3)],0)
    lr=LogisticRegression(max_iter=2000).fit(Ra,y); za,zv=lr.decision_function(Ra),lr.decision_function(Rv)
    pr["RULES2"]=zv
    pr["RES100"]=np.mean([lgb.LGBMClassifier(**P,n_estimators=100,random_state=s).fit(a[base],y,init_score=za).predict_proba(v[base],raw_score=True) for s in range(3)],0)+zv
    out.append(pd.DataFrame({"id_cliente":v.id_cliente.values,"mes":m,"objetivo":v.objetivo.values,**pr}))
    print(m, "done", flush=True)
pd.concat(out).to_csv(sys.argv[2],index=False)
