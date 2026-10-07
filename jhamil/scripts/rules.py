import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"])
tr["k"]=tr.groupby("id_cliente").cumcount()+1
gini=lambda y,p:2*auc(y,p)-1
def rules(d):
    L=(d.banda_riesgo=="low"); H=(d.banda_riesgo=="high"); M=(d.banda_riesgo=="medium")
    p=d.numero_productos; dtx=d.dias_ultima_transaccion/365
    mc=d.activo_movil&d.tiene_tarjeta_credito
    return pd.DataFrame({
      "L":L,"H":H,
      "L_p3":L&(p>=3),"L_mc_p2":L&mc&(p>=2),"L_dtx":L*dtx,
      "H_dtx100":H&(d.dias_ultima_transaccion<100),"H_dtx180":H&(d.dias_ultima_transaccion<180),
      "H_old_branch":H&(d.dias_ultima_transaccion>=180)&(d.canal_adquisicion=="branch"),
      "M_dtx":M*dtx,"k":np.log(d.k)}).astype(float)
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
for c in cats: tr[c]=tr[c].astype("category")
base=[c for c in tr.columns if c not in("id_cliente","mes","objetivo","dias_ultima_interaccion")]
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,n_estimators=400,verbose=-1)
out=[]
for m in [202606,202607,202608,202609,202610,202611]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]
    Ra,Rv=rules(a),rules(v)
    pr=LogisticRegression(C=1.0,max_iter=2000).fit(Ra,a.objetivo).predict_proba(Rv)[:,1]
    pl=np.mean([lgb.LGBMClassifier(**P,random_state=s).fit(a[base],a.objetivo).predict_proba(v[base])[:,1] for s in range(3)],0)
    Xa=pd.concat([a[base],Ra.add_prefix("r_")],axis=1); Xv=pd.concat([v[base],Rv.add_prefix("r_")],axis=1)
    plr=np.mean([lgb.LGBMClassifier(**P,random_state=s).fit(Xa,a.objetivo).predict_proba(Xv)[:,1] for s in range(3)],0)
    out.append(pd.DataFrame({"mes":m,"id":v.id_cliente.values,"y":v.objetivo.values,"RULES":pr,"LGB":pl,"LGB_RULES":plr}))
    print(m%100, {k:round(gini(v.objetivo,x),4) for k,x in [("RULES",pr),("LGB",pl),("LGB_RULES",plr)]},flush=True)
o=pd.concat(out); o.to_csv(f"{sys.argv[2]}",index=False)
per=o.groupby("mes").apply(lambda g:pd.Series({k:gini(g.y,g[k]) for k in ["RULES","LGB","LGB_RULES"]}))
print("MEAN", per.mean().round(4).to_dict(), "STD", per.std().round(4).to_dict())
# paired client bootstrap of mean-of-monthly-gini difference
rng=np.random.default_rng(0); ids=o.id.unique(); B=500
grp={i:idx for i,idx in o.groupby("id").indices.items()}
def stat(df): return df.groupby("mes").apply(lambda g:pd.Series({k:gini(g.y,g[k]) for k in ["RULES","LGB","LGB_RULES"]})).mean()
diffs=[]
for b in range(B):
    s=rng.choice(ids,len(ids)); idx=np.concatenate([grp[i] for i in s]); st=stat(o.iloc[idx])
    diffs.append([st.RULES-st.LGB, st.LGB_RULES-st.LGB])
d=np.array(diffs)
for n,col in [("RULES-LGB",0),("LGB_RULES-LGB",1)]:
    print(n,"mean",d[:,col].mean().round(4),"95%CI",np.percentile(d[:,col],[2.5,97.5]).round(4),"P(>0)",(d[:,col]>0).mean().round(3))
