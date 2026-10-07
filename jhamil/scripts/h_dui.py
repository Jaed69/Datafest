import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from scipy import stats
d=sys.argv[1]
tr=pd.read_csv(f"{d}/train.csv"); te=pd.read_csv(f"{d}/test.csv")
gini=lambda y,p: 2*auc(y,p)-1
# --- 1. mechanism checks
tr["iscopy"]=tr.dias_ultima_interaccion==tr.dias_ultima_transaccion
nc=tr[~tr.iscopy].dias_ultima_interaccion
print("non-copy dui uniformity KS p:", stats.kstest((nc-0.5)/364,"uniform").pvalue.round(4),
      "| test dui KS p:", stats.kstest((te.dias_ultima_interaccion-0.5)/364,"uniform").pvalue.round(4))
print("copy-share by month vs (12-m)/11:", {m:(round(g.iscopy.mean(),3), round((12-(m%100))/11,3)) for m,g in tr.groupby("mes")})
print("copy flag vs target per month (rate copy / noncopy):", {m%100:(round(g[g.iscopy].objetivo.mean(),3) if g.iscopy.any() else None, round(g[~g.iscopy].objetivo.mean(),3) if (~g.iscopy).any() else None) for m,g in tr.groupby("mes")})
# --- 2. rolling
tr=tr.sort_values(["id_cliente","mes"]); tr["k"]=tr.groupby("id_cliente").cumcount()+1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
for c in cats: tr[c]=tr[c].astype("category")
base=[c for c in te.columns if c not in("id_cliente","mes","dias_ultima_interaccion")]+["k"]
def rank_m(x,m): return x.groupby(m).rank(pct=True)
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,n_estimators=400,verbose=-1)
rng=np.random.default_rng(0)
rows=[]
for m in [202606,202607,202608,202609,202610,202611]:
    a=tr[tr.mes<m].copy(); v=tr[tr.mes==m].copy()
    noise=[rng.integers(1,365,len(v)) for _ in range(5)]
    for var in ["RAW","RANK","DROP"]:
        X=a[base].copy(); Xv=v[base].copy()
        if var=="RAW": X["dui"]=a.dias_ultima_interaccion
        if var=="RANK": X["dui"]=rank_m(a.dias_ultima_interaccion,a.mes)
        sc=[]
        for seed in range(3):
            mdl=lgb.LGBMClassifier(**P,random_state=seed).fit(X,a.objetivo)
            def pred(dv):
                Z=Xv.copy()
                if var=="RAW": Z["dui"]=dv
                if var=="RANK": Z["dui"]=pd.Series(dv,index=v.index).rank(pct=True)
                return mdl.predict_proba(Z)[:,1]
            p_real=pred(v.dias_ultima_interaccion.values)
            g_all=gini(v.objetivo,p_real)
            g_nc=gini(v.objetivo[~v.iscopy],p_real[~v.iscopy.values])
            g_dec=np.mean([gini(v.objetivo,pred(n)) for n in noise])
            sc.append((g_all,g_nc,g_dec))
        s=np.mean(sc,0); rows.append(dict(mes=m%100,var=var,all=s[0],noncopy=s[1],dec_sim=s[2]))
        print(rows[-1],flush=True)
r=pd.DataFrame(rows)
print(r.pivot(index="mes",columns="var",values=["all","noncopy","dec_sim"]).round(4))
print("MEAN\n",r.groupby("var")[["all","noncopy","dec_sim"]].mean().round(4))
r.to_csv(sys.argv[2],index=False)
