import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from rulelib import rules
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"])
tr["k"]=tr.groupby("id_cliente").cumcount()+1
gini=lambda y,p:2*auc(y,p)-1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
bools=["tiene_tarjeta_credito","activo_movil","es_nuevo_cliente","tiene_prestamo","tiene_seguro"]
nums=["edad","ingresos","ratio_deuda_ingresos","antiguedad_cuenta_meses","numero_productos","saldo_promedio","dias_ultima_transaccion","antiguedad_direccion_meses","visitas_web_ultimos_90_dias","distancia_sucursal_km","dia_preferido_pago","k"]
for c in cats: tr[c]=tr[c].astype("category")
base=cats+bools+nums
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,verbose=-1)
def binned(a,v,nb=6):
    kb=KBinsDiscretizer(nb,encode="ordinal",strategy="quantile").fit(a[nums])
    f=lambda d:pd.concat([pd.DataFrame(kb.transform(d[nums]),columns=nums,index=d.index).astype(int).astype(str),d[cats+bools].astype(str)],axis=1)
    return f(a),f(v)
def eb_cells(a,v):
    # empirical Bayes: Beta prior from band rate, strength m tuned as 200 pseudo-obs
    key=lambda d:(d.banda_riesgo.astype(str)+"|"+d.numero_productos.clip(upper=3).astype(str)+"|"+(d.activo_movil&d.tiene_tarjeta_credito).astype(str)+"|"+pd.cut(d.dias_ultima_transaccion,[0,100,180,400]).astype(str))
    ka,kv=key(a),key(v); band_rate=a.groupby("banda_riesgo",observed=True).objetivo.mean()
    st=a.groupby(ka).objetivo.agg(["sum","count"]); prior=a.groupby(ka).banda_riesgo.first().astype(str).map(band_rate.rename(index=str))
    m=200; post=(st["sum"]+m*prior)/(st["count"]+m)
    return kv.map(post).fillna(v.banda_riesgo.astype(str).map(band_rate.rename(index=str))).values
rows=[]; oof=[]; perm=[]
for m in [202606,202607,202608,202609,202610,202611]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]; y=v.objetivo.values; pr={}
    pr["LGB"]=np.mean([lgb.LGBMClassifier(**P,n_estimators=400,random_state=s).fit(a[base],a.objetivo).predict_proba(v[base])[:,1] for s in range(3)],0)
    lr=LogisticRegression(max_iter=2000).fit(rules(a),a.objetivo); za,zv=lr.decision_function(rules(a)),lr.decision_function(rules(v))
    pr["RULES"]=zv
    res=[lgb.LGBMClassifier(**P,n_estimators=100,random_state=s).fit(a[base],a.objetivo,init_score=za) for s in range(3)]
    rp=lambda X:np.mean([r.predict_proba(X,raw_score=True) for r in res],0)+zv
    pr["RES100"]=rp(v[base])
    # permutation importance of residual model (only residual part changes)
    g0=gini(y,pr["RES100"]); rng=np.random.default_rng(m)
    for f in base:
        X=v[base].copy(); X[f]=pd.Series(rng.permutation(X[f].values),index=X.index).astype(X[f].dtype); perm.append((m%100,f,g0-gini(y,rp(X))))
    # Naive Bayes
    ba,bv=binned(a,v); enc=OneHotEncoder(handle_unknown="ignore")
    from sklearn.preprocessing import OrdinalEncoder
    oe=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1).fit(ba)
    Ea,Ev=oe.transform(ba).astype(int),np.clip(oe.transform(bv).astype(int),0,None)
    pr["NB_global"]=CategoricalNB(alpha=1,min_categories=Ea.max(0)+1).fit(Ea,a.objetivo).predict_proba(Ev)[:,1]
    nbb=np.zeros(len(v))
    for b in ["low","medium","high"]:
        ia=(a.banda_riesgo==b).values; iv=(v.banda_riesgo==b).values
        nbm=CategoricalNB(alpha=1,min_categories=Ea.max(0)+1).fit(Ea[ia],a.objetivo[ia]); nbb[iv]=nbm.predict_proba(Ev[iv])[:,1]
    pr["NB_by_band"]=nbb
    # Bayesian (MAP, Gaussian prior) logistic: band x all binned features
    Xa=pd.get_dummies(ba.drop(columns="banda_riesgo").apply(lambda s:ba.banda_riesgo+"_"+s.name+"_"+s)).astype(np.float32)
    Xv=pd.get_dummies(bv.drop(columns="banda_riesgo").apply(lambda s:bv.banda_riesgo+"_"+s.name+"_"+s)).reindex(columns=Xa.columns,fill_value=0).astype(np.float32)
    for C in [0.01,0.05]:
        pr[f"BAYES_LR_C{C}"]=LogisticRegression(C=C,max_iter=3000).fit(Xa,a.objetivo).decision_function(Xv)
    pr["EB_CELLS"]=eb_cells(a,v)
    r={k:gini(y,p) for k,p in pr.items()}; r["mes"]=m%100; rows.append(r); print({k:round(x,4) for k,x in r.items()},flush=True)
    oof.append(pd.DataFrame({"mes":m,"id":v.id_cliente.values,"y":y,**{k:pd.Series(p).rank(pct=True).values for k,p in pr.items()}}))
R=pd.DataFrame(rows).set_index("mes"); print(R.round(4)); print("MEAN\n",R.mean().round(4).sort_values(ascending=False))
pm=pd.DataFrame(perm,columns=["mes","f","drop"]).groupby("f")["drop"].agg(["mean","std"]); print("residual permutation importance:\n",pm.sort_values("mean",ascending=False).round(4).head(10))
o=pd.concat(oof); o["BLEND_LGB_BAYES"]=(o.LGB+o["BAYES_LR_C0.01"])/2; o["BLEND_LGB_RES"]=(o.LGB+o.RES100)/2
o.to_csv(sys.argv[2],index=False)
models=[c for c in o.columns if c not in("mes","id","y")]
st=lambda df:df.groupby("mes").apply(lambda g:pd.Series({k:gini(g.y,g[k]) for k in models})).mean()
print("MEAN incl blends:\n",st(o).round(4).sort_values(ascending=False))
rng=np.random.default_rng(0); ids=o.id.unique(); grp=o.groupby("id").indices; D=[]
for b in range(300):
    idx=np.concatenate([grp[i] for i in rng.choice(ids,len(ids))]); s=st(o.iloc[idx]); D.append(s-s.LGB)
D=pd.DataFrame(D)
print("paired bootstrap vs LGB:\n",pd.DataFrame({"mean":D.mean(),"lo":D.quantile(.025),"hi":D.quantile(.975),"P>0":(D>0).mean()}).drop("LGB").round(4).sort_values("mean",ascending=False))
