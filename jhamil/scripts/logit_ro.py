import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from rulelib import rules2
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"]); tr["k"]=tr.groupby("id_cliente").cumcount()+1
gini=lambda y,p:2*auc(y,p)-1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
bools=["tiene_tarjeta_credito","activo_movil","es_nuevo_cliente","tiene_prestamo","tiene_seguro"]
nums=["edad","ingresos","ratio_deuda_ingresos","antiguedad_cuenta_meses","numero_productos","saldo_promedio","dias_ultima_transaccion","antiguedad_direccion_meses","visitas_web_ultimos_90_dias","distancia_sucursal_km","dia_preferido_pago","k"]
def onehot(d,cols,ref_cols=None):
    X=pd.get_dummies(d[cols].astype(str),drop_first=False).astype(np.float32)
    return X if ref_cols is None else X.reindex(columns=ref_cols,fill_value=0)
def lr_plain(a,v):
    sc=StandardScaler().fit(a[nums]); A=pd.concat([pd.DataFrame(sc.transform(a[nums]),index=a.index,columns=nums),a[bools].astype(float),onehot(a,cats)],axis=1)
    V=pd.concat([pd.DataFrame(sc.transform(v[nums]),index=v.index,columns=nums),v[bools].astype(float),onehot(v,cats,onehot(a,cats).columns)],axis=1)
    return LogisticRegression(C=1.0,max_iter=3000).fit(A,a.objetivo).decision_function(V)
def binned(a,v):
    kb=KBinsDiscretizer(6,encode="ordinal",strategy="quantile").fit(a[nums])
    f=lambda d:pd.concat([pd.DataFrame(kb.transform(d[nums]).astype(int),index=d.index,columns=nums).astype(str),d[cats+bools].astype(str)],axis=1)
    return f(a),f(v)
def lr_bins(a,v):
    ba,bv=binned(a,v); A=pd.get_dummies(ba).astype(np.float32); V=pd.get_dummies(bv).reindex(columns=A.columns,fill_value=0).astype(np.float32)
    return LogisticRegression(C=0.1,max_iter=3000).fit(A,a.objetivo).decision_function(V)
def lr_band(a,v):
    ba,bv=binned(a,v)
    cross=lambda b:b.drop(columns="banda_riesgo").apply(lambda s:b.banda_riesgo+"_"+s.name+"_"+s)
    A=pd.get_dummies(cross(ba)).astype(np.float32); V=pd.get_dummies(cross(bv)).reindex(columns=A.columns,fill_value=0).astype(np.float32)
    return LogisticRegression(C=0.05,max_iter=3000).fit(A,a.objetivo).decision_function(V)
def lr_rules(a,v): return LogisticRegression(max_iter=2000).fit(rules2(a),a.objetivo).decision_function(rules2(v))
for c in cats: tr[c]=tr[c].astype("category")
base=cats+bools+nums
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,n_estimators=400,verbose=-1)
def lgbm(a,v): return np.mean([lgb.LGBMClassifier(**P,random_state=s).fit(a[base],a.objetivo).predict_proba(v[base])[:,1] for s in range(2)],0)
M={"LightGBM":lgbm,"Logística simple":lr_plain,"Logística con tramos":lr_bins,"Logística por banda":lr_band,"Logística 12 reglas":lr_rules}
rows=[]; oof=[]
for m in sorted(tr.mes.unique())[1:]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]; r={"mes":m}; o={"mes":m,"id":v.id_cliente.values,"y":v.objetivo.values}
    for n,f in M.items(): p=f(a,v); r[n]=gini(v.objetivo,p); o[n]=pd.Series(p).rank(pct=True).values
    rows.append(r); oof.append(pd.DataFrame(o)); print({k:(round(x,4) if isinstance(x,float) else x) for k,x in r.items()},flush=True)
R=pd.DataFrame(rows); R.to_csv("logit_rolling.csv",index=False)
print("\nMEAN feb-nov:\n",R.drop(columns="mes").mean().round(4)); S=R[R.mes>=202609]
print("\nMEAN sep-nov:\n",S.drop(columns="mes").mean().round(4),"\nSTD sep-nov:\n",S.drop(columns="mes").std().round(4))
o=pd.concat(oof); o=o[o.mes>=202609]; models=list(M)
st=lambda df:df.groupby("mes").apply(lambda g:pd.Series({k:gini(g.y,g[k]) for k in models})).mean()
rng=np.random.default_rng(0); ids=o.id.unique(); grp=o.groupby("id").indices; D=[]
for b in range(1000):
    s=st(o.iloc[np.concatenate([grp[i] for i in rng.choice(ids,len(ids))])]); D.append(s-s.LightGBM)
D=pd.DataFrame(D).drop(columns="LightGBM")
print("\nvs LightGBM sep-nov (1000 bootstrap):\n",pd.DataFrame({"diff":D.mean(),"lo":D.quantile(.025),"hi":D.quantile(.975)}).round(4))
