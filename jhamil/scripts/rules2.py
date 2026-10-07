import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from rulelib import rules
def rules2(d):
    r=rules(d); LM=d.banda_riesgo.astype(str).isin(["low","medium"])
    r["LM_debt60"]=(LM&(d.ratio_deuda_ingresos>=0.6)).astype(float)
    r["saldo_floor"]=(d.saldo_promedio<=300).astype(float); return r
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"]); tr["k"]=tr.groupby("id_cliente").cumcount()+1
gini=lambda y,p:2*auc(y,p)-1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
for c in cats: tr[c]=tr[c].astype("category")
base=[c for c in tr.columns if c not in("id_cliente","mes","objetivo","dias_ultima_interaccion")]
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,verbose=-1)
prev=pd.read_csv("bayes_oof.csv")[["mes","id","LGB","RES100","RULES"]]
oof=[]
for m in [202606,202607,202608,202609,202610,202611]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]
    lr=LogisticRegression(max_iter=2000).fit(rules2(a),a.objetivo); za,zv=lr.decision_function(rules2(a)),lr.decision_function(rules2(v))
    res=np.mean([lgb.LGBMClassifier(**P,n_estimators=100,random_state=s).fit(a[base],a.objetivo,init_score=za).predict_proba(v[base],raw_score=True) for s in range(3)],0)+zv
    oof.append(pd.DataFrame({"mes":m,"id":v.id_cliente.values,"y":v.objetivo.values,"RULES2":pd.Series(zv).rank(pct=True).values,"RES100_R2":pd.Series(res).rank(pct=True).values}))
o=pd.concat(oof).merge(prev,on=["mes","id"])
o["BLEND_LGB_RES_R2"]=(o.LGB+o.RES100_R2)/2; o["BLEND_LGB_RES"]=(o.LGB+o.RES100)/2
models=[c for c in o.columns if c not in("mes","id","y")]
st=lambda df:df.groupby("mes").apply(lambda g:pd.Series({k:gini(g.y,g[k]) for k in models}))
per=st(o); print(per.round(4)); print("MEAN\n",per.mean().round(4).sort_values(ascending=False))
rng=np.random.default_rng(0); ids=o.id.unique(); grp=o.groupby("id").indices; D=[]
for b in range(300):
    s=st(o.iloc[np.concatenate([grp[i] for i in rng.choice(ids,len(ids))])]).mean(); D.append(s-s.LGB)
D=pd.DataFrame(D); print(pd.DataFrame({"mean":D.mean(),"lo":D.quantile(.025),"hi":D.quantile(.975),"P>0":(D>0).mean()}).drop("LGB").round(4))
