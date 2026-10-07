import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from rulelib import rules2
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"]); tr["k"]=tr.groupby("id_cliente").cumcount()+1
gini=lambda y,p:2*auc(y,p)-1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
for c in cats: tr[c]=tr[c].astype("category")
base=[c for c in tr.columns if c not in("id_cliente","mes","objetivo","dias_ultima_interaccion")]
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,n_estimators=400,verbose=-1)
rng=np.random.default_rng(0)
print("== learning curve (client subsample of training data) ==")
for m in [202609,202610,202611]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]; ids=a.id_cliente.unique(); out={}
    for frac in [0.25,0.5,0.75,1.0]:
        g=[]
        for rep in range(3 if frac<1 else 1):
            sel=set(rng.choice(ids,int(len(ids)*frac),replace=False)); aa=a[a.id_cliente.isin(sel)]
            g.append(np.mean([gini(v.objetivo,lgb.LGBMClassifier(**P,random_state=s).fit(aa[base],aa.objetivo).predict_proba(v[base])[:,1]) for s in range(2)]))
        out[frac]=round(np.mean(g),4)
        out[f"rows_{frac}"]=len(aa)
    print(m%100,out,flush=True)
print("\n== oracle test: if model probabilities were the truth ==")
for m in [202609,202610,202611]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]
    lr=LogisticRegression(max_iter=2000).fit(rules2(a),a.objetivo); pr=lr.predict_proba(rules2(v))[:,1]
    pl=np.mean([lgb.LGBMClassifier(**P,random_state=s).fit(a[base],a.objetivo).predict_proba(v[base])[:,1] for s in range(3)],0)
    for n,p in [("RULES2",pr),("LGB",pl)]:
        sim=[gini(rng.binomial(1,p),p) for _ in range(200)]
        print(m%100,n,"real",round(gini(v.objetivo,p),4),"| expected if truth",round(np.mean(sim),4),"+-",round(np.std(sim),4),"| mean pred",round(p.mean(),3),"real rate",round(v.objetivo.mean(),3))
