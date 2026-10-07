import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score as auc
gini=lambda y,p:2*auc(y,p)-1
prev=pd.read_csv(sys.argv[1])  # stored repo OOF (B0,H1b,N1,...)
bat=pd.read_csv(sys.argv[2]); import os; HAVE_NN=os.path.exists(sys.argv[3]); nn=pd.read_csv(sys.argv[3]) if HAVE_NN else bat[["id_cliente","mes"]]
o=bat.merge(nn,on=["id_cliente","mes"]).merge(prev[["id_cliente","mes","objetivo","B0","H1b","N1"]].rename(columns={"N1":"N1_repo","objetivo":"y_repo"}),on=["id_cliente","mes"])
assert (o.objetivo==o.y_repo).all() and len(o)==29800
R=lambda c:o.groupby("mes")[c].rank(pct=True)
o["CURRENT"]=(R("B0")+R("H1b")+R("N1_repo"))/3
o["B1_LGB+RES"]=(R("LGB")+R("RES100"))/2
o["B2_CURRENT+RES"]=(o.groupby("mes").CURRENT.rank(pct=True)+R("RES100"))/2
if HAVE_NN: o["B3_LGBR+CATR+N1R"]=(R("LGB_R")+R("CAT_R")+R("N1_R"))/3
if HAVE_NN: o["B4_RES+CATR+N1R"]=(R("RES100")+R("CAT_R")+R("N1_R"))/3
models=[c for c in o.columns if c not in("id_cliente","mes","objetivo","y_repo","B0","H1b","N1_repo")]
months=sorted(o.mes.unique()); y={m:o.objetivo[o.mes==m].values for m in months}
X={m:o.loc[o.mes==m,models].values for m in months}; ids={m:o.id_cliente[o.mes==m].values for m in months}
def score(sel=None):
    g=[]
    for m in months:
        idx=np.arange(len(y[m])) if sel is None else sel[m]
        yy=y[m][idx]; g.append([gini(yy,X[m][idx,j]) for j in range(len(models))])
    return np.array(g)  # months x models
obs=score(); mean=obs.mean(0)
print("reproduction check CURRENT mean:",round(mean[models.index("CURRENT")],5),"(stored 0.25432)")
# paired client bootstrap, one client draw shared across months/models
allids=o.id_cliente.unique(); rng=np.random.default_rng(2026); B=int(sys.argv[4])
pos={m:pd.Series(np.arange(len(ids[m])),index=ids[m]) for m in months}
boots=np.zeros((B,len(models)))
for b in range(B):
    draw=pd.Series(rng.choice(allids,len(allids)))
    sel={m:pos[m].reindex(draw).dropna().astype(int).values for m in months}
    boots[b]=score(sel).mean(0)
ref=models.index("CURRENT"); d=boots-boots[:,[ref]]; dobs=mean-mean[ref]
p=np.minimum(1,(np.sum(np.abs(d-d.mean(0))>=np.abs(dobs),0)+1)/(B+1))
tab=pd.DataFrame({"Sep":obs[0],"Oct":obs[1],"Nov":obs[2],"Mean":mean,"Std":obs.std(0,ddof=1),"dMean":dobs,"lo":np.percentile(d,2.5,0),"hi":np.percentile(d,97.5,0),"p":p},index=models).drop("CURRENT")
order=np.argsort(tab.p.values); m_=len(tab); holm=np.empty(m_); run=0
for r,i in enumerate(order): run=max(run,min(1,(m_-r)*tab.p.values[i])); holm[i]=run
tab["p_holm"]=holm
best=boots.argmax(1); tab["P(best)"]=[(best==models.index(k)).mean() for k in tab.index]
print("CURRENT", obs[:,ref].round(5), "mean", round(mean[ref],5))
print(tab.sort_values("Mean",ascending=False).round(4).to_string())
