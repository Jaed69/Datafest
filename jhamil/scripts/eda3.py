import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score as auc
d=sys.argv[1]
tr=pd.read_csv(f"{d}/train.csv"); te=pd.read_csv(f"{d}/test.csv")
s=tr.sort_values(["id_cliente","mes"]).copy()
s["k"]=s.groupby("id_cliente").cumcount()+1
s["same"]=s.dias_ultima_interaccion==s.dias_ultima_transaccion
print("eq by month:", s.groupby("mes").same.mean().round(3).to_dict())
print("target rate by eq:", s.groupby("same").objetivo.agg(["size","mean"]).round(3).to_dict())
# is eq persistent? first month eq, then changes?
s["same_prev"]=s.groupby("id_cliente").same.shift()
print("transition eq_prev->eq:\n",pd.crosstab(s.same_prev,s.same))
# when not eq, dui distribution and relation with target
ne=s[~s.same]
print("non-eq dui describe", ne.dias_ultima_interaccion.describe().round(1).to_dict())
print("AUC dui within non-eq rows", round(2*auc(ne.objetivo,-ne.dias_ultima_interaccion)-1,3))
print("AUC dui within eq rows", round(2*auc(s[s.same].objetivo,-s[s.same].dias_ultima_interaccion)-1,3))
# test: client 2 in train
print(s[s.id_cliente.isin([2,3])][["id_cliente","mes","dias_ultima_transaccion","dias_ultima_interaccion","objetivo"]].to_string())
# test new vs returning
te["ret"]=te.id_cliente.isin(tr.id_cliente)
print("test eq by returning:", te.assign(eq=te.dias_ultima_interaccion==te.dias_ultima_transaccion).groupby("ret").eq.mean().to_dict())
# Nov value vs Dec value for returning
nov=s[s.mes==202611].set_index("id_cliente").dias_ultima_interaccion
m=te[te.ret].set_index("id_cliente").dias_ultima_interaccion
j=pd.concat([nov.rename("nov"),m.rename("dec")],axis=1).dropna()
print("Dec==Nov share", (j.nov==j.dec).mean(), "corr", j.corr().iloc[0,1].round(3))
# within train month m -> m+1 same share
print("train consecutive same share", (s.dias_ultima_interaccion==s.groupby("id_cliente").dias_ultima_interaccion.shift())[s.k>1].mean())
# id ordering vs entry month
print("id ranges by entry month:", s.groupby("id_cliente").mes.min().reset_index().groupby("mes").id_cliente.agg(["min","max"]).to_dict("index"))
print("test new ids range", te[~te.ret].id_cliente.agg(["min","max"]).to_dict())
