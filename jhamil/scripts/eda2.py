import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score as auc
d=sys.argv[1]
tr=pd.read_csv(f"{d}/train.csv"); te=pd.read_csv(f"{d}/test.csv")
s=tr.sort_values(["id_cliente","mes"]).copy()
s["k"]=s.groupby("id_cliente").cumcount()+1  # months in panel so far
s["is_last"]=s.groupby("id_cliente").mes.transform("max")==s.mes
last=s[s.is_last]
# dias_ultima_interaccion dynamics
s["dui_prev"]=s.groupby("id_cliente").dias_ultima_interaccion.shift()
s["dui_diff"]=s.dias_ultima_interaccion-s.dui_prev
print(s.groupby("k").objetivo.mean().round(3).to_dict())
# univariate AUC
num=[c for c in te.columns if c not in("id_cliente","mes") and not pd.api.types.is_string_dtype(tr[c])]
r={c:auc(tr.objetivo,tr[c].astype(float))-0.5 for c in num}
r["k_months_in_panel"]=auc(s.objetivo,s.k)-0.5
r["dui_diff(k>1)"]=auc(s.objetivo[s.k>1],s.dui_diff[s.k>1])-0.5
print("univariate Gini (2*(AUC-.5)):\n",(pd.Series(r)*2).sort_values(key=abs,ascending=False).round(3))
for c in [c for c in te.columns if pd.api.types.is_string_dtype(tr[c])]:
    print(tr.groupby(c).objetivo.agg(["size","mean"]).round(3).T)
print("test dui describe\n",te.dias_ultima_interaccion.describe())
print("train dui==dias_ultima_transaccion share", (tr.dias_ultima_interaccion==tr.dias_ultima_transaccion).mean(), "test", (te.dias_ultima_interaccion==te.dias_ultima_transaccion).mean())
print("by k==1 that equality share", s.groupby("k").apply(lambda g:(g.dias_ultima_interaccion==g.dias_ultima_transaccion).mean()).round(3).to_dict())
