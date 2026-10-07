import pandas as pd, numpy as np, sys
d = sys.argv[1]
tr = pd.read_csv(f"{d}/train.csv"); te = pd.read_csv(f"{d}/test.csv")
print("shapes", tr.shape, te.shape, "nulls", tr.isna().sum().sum(), te.isna().sum().sum())
g = tr.groupby("mes").agg(n=("objetivo","size"), pos=("objetivo","sum"), rate=("objetivo","mean"))
print(g)
print("unique clients train", tr.id_cliente.nunique(), "test", te.id_cliente.nunique())
print("test ids in train", te.id_cliente.isin(tr.id_cliente).mean())
print("dup client-month", tr.duplicated(["id_cliente","mes"]).sum())
# rows after positive?
first_pos = tr[tr.objetivo==1].groupby("id_cliente").mes.min()
x = tr.merge(first_pos.rename("fp"), left_on="id_cliente", right_index=True)
print("rows after first positive", (x.mes>x.fp).sum(), "test ids that converted in train", te.id_cliente.isin(first_pos.index).sum())
# months per client
mpc = tr.groupby("id_cliente").mes.nunique(); print("months/client\n", mpc.value_counts().sort_index())
# contiguity
s = tr.sort_values(["id_cliente","mes"]); s["gap"]=s.groupby("id_cliente").mes.diff()
print("gaps", s.gap.value_counts().head())
# entry month
print("entry month\n", tr.groupby("id_cliente").mes.min().value_counts().sort_index())
print("new test ids (not in train)", (~te.id_cliente.isin(tr.id_cliente)).sum())
# which features vary within client
feats=[c for c in te.columns if c not in("id_cliente","mes")]
allx = pd.concat([tr,te])
var = {c:(allx.groupby("id_cliente")[c].nunique()>1).mean() for c in feats}
print("share of clients where feature varies:\n", pd.Series(var).sort_values(ascending=False))
