import pandas as pd, numpy as np, sys
from pytabkit import RealMLP_TD_Classifier
from rulelib import rules2
tr=pd.read_csv(f"{sys.argv[1]}/train.csv").sort_values(["id_cliente","mes"]); tr["k"]=tr.groupby("id_cliente").cumcount()+1
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal"]
bools=["tiene_tarjeta_credito","activo_movil","es_nuevo_cliente","tiene_prestamo","tiene_seguro"]
levels={c:sorted(tr[c].astype(str).unique()) for c in cats}
base=[c for c in tr.columns if c not in("id_cliente","mes","objetivo","dias_ultima_interaccion")]
def prep(d,with_rules):
    o=pd.DataFrame(index=d.index)
    for c in base:
        if c in cats: o[c]=pd.Categorical(d[c].astype(str),categories=levels[c])
        elif c in bools: o[c]=d[c].astype("int8")
        else: o[c]=d[c].astype("float32")
    if with_rules: o=pd.concat([o,rules2(d).add_prefix("r_").astype("float32")],axis=1)
    return o
out=[]
for m in [202609,202610,202611]:
    a=tr[tr.mes<m]; v=tr[tr.mes==m]; pr={}
    for name,wr in [("N1",False),("N1_R",True)]:
        Xa,Xv=prep(a,wr),prep(v,wr)
        pr[name]=np.mean([RealMLP_TD_Classifier(device="cuda",random_state=s,val_metric_name="1-auc_ovr",verbosity=0).fit(Xa,a.objetivo.values,cat_col_names=cats).predict_proba(Xv)[:,1] for s in (42,43,44)],0)
        print(m,name,"done",flush=True)
    out.append(pd.DataFrame({"id_cliente":v.id_cliente.values,"mes":m,**pr}))
pd.concat(out).to_csv(sys.argv[2],index=False)
