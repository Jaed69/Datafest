import pandas as pd, numpy as np, sys
from sklearn.tree import DecisionTreeClassifier, export_text
tr=pd.read_csv(f"{sys.argv[1]}/train.csv")
tr=tr.sort_values(["id_cliente","mes"]); tr["k"]=tr.groupby("id_cliente").cumcount()+1
bools=["tiene_tarjeta_credito","activo_movil","es_nuevo_cliente","tiene_prestamo","tiene_seguro"]
cats=["ocupacion","region","canal_adquisicion","dispositivo_principal"]
X=pd.get_dummies(tr.drop(columns=["id_cliente","mes","objetivo","dias_ultima_interaccion","banda_riesgo"]),columns=cats).astype(float)
for b in ["low","medium","high"]:
    m=tr.banda_riesgo==b
    t=DecisionTreeClassifier(max_depth=3,min_samples_leaf=1500,random_state=0).fit(X[m],tr.objetivo[m])
    print(f"\n===== {b} (n={m.sum()}, rate={tr.objetivo[m].mean():.3f}) =====")
    print(export_text(t,feature_names=list(X.columns),show_weights=False,decimals=1,class_names=None).replace("class: 0","").replace("class: 1",""))
    # leaf rates
    leaf=t.apply(X[m]); print(tr[m].groupby(leaf).objetivo.agg(["size","mean"]).round(3).T)
# fine curves
print("\nconversion by dias_ultima_transaccion (30d bins) per band")
tr["dtx_bin"]=(tr.dias_ultima_transaccion//30)*30
print(tr.pivot_table(index="dtx_bin",columns="banda_riesgo",values="objetivo").round(3).T)
print("\nconversion by numero_productos per band x activo_movil")
print(tr.pivot_table(index=["banda_riesgo","activo_movil"],columns="numero_productos",values="objetivo").round(3))
print("\nhigh band: canal x recent(dtx<90)")
h=tr[tr.banda_riesgo=="high"]
print(h.pivot_table(index="canal_adquisicion",columns=h.dias_ultima_transaccion<90,values="objetivo",aggfunc=["mean","size"]).round(3))
print("\nk (months in panel) per band:")
print(tr.pivot_table(index=np.minimum(tr.k,6),columns="banda_riesgo",values="objetivo").round(3).T)
