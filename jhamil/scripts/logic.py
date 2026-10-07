import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score as auc
tr=pd.read_csv(f"{sys.argv[1]}/train.csv")
# one row per client (features are static) with "ever converted" + months observed -> avoid overweighting long-lived clients
c=tr.sort_values("mes").groupby("id_cliente").agg({**{k:"first" for k in tr.columns if k not in("id_cliente","mes","objetivo","dias_ultima_interaccion")},"objetivo":"max","mes":["min","count"]})
c.columns=[a if b in("first","max") else f"mes_{b}" for a,b in c.columns]
print("clients",len(c))
num=["edad","ingresos","ratio_deuda_ingresos","antiguedad_cuenta_meses","numero_productos","saldo_promedio","dias_ultima_transaccion","antiguedad_direccion_meses","visitas_web_ultimos_90_dias","distancia_sucursal_km","dia_preferido_pago"]
print("\n=== describe ==="); print(c[num].describe(percentiles=[.01,.5,.99]).T.round(2))
print("\n=== most frequent values (detect floors/caps) ===")
for k in num:
    vc=c[k].value_counts(); print(f"{k}: nunique={c[k].nunique()} top={vc.head(3).round(2).to_dict()}")
print("\n=== row-level conversion rate by decile (shape) ===")
for k in num:
    q=pd.qcut(tr[k],10,duplicates="drop"); r=tr.groupby(q,observed=True).objetivo.mean()
    print(f"{k:28s}", " ".join(f"{v:.3f}" for v in r.values), f"| gini={2*auc(tr.objetivo,tr[k])-1:+.3f}")
print("\n=== logical consistency ===")
flags=["tiene_tarjeta_credito","tiene_prestamo","tiene_seguro"]
c["sum_flags"]=c[flags].sum(1)
print("numero_productos vs sum(card,loan,insurance):\n",pd.crosstab(c.numero_productos,c.sum_flags))
print("es_nuevo_cliente vs antiguedad_cuenta_meses:\n",c.groupby("es_nuevo_cliente").antiguedad_cuenta_meses.describe().round(1))
print("banda_riesgo vs numeric means:\n",c.groupby("banda_riesgo")[["ratio_deuda_ingresos","ingresos","saldo_promedio","edad","tiene_prestamo","numero_productos"]].mean().round(3))
print("activo_movil vs dispositivo / dias_ultima_transaccion / visitas:\n",c.groupby("activo_movil")[["dias_ultima_transaccion","visitas_web_ultimos_90_dias"]].mean().round(1), "\n", pd.crosstab(c.dispositivo_principal,c.activo_movil,normalize="index").round(3))
print("distancia by region:\n",c.groupby("region").distancia_sucursal_km.describe().round(2))
print("canal vs dispositivo:\n",pd.crosstab(c.canal_adquisicion,c.dispositivo_principal,normalize="index").round(3))
print("ocupacion vs ingresos:\n",c.groupby("ocupacion").ingresos.median().round(0))
print("impossible: account older than adult life", (c.antiguedad_cuenta_meses>(c.edad-18)*12).mean().round(4),
      "| address older than age", (c.antiguedad_direccion_meses>c.edad*12).mean().round(4))
print("\nspearman among numerics (|r|>0.1):")
cr=c[num].corr("spearman"); 
for i,a in enumerate(num):
    for b in num[i+1:]:
        if abs(cr.loc[a,b])>0.1: print(f"  {a} ~ {b}: {cr.loc[a,b]:.3f}")
print("\nCramer-ish: categorical pairwise target-rate independence")
cats=["ocupacion","region","canal_adquisicion","banda_riesgo","dispositivo_principal","tiene_tarjeta_credito","activo_movil","es_nuevo_cliente","tiene_prestamo","tiene_seguro"]
from scipy.stats import chi2_contingency
for i,a in enumerate(cats):
    for b in cats[i+1:]:
        t=pd.crosstab(c[a],c[b]); chi,p,_,_=chi2_contingency(t); v=np.sqrt(chi/(t.values.sum()*(min(t.shape)-1)))
        if v>0.05: print(f"  {a} x {b}: V={v:.3f}")
