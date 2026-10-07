import pandas as pd, numpy as np
def rules(d):
    L=(d.banda_riesgo=="low"); H=(d.banda_riesgo=="high"); M=(d.banda_riesgo=="medium")
    p=d.numero_productos; dtx=d.dias_ultima_transaccion/365
    mc=d.activo_movil&d.tiene_tarjeta_credito
    return pd.DataFrame({
      "L":L,"H":H,
      "L_p3":L&(p>=3),"L_mc_p2":L&mc&(p>=2),"L_dtx":L*dtx,
      "H_dtx100":H&(d.dias_ultima_transaccion<100),"H_dtx180":H&(d.dias_ultima_transaccion<180),
      "H_old_branch":H&(d.dias_ultima_transaccion>=180)&(d.canal_adquisicion=="branch"),
      "M_dtx":M*dtx,"k":np.log(d.k)}).astype(float)

def rules2(d):
    r=rules(d); LM=d.banda_riesgo.astype(str).isin(["low","medium"])
    r["LM_debt60"]=(LM&(d.ratio_deuda_ingresos>=0.6)).astype(float)
    r["saldo_floor"]=(d.saldo_promedio<=300).astype(float); return r
