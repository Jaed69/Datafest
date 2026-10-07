import pandas as pd, numpy as np, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
D=sys.argv[1]
BG="#F7F5F0"; INK="#14213D"; INK2="#4A5568"; MUTED="#8A94A6"; GRID="#E3E0D8"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; GRAY="#B9BFC9"
plt.rcParams.update({"font.family":"Arial","font.size":17,"text.color":INK,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,
  "axes.edgecolor":GRID,"axes.facecolor":BG,"figure.facecolor":BG,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":1})
W,H,DPI=13.87,5.4,120
from matplotlib.ticker import FuncFormatter
COMMA=lambda d: FuncFormatter(lambda v,_: (f"{v:.{d}f}").replace(".",","))
MES=["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
def save(fig,name): fig.savefig(f"../charts/{name}.png",dpi=DPI,facecolor=BG); plt.close(fig)
tr=pd.read_csv(f"{D}/train.csv"); te=pd.read_csv(f"{D}/test.csv")
# 1 data evolution
entry=tr.groupby("id_cliente").mes.min(); new=tr.mes.map(lambda m:0)  # placeholder
n=tr.groupby("mes").size(); nuevos=entry.value_counts().sort_index(); cont=n-nuevos
cont=list(cont.values)+[8061]; nuevos=list(nuevos.values)+[1839]; rate=tr.groupby("mes").objetivo.mean().values
fig,(a,b)=plt.subplots(1,2,figsize=(W,H),gridspec_kw={"width_ratios":[1.25,1]})
x=np.arange(12)
a.bar(x,cont,width=0.72,color=BLUE,label="Continúan del mes anterior",zorder=3)
a.bar(x,nuevos,bottom=cont,width=0.72,color=ORANGE,label="Clientes nuevos",zorder=3,edgecolor=BG,linewidth=2)
a.set_xticks(x,MES); a.set_title("Clientes por mes",loc="left",fontsize=19,fontweight="bold",color=INK)
a.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.1),ncol=2,fontsize=15); a.set_ylim(0,11500); a.grid(axis="x",visible=False); a.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:,.0f}".replace(",",".")))
a.annotate("Diciembre = test",(11,9900),xytext=(7.6,11000),fontsize=15,color=INK2,arrowprops=dict(arrowstyle="-",color=MUTED))
b.plot(np.arange(11),rate*100,color=BLUE,lw=2.5,marker="o",ms=8,zorder=3)
for i,r in enumerate(rate*100):
    if i in (0,6,9,10): b.annotate(f"{r:.1f}%".replace(".",","),(i,r),xytext=(12 if i==0 else 0,-28 if i in (0,6) else 12),textcoords="offset points",ha="center",fontsize=15,color=INK2)
b.set_xticks(np.arange(11),MES[:11]); b.set_ylim(10,20); b.set_xlim(-0.6,10.6); b.set_title("Tasa de conversión mensual (%)",loc="left",fontsize=19,fontweight="bold",color=INK); b.grid(axis="x",visible=False)
fig.tight_layout(); save(fig,"evolucion_data")
# 2 walk forward
w=pd.read_csv("walk_forward.csv"); xm=[MES[int(str(m)[-2:])-1] for m in w.mes]; xi=np.arange(len(w))
fig,ax=plt.subplots(figsize=(W,H))
ax.axvspan(6.5,9.5,color="#EAE6DC",zorder=0); ax.text(8,0.283,"Validación oficial",ha="center",fontsize=15,color=INK2)
for col,c,lab in [("LGB_acumulada",BLUE,"LightGBM, ventana acumulada"),("LGB_movil3",ORANGE,"LightGBM, últimos 3 meses"),("Reglas_acumulada",AQUA,"12 reglas, ventana acumulada")]:
    ax.plot(xi,w[col],color=c,lw=2.5,marker="o",ms=8,label=lab,zorder=3)
ax.set_xticks(xi,[f"{m}\n{n//1000}k" for m,n in zip(xm,w.n_train)]); ax.set_ylim(0.21,0.29); ax.set_ylabel("Gini del mes predicho")
ax.set_xlabel("Mes predicho  ·  filas de entrenamiento disponibles"); ax.grid(axis="x",visible=False)
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.14),ncol=3,fontsize=15)
fig.tight_layout(); save(fig,"walk_forward")
# 3 model progression
prog=[("Logística con binning WoE",0.213),("Supervivencia (Cox)",0.224),("TabPFN-3 (foundation)",0.234),("LightGBM D",0.2510),("H4b2: LightGBM + CatBoost",0.2525),("H4b2+N1: + RealMLP",0.2543)]
fig,ax=plt.subplots(figsize=(W,H)); y=np.arange(len(prog))[::-1]
cols=[MUTED]*5+[BLUE]
ax.hlines(y,0.205,[p[1] for p in prog],color=GRID,lw=2,zorder=2)
ax.scatter([p[1] for p in prog],y,s=[180]*5+[300],color=cols,zorder=3,edgecolor=BG,linewidth=2)
for yy,(nme,v) in zip(y,prog): ax.text(v+0.0018,yy,f"{v:.4f}".replace(".",","),va="center",fontsize=16,color=INK,fontweight="bold" if nme.startswith("H4b2+N1") else "normal")
ax.set_yticks(y,[p[0] for p in prog]); ax.set_xlim(0.205,0.262); ax.grid(axis="y",visible=False); ax.xaxis.set_major_formatter(COMMA(2)); ax.set_xlabel("Gini medio, validación rolling septiembre–noviembre")
fig.tight_layout(); save(fig,"progresion_modelos")
# 4 validation scheme
fig,ax=plt.subplots(figsize=(W,H)); ax.grid(False)
rowsS=[("Fold Sep",8),("Fold Oct",9),("Fold Nov",10),("Entrega Dic",11)]
for r,(lab,vm) in enumerate(rowsS):
    yy=len(rowsS)-1-r
    for c in range(12):
        col= BLUE if c<vm else (ORANGE if c==vm else "#E3E0D8")
        ax.add_patch(Rectangle((c+0.04,yy+0.12),0.92,0.76,color=col,lw=0))
    ax.text(-0.15,yy+0.5,lab,ha="right",va="center",fontsize=17,color=INK)
for c in range(12): ax.text(c+0.5,4.25,MES[c],ha="center",fontsize=16,color=INK2)
ax.set_xlim(-2.3,12.1); ax.set_ylim(-0.9,4.6); ax.axis("off")
ax.add_patch(Rectangle((1.5,-0.75),0.5,0.4,color=BLUE)); ax.text(2.15,-0.55,"Entrenamiento (solo pasado)",va="center",fontsize=16)
ax.add_patch(Rectangle((6.6,-0.75),0.5,0.4,color=ORANGE)); ax.text(7.25,-0.55,"Mes que se predice",va="center",fontsize=16)
fig.tight_layout(); save(fig,"esquema_validacion")
# 5 CV comparison
cv=pd.read_csv("cv_compare.csv",index_col=0).loc[["KFold aleatorio","GroupKFold por cliente","Temporal sep-nov"]]
labels=["Mezclando meses\n(KFold aleatorio)","Mezclando meses\n(agrupado por cliente)","Temporal\n(pasado → futuro)"]
fig,ax=plt.subplots(figsize=(W,H)); xi=np.arange(3)
for i,c in enumerate([MUTED,MUTED,BLUE]): ax.errorbar(i,cv["mean"].iloc[i],yerr=cv["std"].iloc[i],fmt="o",ms=16,color=c,ecolor=c,elinewidth=2.5,capsize=10,zorder=3,mec=BG,mew=2)
for i,v in enumerate(cv["mean"]): ax.text(i+0.08,v,f"{v:.3f}".replace(".",","),ha="left",va="center",fontsize=18,fontweight="bold")
ax.set_xticks(xi,labels); ax.set_xlim(-0.5,2.6); ax.set_ylim(0.225,0.28); ax.yaxis.set_major_formatter(COMMA(2)); ax.set_ylabel("Gini medio (± desvío)"); ax.grid(axis="x",visible=False)
fig.tight_layout(); save(fig,"cv_comparacion")
# 6 candidates by month
cand={"H4b2+N1 (elegido)":[0.24867,0.26727,0.24701],"H4b2":[0.24715,0.26804,0.24216],"LightGBM D":[0.2434,0.2677,0.2420]}
fig,ax=plt.subplots(figsize=(W,H)); xi=np.arange(3); ww=0.26
for j,((k,v),c) in enumerate(zip(cand.items(),[BLUE,ORANGE,AQUA])):
    ax.plot(xi,v,color=c,lw=2.5 if j else 3.5,marker="o",ms=10 if j else 13,label=k,zorder=4-j,mec=BG,mew=2)
    ax.text(2.08,v[2]+(0.0010 if j==0 else (0.0005 if j==1 else -0.0012)),(k.split(" (")[0]+"  "+f"{v[2]:.3f}".replace(".",",")),va="center",fontsize=15,color=INK)
ax.set_xticks(xi,["Septiembre","Octubre","Noviembre"]); ax.set_xlim(-0.2,2.75); ax.set_ylim(0.238,0.272); ax.yaxis.set_major_formatter(COMMA(3)); ax.set_ylabel("Gini"); ax.grid(axis="x",visible=False)
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.13),ncol=3,fontsize=15)
fig.tight_layout(); save(fig,"candidatos_mes")
print("ok")
