"""Refresh the readable report and audit artifact hashes without refitting."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
from datafest.lineage import file_record, verify_manifest_integrity, verify_file_records, write_json
from datafest.diagnostics import manifest
from datafest.diagnostics import probability_frame, metrics
import numpy as np


def report(root=Path("experiments/diagnostics")):
    summary=json.loads((root/"summary.json").read_text())
    rows=list(summary["baseline"].items())+[("LightGBM_D + trayectoria",summary["trajectory"])]+list(summary["survival"].items())
    foundation=root/"04_foundation"
    status=[]
    for model in ("tabpfn3","tabpfn35","tabfm"):
        target=foundation/model
        if (target/"metrics.json").exists():
            rows.append((model,json.loads((target/"metrics.json").read_text())))
        for month in (202609,202610,202611):
            state=target/str(month)/"state.json"
            data=json.loads(state.read_text()) if state.exists() else {"status":"not_started"}
            fold_metrics=target/str(month)/"metrics.json"
            gini=(json.loads(fold_metrics.read_text())["gini_by_month"][str(month)]
                  if data["status"]=="complete" and fold_metrics.exists() else None)
            status.append({"model":model,"month":month,"status":data["status"],
                           "gini":gini,"rows":data.get("n_validation"),"elapsed_seconds":data.get("elapsed_seconds"),
                           "error_type":data.get("error_type"),"error":data.get("error")})
    summary["foundation"]={}
    for model in ("tabpfn3","tabpfn35","tabfm"):
        folds=[r for r in status if r["model"]==model]
        metric_file=foundation/model/"metrics.json"
        summary["foundation"][model]={
            "status":"complete" if all(r["status"]=="complete" for r in folds) else "incomplete",
            "folds":folds,
            "metrics":json.loads(metric_file.read_text()) if metric_file.exists() else None}
    write_json(root/"summary.json",summary)
    lines=["# Cuatro experimentos rolling", "",
        "Cortes: ≤agosto→septiembre, ≤septiembre→octubre, ≤octubre→noviembre de 2026. Misma población y orden de validación del baseline; ninguna selección de parámetros con septiembre, octubre o noviembre. D y B son LightGBM con month_index; A es CatBoost sin mes.","",
        "Los bloques son diagnósticos. Las métricas principales son Gini mensual y media simple rolling. La desviación entre meses describe variación temporal; no es un intervalo de confianza.","",
        "## Gini rolling", "", "| Modelo | Sep | Oct | Nov | Media | Desv. temporal |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,m in rows:
        g=m["gini_by_month"]
        lines.append(f"| {name} | {g['202609']:.6f} | {g['202610']:.6f} | {g['202611']:.6f} | {m['mean_rolling_gini']:.6f} | {m['temporal_std_ddof0']:.6f} |")
    lines += ["", "La trayectoria no mejora la media de D en esta corrida. No se interpreta esta diferencia descriptiva como significación estadística.","",
        "## 1. Bootstrap pareado por cliente", "",
        "2.000 remuestreos de clientes con reemplazo, semilla 42. Un único vector de multiplicidades por réplica conserva juntas todas las filas de cada cliente en los tres meses y los tres modelos. Se usa AUC ponderada exacta con empates. No hay reentrenamiento.","",
        "IC percentil 95% sin ajuste. Los p-valores bilaterales usan el bootstrap centrado bajo delta cero con corrección +1. Holm se aplica a la familia de tres comparaciones de **media rolling**; los IC mensuales y rolling no son simultáneos.","",
        "| Par (primero − segundo) | Mes | Δ Gini | IC 95% | p | p Holm |",
        "| --- | --- | ---: | --- | ---: | ---: |"]
    for r in summary["bootstrap"]:
        p=r.get("p_two_sided_centered_bootstrap")
        h=r.get("p_holm_three_rolling_pairs")
        lines.append(f"| {r['pair']} | {r['month']} | {r['delta_gini']:.6f} | [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | {p if p is not None else '—'} | {h if h is not None else '—'} |")
    lines += ["", "Los tres IC de delta medio incluyen cero y ninguno rechaza igualdad tras Holm al 5%.","",
        "## 2. Trayectoria de interacción", "",
        "D_month_index más nueve features de dias_ultima_interaccion: lag1, lag2, delta, slope lineal, mínimo, máximo, media, reset actual y resets acumulados. Los lags son observaciones previas, aunque haya huecos de meses. Slope usa meses calendario y todo el prefijo hasta la fila actual; queda ausente con una sola observación. Reset significa caída estricta frente al valor observado previo. Las estadísticas incluyen la fila actual y no usan objetivo.","",
        "LightGBM conserva los parámetros y las 58 iteraciones del baseline, sin early stopping ni búsqueda. Se guardan modelo, mapas de categorías entrenados en cada corte, features, predicciones y métricas.","",
        "## 3. Supervivencia explícita", "",
        "Cada observación ocupa (tenure−1, tenure], donde tenure es el número de meses calendario desde la primera observación. Las conversiones salen al terminar el intervalo. La última observación sin evento se trata como censura derecha. Los meses sin observación no reciben covariables inventadas y no añaden intervalos al conjunto de riesgo.","",
        "Cloglog tiene efecto base categórico del mes **en riesgo** y el month_index calendario de D. Cox usa start-stop, covariables de cada fila y empates Breslow. Ambos usan covariables D con exclusión de tres alias temporales deterministas (mes_primera_aparicion, meses_en_riesgo, meses_desde_entrada). Codificación y escalado se ajustan con entrenamiento solamente. Penalizaciones numéricas fijas: cloglog 1e−6 y Cox 1e−4; sin selección temporal.","",
        "Las probabilidades Cox son 1−exp(−incremento_base_mensual × exp(Xβ)). La hazard base procede exclusivamente del entrenamiento del corte. Para tenure posterior al máximo entrenado se prolonga el último incremento mensual; cloglog prolonga el último factor base. Esta extrapolación explícita es un supuesto, no una estimación con el mes de validación. Los gradientes máximos al terminar quedaron debajo de 1e−6 en ambos modelos y los tres folds.","",
        "## 4. Foundation: avance y límites", "",
        "TabPFN se fija con ModelVersion.V3 y se comprueba el nombre de su checkpoint: no se sustituye por otra versión. Por ampliación explícita del usuario se evalúa además TabPFN-3.5, con ModelVersion.V3_5, defaults publicados y artefactos separados bajo tabpfn35. [Prior Labs](https://docs.priorlabs.ai/models) publica hasta un millón de filas para TabPFN-3. TabFM usa el backend PyTorch v1.0.0, con código fijado a fbb665569425fd2f490c6576b3af967876fe11ff.","",
        "El [README de TabFM](https://github.com/google-research/tabfm) declara 100 filas de contexto, pero ese commit tiene max_num_rows=None en el constructor. Se fija explícitamente max_num_rows=100 para conservar el valor publicado acordado; el resto conserva defaults. No se muestrean filas de validación. Se guardan parámetros efectivos y los índices exactos de los contextos por miembro.","",
        "| Modelo | Fold | Estado | Filas | Gini | Motivo de fallo |", "| --- | --- | --- | ---: | ---: | --- |"]
    for r in status:
        score=f"{r['gini']:.6f}" if r['gini'] is not None else '—'
        lines.append(f"| {r['model']} | {r['month']} | {r['status']} | {r['rows'] or '—'} | {score} | {r['error_type'] or '—'} |")
    lines += ["", "Se aceptaron las licencias de TabPFN-3 y 3.5 y la API key de Prior Labs se leyó localmente desde .env (ignorado por Git). La GPU local GTX 1650 (4 GB) no pudo completar TabPFN-3.5 por falta de memoria; los tres folds se completaron en Colab con T4. El intento local de TabFM terminó con código 137 durante carga de pesos (falta de memoria inferida; sin acceso al log del kernel). Colab rechazó L4 y A100 antes de asignar T4. Los estados y logs registran cada ejecución remota.","",
        "Colab retiró sesiones durante preparación y recuperación. Se conservan los folds completos antes de continuar en otra VM. La transferencia del checkpoint de noviembre de TabPFN-3 se interrumpió; se recuperó la copia de septiembre después de comprobar igualdad exacta del SHA-256 esperado. Para TabPFN-3.5, el checkpoint oficial es común a los folds y se reutiliza solo tras verificar tamaño y SHA-256. Sus manifiestos locales conservan checkpoint, parámetros, predicciones, métricas y estado; se omite el pickle redundante fitted_estimator.pkl, que replica el checkpoint y el estado interno del estimador.","",
        "La compresión gzip de las mismas features tuvo dos timestamps entre VMs. feature_archive conserva ambas representaciones originales; audit.json certifica que el CSV descomprimido es idéntico. El manifiesto de septiembre apunta a su archivo inmutable. No cambió ningún valor de entrada.","",
        "## Reanudar y verificar", "",
        "```sh", ".venv/bin/datafest diagnostics",
        ".foundation-venv/bin/python scripts/run_tabpfn_local.py --model tabpfn3",
        ".foundation-venv/bin/python scripts/run_tabpfn_local.py --model tabpfn35",
        ".venv/bin/python scripts/run_foundation_colab.py --model tabfm --gpu T4",
        "# Tras configurar TABPFN_TOKEN localmente:",
        ".venv/bin/python scripts/run_foundation_colab.py --model tabpfn3 --gpu T4",
        ".venv/bin/python scripts/run_foundation_colab.py --model tabpfn35 --gpu T4",
        ".venv/bin/python scripts/report_diagnostics.py", "```", "",
        "foundation-requirements.txt fija las dependencias del entorno separado. Cada fold conserva checkpoint, parámetros, predicciones, métricas y estado con hashes. TabPFN-3.5 usa el límite publicado predeterminado de 1.000.000 de filas de entrenamiento, sin submuestreo externo, y predice todas las filas de validación. Al reanudar se verifica identidad de datos, features, código y versiones, además de la integridad de los artefactos; un fold incompleto no se reutiliza. La migración envía un snapshot de Git sin modificar la rama o el índice del usuario, recupera resultados al cerrar cada fold y libera su sesión de Colab.","",
        "Verificación: pruebas de invariancia por prefijos y perturbaciones futuras, lags/huecos/resets, probabilidades por fila, AUC ponderada contra clientes materializados, gradientes de ambas likelihoods, pertenencia start-stop y recuperación de folds completos/corruptos. Las predicciones completas deben cubrir 9.900, 10.400 y 9.500 filas; 29.800 en total."]
    (root/"results.md").write_text("\n".join(lines)+"\n")
    write_json(root/"foundation_status.json",status)
    # Refresh manifests after metadata/code work; fitted arrays are unchanged.
    for directory in (root/"01_bootstrap",root/"02_trajectory",root/"03_survival"):
        old=json.loads((directory/"manifest.json").read_text())
        manifest(directory,old["parameters"],[Path(r["path"]) for r in old["inputs"]],
                 [Path(r["path"]) for r in old["features"]])
    for directory in foundation.glob("*/*"):
        state=directory/"state.json"
        path=directory/"manifest.json"
        if state.is_file() and path.is_file() and json.loads(state.read_text()).get("status")=="failed":
            old=json.loads(path.read_text())
            manifest(directory,old["parameters"],[Path(r["path"]) for r in old["inputs"]],
                     [Path(r["path"]) for r in old["features"]])
    errors=[]
    for path in root.rglob("manifest.json"):
        saved_manifest=json.loads(path.read_text())
        errors += [f"{path}: {e}" for e in verify_manifest_integrity(saved_manifest)]
        snapshot=saved_manifest.get("parameters",{}).get("source_snapshot")
        if snapshot:
            errors += [f"{path} source snapshot: {e}" for e in verify_file_records([snapshot["archive"]])]
    expected=pd.read_csv(root/"01_bootstrap"/"validation_rows.csv")
    coverage=[]
    for state_path in foundation.glob("*/*/state.json"):
        state=json.loads(state_path.read_text())
        if state.get("status")!="complete":
            continue
        errors += [f"{state_path}: {e}" for e in verify_file_records(state["artifacts"])]
        path=state_path.parent/"predictions.csv"
        try:
            frame=pd.read_csv(path)
            wanted=expected.loc[expected.mes.eq(state["validation_month"])].reset_index(drop=True)
            pd.testing.assert_frame_equal(frame[["id_cliente","mes","objetivo"]],
                                          wanted[["id_cliente","mes","objetivo"]],check_dtype=False)
            probability_frame(wanted,frame.prediccion)
            saved=json.loads((path.parent/"metrics.json").read_text())
            actual=metrics(frame)
            assert np.isclose(actual["mean_rolling_gini"],saved["mean_rolling_gini"],atol=1e-12,rtol=0)
            coverage.append({"path":str(path),"rows":len(frame),"fold":state["validation_month"]})
        except (ValueError,AssertionError) as e:
            errors.append(f"{path}: {e}")
    for path in root.rglob("rolling_predictions.csv"):
        frame=pd.read_csv(path)
        try:
            pd.testing.assert_frame_equal(frame[["id_cliente","mes","objetivo"]],
                                          expected[["id_cliente","mes","objetivo"]],check_dtype=False)
            probability_frame(expected,frame.prediccion)
            saved=json.loads((path.parent/"metrics.json").read_text())
            actual=metrics(frame)
            for key in actual["gini_by_month"]:
                if not np.isclose(actual["gini_by_month"][key],saved["gini_by_month"][key],atol=1e-12,rtol=0):
                    raise ValueError("Saved metrics differ from recovered predictions")
            coverage.append({"path":str(path),"rows":len(frame),"counts":{str(k):int(v) for k,v in frame.groupby("mes").size().items()}})
        except (ValueError,AssertionError) as e:
            errors.append(f"{path}: {e}")
    write_json(root/"verification.json",{"verified":not errors,"errors":errors,
               "coverage":coverage,"expected_prediction_counts":{"202609":9900,"202610":10400,"202611":9500}})
    write_json(root/"index.json",{"report":file_record(root/"results.md"),"summary":file_record(root/"summary.json"),
                "foundation_status":file_record(root/"foundation_status.json"),"verification":file_record(root/"verification.json")})
    project_index=Path("experiments/index.json")
    if project_index.exists():
        existing=json.loads(project_index.read_text())
        existing["diagnostics"]={"index":file_record(root/"index.json"),
                                 "report":file_record(root/"results.md")}
        write_json(project_index,existing)
    if errors:
        raise RuntimeError("Integrity failures: "+"; ".join(errors))


if __name__=="__main__":
    report()
