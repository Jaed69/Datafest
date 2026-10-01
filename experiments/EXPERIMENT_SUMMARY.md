# Resumen experimental — Datafest

> Fuente de verdad experimental para el equipo. Actualizado con artefactos existentes hasta el 1 de octubre de 2026. No se entrenaron modelos para preparar este documento.

## Estado actual

- **Mejor Gini rolling observado:** **0.25104**, LightGBM **D + AAAAMM** y D + `month_index` (empate exacto a la precisión almacenada). Usa features base + cinco features de historial + mes; excluye `id_cliente`.
- **Siguientes referencias:** CatBoost B + AAAAMM **0.24948** y CatBoost A **0.24932**. La diferencia con el mejor es solo **0.00156** y **0.00173** Gini; aún es pequeña y no concluyente. LightGBM B queda en **0.24869**.
- **ID crudo:** no está en el baseline. El gate por cliente no se superó para LightGBM ni CatBoost y el resultado rolling actual lo excluye por diseño.
- **Mes:** parece útil para LightGBM; ambas codificaciones dieron los mismos scores por variante en esta corrida. CatBoost muestra diferencias pequeñas.
- **Historial:** por sí solo no mejora de forma consistente; combinado con mes sí ayuda a LightGBM, pero no a CatBoost en esta corrida.
- **Modelos:** LightGBM D es la referencia rolling; CatBoost A/B sigue siendo competitivo. No hay evidencia para declarar un ganador definitivo.

## Problema

Cada fila representa un par cliente-mes. `objetivo=1` significa que el cliente realiza su **primera conversión** durante ese mes. Después de convertir, sale del conjunto en riesgo y no vuelve a aparecer. La estructura es longitudinal y equivale a datos persona-período para un problema de hazard discreto: se ordenan clientes por su propensión de conversión en cada período. El objetivo práctico es el ranking; no se ha demostrado calibración perfecta de probabilidades.

El entrenamiento disponible cubre enero–noviembre de 2026; el test final corresponde a diciembre y no tiene etiqueta. La métrica de competencia es `Gini = 2 × ROC_AUC − 1`.

## Protocolo oficial de comparación

Desde este resumen, la comparación principal del equipo es el rolling temporal:

| Corte | Entrenamiento | Validación |
|---|---|---|
| 1 | ≤ agosto | septiembre |
| 2 | ≤ septiembre | octubre |
| 3 | ≤ octubre | noviembre |

Para cada alternativa se reportan Gini de septiembre, octubre y noviembre, media simple de los tres y desviación estándar poblacional (`ddof=0`). Este protocolo aproxima el despliegue final `enero–noviembre → diciembre`: cada predicción usa la historia disponible hasta el mes anterior y se reentrena cada mes.

Hiperparámetros estructurales deben quedar fijados antes de evaluar octubre y noviembre. En la ablación actual, los parámetros base son comunes dentro de cada algoritmo y las iteraciones se determinaron con folds internos hasta agosto sobre A/C; no hubo early stopping en los cortes externos. Los parámetros y conteos exactos se registran en el manifiesto de la ablación.

### Métricas secundarias

- **Frozen:** entrenar hasta septiembre y puntuar octubre y noviembre con ese mismo modelo. Mide robustez ante el envejecimiento del modelo por más de un mes.
- **Pooled octubre-noviembre:** concatenar octubre y noviembre y calcular una sola AUC/Gini. Es secundaria; no se usa para elegir el baseline, porque mezcla comparaciones intra e intermes.

## Cómo leer nuestros resultados

### Gini y ROC AUC

`Gini = 2 × AUC − 1`; mayor es mejor. Gini 0 equivale a AUC 0.50 (ranking aleatorio); Gini 0.25 equivale a AUC 0.625; Gini 0.30 equivale a AUC 0.65. ROC AUC mide la capacidad de ordenar positivos por encima de negativos, no calibración.

La media rolling es el resumen principal actual. Una diferencia de 0.0005, 0.001 o 0.002 no es automáticamente significativa. Antes de promocionar una mejora pequeña, revisar consistencia entre meses y calcular intervalos mediante bootstrap pareado por cliente; para modelos con aleatoriedad, confirmar también con varias seeds.

### Desviación temporal

La std de los tres Gini describe cuánto varió el rendimiento observado entre meses. No es un intervalo de confianza ni una medida directa de incertidumbre estadística. Una media mayor puede venir acompañada de más inestabilidad.

### PR AUC / Average Precision

El pipeline calcula `average_precision_score` (Average Precision), no la integración trapezoidal de la curva PR. Es secundaria: la competencia selecciona por Gini.

### Pooled frente a media mensual

`Gini(concatenado) ≠ mean(Gini_mes)` en general. El pooled puede mejorar por diferencias de escala/ranking entre los scores de distintos meses. No debe ponerse en la misma columna ni compararse como si fuera la media rolling.

## Tabla maestra: ablación rolling comparable

A–D comparten los mismos cortes y parámetros baseline dentro de cada algoritmo. **Negrita** marca los máximos observados por columna; los empates se muestran en todas las filas correspondientes.

| Modelo | Variante | Representación mes | Historial | ID crudo | Gini Sep | Gini Oct | Gini Nov | Mean | Std |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| catboost | A | — | No | No | **0.24713** | 0.26508 | 0.23575 | 0.24932 | 0.01207 |
| catboost | B_absolute | AAAAMM | No | No | 0.24633 | 0.26756 | 0.23456 | 0.24948 | 0.01365 |
| catboost | B_month_index | month_index | No | No | 0.24419 | 0.26171 | 0.23373 | 0.24655 | 0.01154 |
| catboost | C | — | Sí | No | 0.22701 | 0.26520 | 0.23372 | 0.24197 | 0.01665 |
| catboost | D_absolute | AAAAMM | Sí | No | 0.24091 | 0.26499 | 0.23891 | 0.24827 | 0.01185 |
| catboost | D_month_index | month_index | Sí | No | 0.23128 | 0.26544 | 0.23840 | 0.24504 | 0.01471 |
| lightgbm | A | — | No | No | 0.23931 | 0.26290 | 0.23509 | 0.24577 | 0.01223 |
| lightgbm | B_absolute | AAAAMM | No | No | 0.24193 | **0.27045** | 0.23368 | 0.24869 | 0.01575 |
| lightgbm | B_month_index | month_index | No | No | 0.24193 | **0.27045** | 0.23368 | 0.24869 | 0.01575 |
| lightgbm | C | — | Sí | No | 0.23867 | 0.26412 | 0.23158 | 0.24479 | 0.01397 |
| lightgbm | D_absolute | AAAAMM | Sí | No | 0.24337 | 0.26773 | **0.24203** | **0.25104** | 0.01181 |
| lightgbm | D_month_index | month_index | Sí | No | 0.24337 | 0.26773 | **0.24203** | **0.25104** | 0.01181 |

## Variantes exactas

| Variante | Features incluidas |
|---|---|
| A — Base | Features originales, excluyendo `id_cliente`, `mes` y `objetivo`. |
| B — Base + mes | A + `mes` codificado como AAAAMM entero o como `month_index` entero (enero=0,…,noviembre=10). |
| C — Base + historial | A + `n_observaciones_previas`, `mes_primera_aparicion`, `meses_desde_entrada`, `meses_en_riesgo`, `cliente_recurrente`. |
| D — Base + historial + mes | C + `mes` como AAAAMM o `month_index`. |

“Sin mes” significa sin la columna del mes actual `mes`; C conserva las variables de calendario incluidas explícitamente en el historial, en particular `mes_primera_aparicion`, además de la tenure. `id_cliente` se usa para agrupar las filas y construir el historial causal, pero nunca es predictor en A–D. El historial se construye sin `objetivo`, con información observable del mes actual y los previos. En secuencias mensuales sin huecos, `meses_en_riesgo ≈ n_observaciones_previas + 1`; `cliente_recurrente` equivale a `I(n_observaciones_previas>0)`. Por eso estas cinco variables contienen información solapada. `mes_primera_aparicion` y `meses_desde_entrada` añaden cohorte/tenure, no una historia rica de interacciones.

## Qué aprendimos de cada familia

- **ID crudo:** los IDs están asignados en bloques crecientes por mes de primera aparición en los datos; por tanto codifican fuertemente cohorte. Sin embargo, el gate por grupos no encontró mejora robusta. En OOF, `ID − historial` fue −0.00109 Gini en LightGBM (IC95% [−0.00687, 0.00401]) y −0.00721 en CatBoost (IC95% [−0.01400, 0.00000]); los gates quedaron rechazados. Mantener ID fuera del baseline.
- **Mes:** en LightGBM, B mejora A en 0.00292 Gini y D mejora C en 0.00626 con AAAAMM. AAAAMM y `month_index` dieron scores idénticos en LightGBM. CatBoost distingue las representaciones modestamente; no hay evidencia de que AAAAMM tenga más información, y una transformación afín puede cambiar sus umbrales internos de cuantización.
- **Historial:** C queda por debajo de A en ambos modelos (`−0.00098` LightGBM; `−0.00734` CatBoost). Al añadir mes, LightGBM D-AAAAMM supera B-AAAAMM en `0.00236`; CatBoost D queda `0.00121` debajo de B. Interpretar las variables actuales principalmente como exposición, tenure y cohorte.
- **CatBoost vs LightGBM:** ambos son competitivos. El máximo rolling favorece LightGBM D; CatBoost B-AAAAMM y A están a menos de 0.002 Gini. La diferencia aún requiere bootstrap pareado con predicciones rolling para sostener una promoción.

## Comparaciones inválidas o engañosas

- Pooled Oct–Nov frente a media rolling mensual: resumen y población de predicciones distintos.
- Frozen Oct/Nov frente a rolling Oct/Nov: el modelo frozen se entrenó hasta septiembre; el rolling de noviembre se entrenó hasta octubre.
- Corridas iniciales `clean/history/id_raw` frente a A–D: las iniciales usaban un único modelo entrenado hasta septiembre y sus variantes `clean` y `history` **sí incluían `mes`**. Además, `clean` e `history` se ajustaron por variante, mientras `id_raw` quedó en la configuración baseline al no superar su gate; la ablación reciente usa parámetros estructurales baseline comunes entre variantes del mismo algoritmo.
- Resultados con y sin mes, historial o ID crudo sin indicar la variante y codificación.
- Comparar configuraciones tuned con baseline o distinto número de árboles sin declararlo.
- Tomar el `0.25394` de LightGBM `clean` inicial como equivalente al `0.25104` de LightGBM D: el primero es pooled frozen con tuning por variante; el segundo es media de tres Gini rolling con configuración baseline común.

### Corridas iniciales: referencia frozen pooled, no rolling

Estas corridas sirven como antecedente y diagnóstico. No reemplazan la tabla rolling y no deben ordenarse junto a ella como una sola clasificación.

| Modelo | Variante inicial | Gini pooled Oct-Nov | Gini Oct | Gini Nov | Iteraciones | Configuración seleccionada | ID gate | Manifiesto |
|---|---|---:|---:|---:|---:|---|---|---|
| catboost | clean | 0.24483 | 0.25852 | 0.22933 | 208 | deep (depth=8, lr=0.05, L2=5.0) | No aplica | [20261001T040836113543Z_7e1c7547_catboost_clean](catboost/runs/20261001T040836113543Z_7e1c7547_catboost_clean/manifest.json) |
| catboost | history | 0.25037 | 0.26782 | 0.23066 | 267 | compact (depth=5, lr=0.03, L2=3.0) | No aplica | [20261001T040836113543Z_7e1c7547_catboost_history](catboost/runs/20261001T040836113543Z_7e1c7547_catboost_history/manifest.json) |
| catboost | id_raw | 0.24725 | 0.25630 | 0.23702 | 150 | baseline (depth=6, lr=0.05, L2=3.0) | No superado | [20261001T040836113543Z_7e1c7547_catboost_id_raw](catboost/runs/20261001T040836113543Z_7e1c7547_catboost_id_raw/manifest.json) |
| lightgbm | clean | 0.25394 | 0.27221 | 0.23329 | 101 | compact (leaves=15, depth=5, lr=0.03, lambda=1.0) | No aplica | [20261001T040836113543Z_7e1c7547_lightgbm_clean](lightgbm/runs/20261001T040836113543Z_7e1c7547_lightgbm_clean/manifest.json) |
| lightgbm | history | 0.25360 | 0.27173 | 0.23305 | 151 | compact (leaves=15, depth=5, lr=0.03, lambda=1.0) | No aplica | [20261001T040836113543Z_7e1c7547_lightgbm_history](lightgbm/runs/20261001T040836113543Z_7e1c7547_lightgbm_history/manifest.json) |
| lightgbm | id_raw | 0.25187 | 0.26987 | 0.23147 | 52 | baseline (leaves=31, depth=-1, lr=0.05, lambda=0.0) | No superado | [20261001T040836113543Z_7e1c7547_lightgbm_id_raw](lightgbm/runs/20261001T040836113543Z_7e1c7547_lightgbm_id_raw/manifest.json) |

La corrida inicial seleccionó su “champion” por media de folds internos y comparó el holdout pooled después. LightGBM history fue apenas inferior a clean (`0.25360` vs `0.25394`); la diferencia pareada champion-runner fue −0.00034 Gini con IC95% [−0.00456, 0.00400], por lo que el estado quedó `no_conclusive_winner`. CatBoost history fue superior a clean en ese holdout pooled, pero las configuraciones y las features difieren de la ablación rolling actual.

## Baseline experimental vigente

**LightGBM D + `month_index`** (o AAAAMM, empatado en estos datos): referencia para futuros experimentos. Incluye las features base, las cinco features de historial y mes; no incluye ID crudo. Usa configuración `baseline` del proyecto: `num_leaves=31`, `max_depth=-1`, `learning_rate=0.05`, `min_child_samples=20`, `reg_lambda=0`; **58 iteraciones**, seed 42. Los hiperparámetros estructurales se fijaron antes de leer octubre/noviembre; las iteraciones se derivaron de A/C hasta agosto.

| Gini Sep | Gini Oct | Gini Nov | Mean | Std |
|---:|---:|---:|---:|---:|
| 0.24337 | 0.26773 | 0.24203 | **0.25104** | **0.01181** |

“Baseline vigente” significa punto de comparación experimental para cortes idénticos, no el modelo final de diciembre ni una victoria estadística. Referencias adicionales: CatBoost A (media 0.24932), CatBoost B-AAAAMM (0.24948) y LightGBM B (0.24869). CatBoost baseline: `depth=6`, `learning_rate=0.05`, `l2_leaf_reg=3`, 192 iteraciones, seed 42.

## Frozen: envejecimiento del modelo

| Modelo | Variante | Representación mes | Gini Oct | Gini Nov | Mean | Std | Pooled Oct-Nov (sec.) |
|---|---|---:|---:|---:|---:|---:|---:|
| catboost | A | — | 0.26508 | 0.23614 | 0.25061 | 0.01447 | 0.25138 |
| catboost | B_absolute | AAAAMM | 0.26756 | 0.23045 | 0.24900 | 0.01855 | 0.25009 |
| catboost | B_month_index | month_index | 0.26171 | 0.23601 | 0.24886 | 0.01285 | 0.24962 |
| catboost | C | — | 0.26520 | 0.23654 | **0.25087** | 0.01433 | 0.25166 |
| catboost | D_absolute | AAAAMM | 0.26499 | 0.23267 | 0.24883 | 0.01616 | 0.25012 |
| catboost | D_month_index | month_index | 0.26544 | 0.22870 | 0.24707 | 0.01837 | 0.24808 |
| lightgbm | A | — | 0.26290 | 0.23402 | 0.24846 | 0.01444 | 0.24939 |
| lightgbm | B_absolute | AAAAMM | 0.27045 | 0.22457 | 0.24751 | 0.02294 | 0.24886 |
| lightgbm | B_month_index | month_index | 0.27045 | 0.22457 | 0.24751 | 0.02294 | 0.24886 |
| lightgbm | C | — | 0.26412 | 0.23069 | 0.24740 | 0.01671 | 0.24855 |
| lightgbm | D_absolute | AAAAMM | 0.26773 | 0.22362 | 0.24567 | 0.02206 | 0.24708 |
| lightgbm | D_month_index | month_index | 0.26773 | 0.22362 | 0.24567 | 0.02206 | 0.24708 |

Los mejores promedios frozen son CatBoost C (0.25087) y, entre LightGBM, A (0.24846). CatBoost C supera A en apenas 0.00026 frozen; el resultado respalda investigarlo como robustez, no reemplazar el baseline rolling. La columna pooled está incluida solo como secundaria y no se usa para selección.

## Incertidumbre y promoción

Para promover un modelo como candidato principal, revisar en este orden: (1) media rolling, (2) rendimiento en cada mes, (3) peor degradación mensual, (4) std temporal, (5) bootstrap pareado por cliente frente al baseline y (6) diversidad útil si se evalúa un ensemble. Para diferencias menores a aproximadamente **0.002 Gini**, usar por ahora la etiqueta operativa “marginal / requiere confirmación”; no es un umbral estadístico universal. Calcular el bootstrap sobre predicciones alineadas por cliente y corte; no reutilizar como si fuera equivalente el bootstrap pooled de la corrida inicial.

La auditoría inicial usó bootstrap por cliente para el gate de ID (1,000 réplicas) y para la comparación pooled champion-runner (2,000 réplicas). **No hay IC pareado disponible para las variantes rolling A–D** ni repeticiones por seed; son pendientes antes de declarar diferencias pequeñas concluyentes.

## Backlog experimental priorizado

| Prioridad | Experimento | Hipótesis | Qué compara | Criterio de éxito |
|---|---|---|---|---|
| Alta | Features dinámicas derivadas de `dias_ultima_interaccion` | Cambios/recencia entre meses aportan señal más allá del valor de fila | Lags, delta y resumen causal vs D | Mejora rolling consistente; no depender de un solo mes; bootstrap pareado positivo |
| Alta | Hazard discreto explícito / supervivencia | Separar hazard base temporal y covariables puede representar mejor riesgo y censura | Clasificador actual persona-período vs formulación explícita de hazard | Mejora media sin deterioro fuerte por mes; evaluación temporal idéntica |
| Alta | Blend LightGBM D + CatBoost A/B | Errores complementarios pueden mejorar ranking | Pesos fijados solo con cortes internos vs cada miembro | Mejora fuera de selección en los tres meses, con bootstrap y pesos estables |
| Alta | Bootstrap pareado de candidatos actuales | Cuantificar incertidumbre de diferencias de 0.001–0.002 | LightGBM D vs CatBoost B/A y LightGBM B, por cliente y mes | IC de delta estrecho y compatible con mejora; reportar también inconcluso |
| Alta | TabPFN-3 | Un predictor tabular alternativo podría generalizar mejor | Mismos splits y esquema de features, con factibilidad/versión documentada | Superar baseline en media y estabilidad; sin leakage ni tuning del holdout |
| Alta | TabFM | Un modelo tabular fundacional puede aportar ranking complementario | Mismo protocolo que LightGBM/CatBoost | Mejora rolling o diversidad que mejore un blend validado |
| Exploratoria | TimesFM-3 como generador de representaciones/features | Una representación temporal puede capturar trayectoria de cliente | Features generadas causalmente vs D | Ganancia rolling repetible; documentar cómo se adapta un modelo temporal a estas series |
| Exploratoria | Jev como zero-shot/señal auxiliar | Una señal externa podría ordenar clientes complementariamente | Señal zero-shot sola y blend frente al baseline | Ganancia fuera de muestra y disponibilidad/reproducibilidad confirmadas |

Estas filas son hipótesis, no resultados ni afirmaciones de superioridad. La disponibilidad, versión y coste de TabPFN-3, TabFM, TimesFM-3 y Jev no están verificados en los artefactos actuales.

## Historial experimental

| Fecha/lote | Qué se hizo | Referencias |
|---|---|---|
| 2026-10-01, lote `20261001T040836113543Z_7e1c7547` | Primer benchmark de LightGBM/CatBoost con `clean`, `history` e `id_raw`; tuning en folds internos y holdout congelado Oct–Nov. Ningún candidato quedó concluyente. | [Índice](index.json), [split](../data/splits/temporal_202601-202609__202610-202611/manifest.json); manifiestos de las corridas enlazados en la tabla anterior. |
| 2026-10-01, auditoría metodológica | Inspección de predicciones, matrices, splits, modelos y hashes existentes; sin entrenamiento. Confirmó fórmulas de historial y checksums. | [Evidencia de auditoría](../audit/methodology_evidence.json) |
| 2026-10-01, lote `20261001T051935820299Z_7e1c7547` | Ablación A–D, dos representaciones del mes, rolling y frozen. | [Manifiesto](ablation/runs/20261001T051935820299Z_7e1c7547/manifest.json), [resultados](ablation/runs/20261001T051935820299Z_7e1c7547/results.md), [CSV](ablation/runs/20261001T051935820299Z_7e1c7547/summary.csv), [predicciones y modelos](ablation/runs/20261001T051935820299Z_7e1c7547/). |

## Plantilla para nuevas corridas

### Nuevo experimento
- Fecha/run:
- Hipótesis:
- Cambio respecto al baseline:
- Features:
- Modelo:
- Split:
- Hiperparámetros:
- Gini Sep:
- Gini Oct:
- Gini Nov:
- Mean:
- Std:
- Delta vs baseline:
- Bootstrap/CI:
- Conclusión:
- ¿Promover?: sí/no/inconcluso
