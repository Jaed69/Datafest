# Datafest: propensión de conversión de clientes

## Caso

El proyecto aborda una competencia de propensión de conversión para una entidad financiera. Cada fila representa un cliente en un mes; un cliente puede aparecer en varios meses. La etiqueta `objetivo` vale 1 cuando ocurre su primera conversión en ese mes y 0 cuando no ocurre. Después de convertir, el cliente deja de aparecer.

El conjunto contiene 110.100 observaciones etiquetadas de enero a noviembre de 2026 y 9.900 observaciones de prueba de diciembre. La unidad de predicción y evaluación es cliente-mes. La entrega debe conservar una fila por cada registro de prueba y tener las columnas `id_cliente,prediccion`, con una probabilidad entre 0 y 1.

## Objetivo

Ordenar las observaciones de diciembre según su propensión de conversión. La métrica de la competencia es Gini, calculada como `2 × ROC AUC − 1`. Las decisiones de modelado se evalúan con cortes temporales para aproximar el uso real: predecir meses futuros con la información disponible hasta ese momento.

## Planteamiento y protocolo

El historial de un cliente aporta información, pero solo se puede construir con observaciones conocidas en la fecha de predicción. Además, la etiqueta describe un evento único y la población cambia mes a mes cuando los clientes convierten. Por ello, el proyecto separa tres preguntas: cuánto aporta el calendario, cuánto aporta el historial causal y si formulaciones de supervivencia o modelos tabulares foundation mejoran el ordenamiento.

El protocolo de validación principal es rolling:

| Entrenamiento disponible | Validación |
| --- | --- |
| Hasta agosto de 2026 | Septiembre de 2026 |
| Hasta septiembre de 2026 | Octubre de 2026 |
| Hasta octubre de 2026 | Noviembre de 2026 |

La comparación principal presenta Gini por mes y la media aritmética simple de los tres meses. La desviación entre meses describe variación temporal; no es un intervalo de confianza. Octubre y noviembre no se usan para buscar hiperparámetros ni para early stopping. La evaluación frozen hasta septiembre se conserva como análisis complementario. El conjunto de diciembre permanece separado para la entrega final.

## Metodología

Las corridas registran configuración, artefactos, predicciones y métricas en manifiestos con hashes SHA-256. Se preservan las predicciones por fila para que las métricas se puedan recalcular y auditar. Las transformaciones aprendidas, como codificación de categorías o escalado, se ajustan solo con el entrenamiento de cada corte.

Las comparaciones principales excluyen `id_cliente` como predictor. El ID crudo se trata como diagnóstico y no como señal candidata a menos que supere controles temporales y por grupos de clientes. Las variables históricas usan únicamente el prefijo temporal observado del cliente y nunca `objetivo`. No se modifican los CSV originales de `data/`.

## Feature engineering

Las columnas originales de cliente se mantienen como variables base. La ablación añadió, por separado, calendario e historial para medir su aporte. Cada fila conserva como fecha `mes`; se probó sin calendario, con el entero `AAAAMM` o con `month_index` consecutivo —enero de 2026 = 0—. `id_cliente` se usa para agrupar al construir el historial y luego se elimina de las variantes principales.

Las cinco variables de historial se calculan ordenando cada cliente por mes, sin consultar `objetivo`:

| Feature | Cálculo hasta la fila actual |
| --- | --- |
| `n_observaciones_previas` | Número de filas anteriores del cliente; vale 0 en su primera observación. |
| `mes_primera_aparicion` | Primer mes observado para ese cliente, en formato `AAAAMM`. |
| `meses_desde_entrada` | Distancia entre el mes actual y el primero, medida en meses calendario. |
| `meses_en_riesgo` | `meses_desde_entrada + 1`; cuenta también el mes actual. |
| `cliente_recurrente` | Indicador de si existe al menos una observación anterior. |

La variante D usada como referencia combina las variables base, el mes y esas cinco variables de historial; D_month_index es la misma variante con el calendario codificado como índice consecutivo. La comparación también incluyó A (solo base), B (base + mes) y C (base + historial).

En un experimento diagnóstico se amplió D_month_index con nueve variables de `dias_ultima_interaccion`: lags de una y dos observaciones, diferencia actual menos lag 1, pendiente lineal por mes calendario, mínimo, máximo y media acumulados, indicador de reset y conteo acumulado de resets. Los lags cuentan observaciones previas —aunque haya huecos entre meses— y quedan ausentes si aún no hay suficientes observaciones. La pendiente queda ausente con una sola observación. Un reset indica que el valor actual cayó estrictamente respecto de la observación anterior. Las estadísticas acumuladas incluyen la fila actual. Pruebas de prefijo y de perturbación de meses futuros verifican que ninguna de estas features incorpora información futura.

## Experimentos realizados

### Ablation de calendario e historial

Se compararon LightGBM y CatBoost con cuatro variantes de features: A, base sin mes ni historial; B, base más mes; C, base más cinco variables de historial; y D, base más mes e historial. El mes se probó como entero `AAAAMM`, como índice consecutivo y ausente cuando correspondía. La configuración se fijó antes de evaluar los meses externos.

En el protocolo inicial con validación octubre–noviembre, LightGBM/history obtuvo media interna Gini de 0,2613 y LightGBM/clean 0,2594. La ventaja no fue concluyente: el IC bootstrap del delta fue [-0,0046, 0,0040]. Las puertas de seguridad para incluir el ID crudo no se superaron; por tanto, no se declaró un ganador concluyente. El detalle está en [`experiments/index.json`](experiments/index.json) y los manifiestos de cada corrida.

### Diagnósticos rolling

Sobre los cortes septiembre, octubre y noviembre se ejecutaron cuatro bloques:

1. **Bootstrap pareado por cliente:** 2.000 remuestreos con reemplazo de `id_cliente`, manteniendo juntas todas las filas del cliente. Se compararon LightGBM D, LightGBM B y CatBoost A sin reentrenar. Los intervalos mensuales y rolling son percentiles del 95%; Holm se aplicó a los tres contrastes rolling.
2. **Trayectoria de interacción:** se añadió a D el lag 1 y 2, el cambio respecto al lag 1, la pendiente lineal por mes, mínimo, máximo, media, indicador de reset y número acumulado de resets. Las features se calcularon por cliente hasta la fila actual. Se mantuvieron congelados los parámetros del baseline.
3. **Supervivencia explícita:** hazard discreto con enlace complementary log-log y efecto base mensual, y Cox start-stop con covariables variables por fila. Conversiones salen del conjunto en riesgo; la última fila sin conversión se censura. La hazard base de Cox se estimó solo con el entrenamiento de cada corte.
4. **Modelos foundation:** TabPFN-3, TabPFN-3.5 y TabFM con features D_month_index y defaults declarados. Cada modelo predijo todas las filas de validación en los tres cortes. TabPFN-3 y 3.5 se fijaron explícitamente; no se sustituyeron por otra versión. TabFM se ejecutó con contexto de 100 filas. Los folds y artefactos se guardaron para recuperación por fold.

## Resultados rolling

Gini por mes de validación y media simple. Los nombres y detalles metodológicos están documentados en [`experiments/diagnostics/results.md`](experiments/diagnostics/results.md).

| Modelo | Septiembre | Octubre | Noviembre | Media rolling |
| --- | ---: | ---: | ---: | ---: |
| CatBoost A | 0,247129 | 0,265077 | 0,235747 | 0,249318 |
| LightGBM B | 0,241934 | 0,270445 | 0,233684 | 0,248688 |
| LightGBM D | **0,243367** | 0,267735 | **0,242031** | **0,251044** |
| LightGBM D + trayectoria | 0,236283 | **0,273427** | 0,232271 | 0,247327 |
| Hazard cloglog | 0,229704 | 0,238825 | 0,201369 | 0,223299 |
| Cox start-stop | 0,231229 | 0,239150 | 0,201807 | 0,224062 |
| TabPFN-3 | 0,236503 | 0,243217 | 0,223520 | 0,234413 |
| TabPFN-3.5 | 0,163321 | 0,169806 | 0,137796 | 0,156975 |
| TabFM | 0,156399 | 0,146068 | 0,152641 | 0,151703 |

LightGBM D tuvo la media rolling más alta dentro de estos diagnósticos. Añadir la trayectoria no mejoró esa media. Cox y cloglog quedaron por debajo de D. Entre los modelos foundation, TabPFN-3 obtuvo la media más alta. Estas comparaciones son diagnósticas y no reemplazan el protocolo temporal acordado ni constituyen una selección estadística del modelo final.

En el bootstrap pareado, los tres IC del delta medio incluyen cero. Los p-valores con ajuste Holm fueron 1,0 para las tres comparaciones (D–B, D–A y B–A); no hay evidencia de diferencias en la media rolling al nivel del 5%. La variación entre meses reportada en la tabla es descriptiva, no inferencial.

Los tres modelos foundation completaron los tres folds: 9.900 filas de septiembre, 10.400 de octubre y 9.500 de noviembre por fold, 29.800 predicciones cada uno. Las ejecuciones remotas se hicieron en Colab con T4 después de que otros aceleradores no estuvieran disponibles; los hashes y estados están bajo [`experiments/diagnostics/04_foundation/`](experiments/diagnostics/04_foundation/). Los modelos y versiones se fijaron en sus configuraciones y manifiestos.

Los pesos y checkpoints, modelos serializados y cachés comprimidas de features permanecen en el workspace local y no se incluyen en el historial publicado. El repositorio conserva el código, los informes, las configuraciones, los manifiestos, las métricas y las predicciones tabulares. Para regenerar archivos locales se usan los comandos de ejecución descritos en el informe de diagnósticos.

## Estructura del repositorio

- `data/`: datos de entrenamiento, prueba, metadatos y formato de entrega.
- `src/datafest/`: pipeline, CLI, ablation, diagnósticos, supervivencia y modelos foundation.
- `experiments/`: corridas, predicciones, métricas, manifiestos e índices.
- `scripts/`: ejecución, recuperación de folds remotos y generación del informe.
- `tests/`: pruebas de causalidad temporal, predicciones por fila y recuperación de experimentos.
- `DATASET_DESCRIPTION.md`: diccionario y formato de la competencia.
- `NON_NEGOTIABLE.md`: reglas de evaluación que deben conservarse.

## Reproducir verificaciones

Con `uv` instalado:

```sh
uv sync
uv run pytest -q
```

Las corridas documentadas se pueden inspeccionar desde `experiments/diagnostics/results.md`; cada directorio de experimento contiene su configuración y manifiesto. Los comandos para reanudar diagnósticos o foundation están descritos en ese informe. Ejecutar un comando de foundation puede solicitar una GPU de Colab y consumir recursos; los folds ya completos se verifican y se reutilizan según el estado y los hashes.

Las credenciales, incluida `TABPFN_TOKEN`, se configuran localmente en el entorno o en `.env`, que está excluido de Git. Nunca deben guardarse en los artefactos versionados.
