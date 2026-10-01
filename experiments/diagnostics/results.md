# Cuatro experimentos rolling

Cortes: ≤agosto→septiembre, ≤septiembre→octubre, ≤octubre→noviembre de 2026. Misma población y orden de validación del baseline; ninguna selección de parámetros con septiembre, octubre o noviembre. D y B son LightGBM con month_index; A es CatBoost sin mes.

Los bloques son diagnósticos. Las métricas principales son Gini mensual y media simple rolling. La desviación entre meses describe variación temporal; no es un intervalo de confianza.

## Gini rolling

| Modelo | Sep | Oct | Nov | Media | Desv. temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost_A | 0.247129 | 0.265077 | 0.235747 | 0.249318 | 0.012074 |
| LightGBM_B | 0.241934 | 0.270445 | 0.233684 | 0.248688 | 0.015749 |
| LightGBM_D | 0.243367 | 0.267735 | 0.242031 | 0.251044 | 0.011815 |
| LightGBM_D + trayectoria | 0.236283 | 0.273427 | 0.232271 | 0.247327 | 0.018528 |
| cloglog | 0.229704 | 0.238825 | 0.201369 | 0.223299 | 0.015948 |
| cox | 0.231229 | 0.239150 | 0.201807 | 0.224062 | 0.016065 |
| tabpfn3 | 0.236503 | 0.243217 | 0.223520 | 0.234413 | 0.008176 |
| tabpfn35 | 0.163321 | 0.169806 | 0.137796 | 0.156975 | 0.013817 |
| tabfm | 0.156399 | 0.146068 | 0.152641 | 0.151703 | 0.004269 |

La trayectoria no mejora la media de D en esta corrida. No se interpreta esta diferencia descriptiva como significación estadística.

## 1. Bootstrap pareado por cliente

2.000 remuestreos de clientes con reemplazo, semilla 42. Un único vector de multiplicidades por réplica conserva juntas todas las filas de cada cliente en los tres meses y los tres modelos. Se usa AUC ponderada exacta con empates. No hay reentrenamiento.

IC percentil 95% sin ajuste. Los p-valores bilaterales usan el bootstrap centrado bajo delta cero con corrección +1. Holm se aplica a la familia de tres comparaciones de **media rolling**; los IC mensuales y rolling no son simultáneos.

| Par (primero − segundo) | Mes | Δ Gini | IC 95% | p | p Holm |
| --- | --- | ---: | --- | ---: | ---: |
| LightGBM_D - LightGBM_B | 202609 | 0.001433 | [-0.006451, 0.009293] | — | — |
| LightGBM_D - LightGBM_B | 202610 | -0.002711 | [-0.010079, 0.004453] | — | — |
| LightGBM_D - LightGBM_B | 202611 | 0.008346 | [-0.002288, 0.018988] | — | — |
| LightGBM_D - LightGBM_B | rolling_mean | 0.002356 | [-0.003077, 0.007665] | 0.36881559220389803 | 1.0 |
| LightGBM_D - CatBoost_A | 202609 | -0.003761 | [-0.018930, 0.011177] | — | — |
| LightGBM_D - CatBoost_A | 202610 | 0.002657 | [-0.011559, 0.016600] | — | — |
| LightGBM_D - CatBoost_A | 202611 | 0.006284 | [-0.010448, 0.023933] | — | — |
| LightGBM_D - CatBoost_A | rolling_mean | 0.001727 | [-0.007126, 0.010516] | 0.7361319340329835 | 1.0 |
| LightGBM_B - CatBoost_A | 202609 | -0.005195 | [-0.020808, 0.010121] | — | — |
| LightGBM_B - CatBoost_A | 202610 | 0.005368 | [-0.007567, 0.018679] | — | — |
| LightGBM_B - CatBoost_A | 202611 | -0.002063 | [-0.017869, 0.015032] | — | — |
| LightGBM_B - CatBoost_A | rolling_mean | -0.000630 | [-0.009395, 0.008269] | 0.8900549725137431 | 1.0 |

Los tres IC de delta medio incluyen cero y ninguno rechaza igualdad tras Holm al 5%.

## 2. Trayectoria de interacción

D_month_index más nueve features de dias_ultima_interaccion: lag1, lag2, delta, slope lineal, mínimo, máximo, media, reset actual y resets acumulados. Los lags son observaciones previas, aunque haya huecos de meses. Slope usa meses calendario y todo el prefijo hasta la fila actual; queda ausente con una sola observación. Reset significa caída estricta frente al valor observado previo. Las estadísticas incluyen la fila actual y no usan objetivo.

LightGBM conserva los parámetros y las 58 iteraciones del baseline, sin early stopping ni búsqueda. Se guardan modelo, mapas de categorías entrenados en cada corte, features, predicciones y métricas.

## 3. Supervivencia explícita

Cada observación ocupa (tenure−1, tenure], donde tenure es el número de meses calendario desde la primera observación. Las conversiones salen al terminar el intervalo. La última observación sin evento se trata como censura derecha. Los meses sin observación no reciben covariables inventadas y no añaden intervalos al conjunto de riesgo.

Cloglog tiene efecto base categórico del mes **en riesgo** y el month_index calendario de D. Cox usa start-stop, covariables de cada fila y empates Breslow. Ambos usan covariables D con exclusión de tres alias temporales deterministas (mes_primera_aparicion, meses_en_riesgo, meses_desde_entrada). Codificación y escalado se ajustan con entrenamiento solamente. Penalizaciones numéricas fijas: cloglog 1e−6 y Cox 1e−4; sin selección temporal.

Las probabilidades Cox son 1−exp(−incremento_base_mensual × exp(Xβ)). La hazard base procede exclusivamente del entrenamiento del corte. Para tenure posterior al máximo entrenado se prolonga el último incremento mensual; cloglog prolonga el último factor base. Esta extrapolación explícita es un supuesto, no una estimación con el mes de validación. Los gradientes máximos al terminar quedaron debajo de 1e−6 en ambos modelos y los tres folds.

## 4. Foundation: avance y límites

TabPFN se fija con ModelVersion.V3 y se comprueba el nombre de su checkpoint: no se sustituye por otra versión. Por ampliación explícita del usuario se evalúa además TabPFN-3.5, con ModelVersion.V3_5, defaults publicados y artefactos separados bajo tabpfn35. [Prior Labs](https://docs.priorlabs.ai/models) publica hasta un millón de filas para TabPFN-3. TabFM usa el backend PyTorch v1.0.0, con código fijado a fbb665569425fd2f490c6576b3af967876fe11ff.

El [README de TabFM](https://github.com/google-research/tabfm) declara 100 filas de contexto, pero ese commit tiene max_num_rows=None en el constructor. Se fija explícitamente max_num_rows=100 para conservar el valor publicado acordado; el resto conserva defaults. No se muestrean filas de validación. Se guardan parámetros efectivos y los índices exactos de los contextos por miembro.

| Modelo | Fold | Estado | Filas | Gini | Motivo de fallo |
| --- | --- | --- | ---: | ---: | --- |
| tabpfn3 | 202609 | complete | 9900 | 0.236503 | — |
| tabpfn3 | 202610 | complete | 10400 | 0.243217 | — |
| tabpfn3 | 202611 | complete | 9500 | 0.223520 | — |
| tabpfn35 | 202609 | complete | 9900 | 0.163321 | — |
| tabpfn35 | 202610 | complete | 10400 | 0.169806 | — |
| tabpfn35 | 202611 | complete | 9500 | 0.137796 | — |
| tabfm | 202609 | complete | 9900 | 0.156399 | — |
| tabfm | 202610 | complete | 10400 | 0.146068 | — |
| tabfm | 202611 | complete | 9500 | 0.152641 | — |

Se aceptaron las licencias de TabPFN-3 y 3.5 y la API key de Prior Labs se leyó localmente desde .env (ignorado por Git). La GPU local GTX 1650 (4 GB) no pudo completar TabPFN-3.5 por falta de memoria; los tres folds se completaron en Colab con T4. El intento local de TabFM terminó con código 137 durante carga de pesos (falta de memoria inferida; sin acceso al log del kernel). Colab rechazó L4 y A100 antes de asignar T4. Los estados y logs registran cada ejecución remota.

Colab retiró sesiones durante preparación y recuperación. Se conservan los folds completos antes de continuar en otra VM. La transferencia del checkpoint de noviembre de TabPFN-3 se interrumpió; se recuperó la copia de septiembre después de comprobar igualdad exacta del SHA-256 esperado. Para TabPFN-3.5, el checkpoint oficial es común a los folds y se reutiliza solo tras verificar tamaño y SHA-256. Sus manifiestos locales conservan checkpoint, parámetros, predicciones, métricas y estado; se omite el pickle redundante fitted_estimator.pkl, que replica el checkpoint y el estado interno del estimador.

La compresión gzip de las mismas features tuvo dos timestamps entre VMs. feature_archive conserva ambas representaciones originales; audit.json certifica que el CSV descomprimido es idéntico. El manifiesto de septiembre apunta a su archivo inmutable. No cambió ningún valor de entrada.

## Reanudar y verificar

```sh
.venv/bin/datafest diagnostics
.foundation-venv/bin/python scripts/run_tabpfn_local.py --model tabpfn3
.foundation-venv/bin/python scripts/run_tabpfn_local.py --model tabpfn35
.venv/bin/python scripts/run_foundation_colab.py --model tabfm --gpu T4
# Tras configurar TABPFN_TOKEN localmente:
.venv/bin/python scripts/run_foundation_colab.py --model tabpfn3 --gpu T4
.venv/bin/python scripts/run_foundation_colab.py --model tabpfn35 --gpu T4
.venv/bin/python scripts/report_diagnostics.py
```

foundation-requirements.txt fija las dependencias del entorno separado. Cada fold conserva checkpoint, parámetros, predicciones, métricas y estado con hashes. TabPFN-3.5 usa el límite publicado predeterminado de 1.000.000 de filas de entrenamiento, sin submuestreo externo, y predice todas las filas de validación. Al reanudar se verifica identidad de datos, features, código y versiones, además de la integridad de los artefactos; un fold incompleto no se reutiliza. La migración envía un snapshot de Git sin modificar la rama o el índice del usuario, recupera resultados al cerrar cada fold y libera su sesión de Colab.

Verificación: pruebas de invariancia por prefijos y perturbaciones futuras, lags/huecos/resets, probabilidades por fila, AUC ponderada contra clientes materializados, gradientes de ambas likelihoods, pertenencia start-stop y recuperación de folds completos/corruptos. Las predicciones completas deben cubrir 9.900, 10.400 y 9.500 filas; 29.800 en total.
