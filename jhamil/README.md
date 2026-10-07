# Experimentación de Jhamil: de la data cruda a la submission final

Análisis independiente que parte de la data original (`Data_inicial/`) y vuelve a validar las decisiones del proyecto. Conclusión: **se envía H4b2+N1**, y ninguna de las variantes probadas aquí la supera.

Presentación para el equipo: https://claude.ai/artifact/VR6mRzPm7LoNcaHX51hnEb

## Archivo de envío

`submission/submission_h4b2_n1.csv`, la copia exacta de `experiments/final/20261006T185409831403Z_a714ebcd_ensemble-h4b2-n1/submission.csv`.

| Verificación | Resultado |
| --- | --- |
| Columnas | `id_cliente,prediccion`, iguales a `sample_submission.csv` |
| Filas | 9.900 (igual que `test.csv`) |
| Orden de `id_cliente` | Idéntico a `test.csv` y `sample_submission.csv` |
| IDs únicos, sin vacíos | Sí |
| Rango de `prediccion` | 0,0293 a 0,4783, todo en [0, 1] |
| SHA-256 | `bfdfd1a0584f9dcd9c0530f4c144755dca621de5a8e6b495a3c47d4acc0751ba` (igual al original) |

Respaldos en orden: H4b2 y, después, LightGBM D.

## Hallazgos, paso a paso

### 1. Estructura de la data (`eda.py`, `eda2.py`)

- `data/` y `Data_inicial/` son idénticas byte a byte.
- Es un panel hasta el evento: cada cliente aparece todos los meses hasta su primera conversión. Ninguno desaparece sin convertir (16.567 convirtieron y 8.061 seguían activos en noviembre).
- Test = 8.061 sobrevivientes de noviembre + 1.839 clientes nuevos (IDs 24629–26467).
- 21 de las 22 variables son fijas por cliente. La tasa de conversión mensual es estable, entre 14% y 16%.

### 2. `dias_ultima_interaccion` es ruido en diciembre (`eda3.py`, `h_dui.py`)

- Cada mes es una copia exacta de `dias_ultima_transaccion` con probabilidad (12 − m)/11: 100% en enero, 9,4% en noviembre y 0,3% en diciembre.
- Cuando no es copia, es ruido sin relación con el objetivo (Gini 0,01).
- En la validación rolling de junio a noviembre, quitarla nunca perdió en promedio (DROP 0,2536, RAW 0,2521, RANK 0,2521). La diferencia está dentro del ruido.

### 3. Las variables son independientes entre sí (`logic.py`)

- Ningún par numérico supera |Spearman| 0,1 y ningún par categórico supera V de Cramér 0,05. Los datos son sintéticos y no hay lógica de negocio que explotar.

### 4. La señal sigue reglas por banda de riesgo (`thr.py`, `rules.py`, `resid.py`, `rules2.py`)

- **LOW:** base ~13%; +12 puntos con 3 o más productos; +12 puntos con app móvil, tarjeta y 2 o más productos; una deuda ≥ 0,6 la baja.
- **HIGH:** depende de la recencia de la última transacción: menos de 100 días ~15,5%, de 100 a 180 ~12,5%, 180 o más ~5%.
- **MEDIUM:** casi plana.
- Los umbrales aparecen igual usando solo enero a agosto.

### 5. Métodos bayesianos (`bayes.py`)

En junio a noviembre, todos quedaron por debajo de LightGBM (0,2543): Naive Bayes global 0,2296, Naive Bayes por banda 0,2415, logística bayesiana por banda 0,2407 y suavizado bayesiano de reglas 0,2459.

### 6. Techo (`ceiling.py`)

- La curva de aprendizaje todavía sube, pero cada vez menos.
- Si las probabilidades del modelo fueran la verdad, el Gini esperado sería ~0,25–0,26, con ±0,015 de ruido por mes.

### 7. Validación limpia septiembre–noviembre (`battery.py`, `neural_rules.py`, `evaluate.py`)

Candidatos fijados antes de mirar los resultados, 2000 réplicas de bootstrap pareado y corrección de Holm. La referencia es la submission actual (0,25432, reproducida exacta).

| Modelo | Gini medio | vs actual |
| --- | ---: | ---: |
| H4b2+N1 (actual) | 0,2543 | — |
| Actual + reglas y residuos | 0,2541 | −0,0003 (p Holm 0,91) |
| LightGBM sin `dias_ultima_interaccion` | 0,2489 | −0,0054 |
| LightGBM con reglas | 0,2456 | −0,0087 |
| XGBoost con reglas | 0,2448 | −0,0095 |
| CatBoost con reglas (configuración propia) | 0,2301 | −0,0242 |

Agregar las reglas como variables nunca ayudó a los árboles. La corrida de RealMLP con reglas se cortó a pedido, por su costo (17 a 30 minutos por ajuste).

### 8. Rolling origin y validación cruzada (`walk.py`)

- Con ventana acumulada se predice mejor que usando solo los últimos 3 meses: en noviembre, 0,235 contra 0,222.
- Mezclar meses infla el puntaje: KFold aleatorio da 0,262 y GroupKFold por cliente 0,263, contra 0,249 de la validación temporal.

### 9. Regresión logística con rolling origin (`logit_ro.py`)

| Modelo (sep–nov) | Gini medio | Desvío | vs LightGBM (IC95) |
| --- | ---: | ---: | --- |
| LightGBM | 0,2496 | 0,0140 | — |
| Logística con 12 reglas | 0,2457 | 0,0094 | −0,004 [−0,016; +0,007], empate |
| Logística por banda | 0,2345 | 0,0143 | −0,015 |
| Logística simple | 0,2252 | 0,0202 | −0,024 |
| Logística con tramos | 0,2216 | 0,0221 | −0,028 |

Discretizar empeora a la logística. Lo que la mejora es la interacción con la banda.

## Advertencias

- Los umbrales de las reglas se descubrieron mirando todos los meses. Se verificó que aparecen con enero a agosto, pero los resultados de las reglas en los meses tempranos son optimistas.
- La configuración de CatBoost de `battery.py` rinde peor que CatBoost A del repo, así que su valor absoluto no es comparable.
- La sección 5 (métodos bayesianos) y la tabla de `dias_ultima_interaccion` usan la ventana junio–noviembre. No se comparan directamente con las tablas de septiembre a noviembre.

## Estructura

```
jhamil/
  scripts/      análisis y experimentos (Python)
  results/      predicciones fuera de muestra y tablas de resultados (CSV)
  charts/       gráficos de la presentación (PNG)
  submission/   archivo de envío verificado
```

## Cómo reproducir

Desde `jhamil/results/`, cada script recibe la carpeta de la data cruda y escribe sus salidas en el directorio actual:

```bash
cd jhamil/results
uv run --project ../.. python ../scripts/walk.py ../../Data_inicial
uv run --project ../.. python ../scripts/logit_ro.py ../../Data_inicial
uv run --project ../.. python ../scripts/charts.py ../../Data_inicial
```

`evaluate.py` recibe el OOF del repo, el de la batería, el neuronal y la cantidad de réplicas:

```bash
uv run --project ../.. python ../scripts/evaluate.py ../../experiments/ensembles/20261006T173753810737Z_a714ebcd_ensemble-search/joined_oof.csv battery_oof.csv neural_oof.csv 2000
```

`neural_rules.py` necesita el entorno aparte de `scripts/neural/requirements_frozen.txt` (torch con CUDA y pytabkit).
