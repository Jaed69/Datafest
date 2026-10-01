# Diseño: ablación rolling de mes e historial

## Objetivo

Separar el aporte del calendario y del historial causal a la predicción de conversión. La métrica principal será el Gini de cada mes de validación en cortes rolling; el holdout pooled de octubre y noviembre no decidirá el ganador.

## Matriz de experimentos

Se comparan LightGBM y CatBoost con una matriz factorial de cuatro conjuntos de features y tres tratamientos del mes (24 combinaciones):

| Variante | Features |
| --- | --- |
| A | Base, sin `id_cliente`, sin mes ni historial |
| B | Base + mes, sin `id_cliente` ni historial |
| C | Base + las cinco features de historial, sin mes ni `id_cliente` |
| D | Base + mes + las cinco features de historial, sin `id_cliente` |

Tratamientos de mes: AAAAMM entero; `month_index` entero consecutivo, enero=0 a noviembre=10; y ausente. El tratamiento ausente hace que A y C se repitan entre esos niveles; se ejecuta una sola vez para evitar duplicados, por lo que el total efectivo es 12 combinaciones (seis por modelo: A/C sin mes una vez, B/D para AAAAMM y `month_index`).

Las variables de historial son `n_observaciones_previas`, `mes_primera_aparicion`, `meses_desde_entrada`, `meses_en_riesgo` y `cliente_recurrente`. Se calculan solo con la secuencia observable del cliente hasta el mes de la fila, sin usar `objetivo`.

## Evaluación y selección

Los cortes principales son entrenar con datos hasta agosto y evaluar septiembre; hasta septiembre y evaluar octubre; hasta octubre y evaluar noviembre. Se informa Gini Sep, Oct y Nov, media aritmética de los tres y desviación estándar poblacional (`ddof=0`). Se incluyen resultados por corte aunque un tratamiento de mes tenga combinaciones repetidas deduplicadas.

Las configuraciones se seleccionan para cada combinación modelo/variante con los folds walk-forward internos cuya validación termina a más tardar en agosto. Así, la configuración y las iteraciones quedan fijadas antes de puntuar septiembre, octubre o noviembre en los cortes externos. La cantidad de iteraciones se fija con la mediana de best iterations internos del candidato seleccionado. No hay búsqueda ni early stopping en los cortes externos.

Se mantiene una evaluación frozen por cada combinación: entrenar hasta septiembre y predecir octubre y noviembre con el mismo modelo. Se reporta Gini de octubre, Gini de noviembre, media y desviación estándar, además del pooled como dato secundario. No participa en la decisión principal.

## Artefactos

El lote registra las features, codificación del calendario, configuración congelada, iteraciones, predicciones por fila, métricas por mes, media/desviación, resultados frozen y hashes SHA-256 de fuentes, split, código y salidas. Se guarda un resumen comparativo legible y se añade el lote al índice de experimentos sin reemplazar corridas previas.

## Criterios de aceptación

- Todas las variantes excluyen `id_cliente`.
- Los tres cortes rolling respetan estrictamente el orden temporal.
- Octubre y noviembre no afectan selección de configuración ni early stopping.
- La tabla principal contiene Gini por mes, media y desviación estándar; pooled está separado.
- La evaluación frozen se presenta como análisis de robustez y no como métrica principal.
- Los manifiestos permiten verificar la integridad de los artefactos.
