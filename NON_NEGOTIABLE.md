# NON-NEGOTIABLE

## Propósito

Este proyecto participa en una competencia de propensión de conversión de clientes. Su propósito es ordenar a los clientes según la probabilidad de que realicen su primera conversión en un mes determinado.

## Objetivo de predicción

Predecir `objetivo`, una etiqueta binaria por observación cliente-mes. `1` indica primera conversión del cliente durante ese mes; `0` indica que no convirtió. Un cliente puede aparecer en varios meses y, tras convertir, no vuelve a aparecer en meses posteriores.

El entrenamiento disponible abarca enero–noviembre de 2026. La prueba corresponde a diciembre de 2026 y no incluye la etiqueta `objetivo`.

## Reglas congeladas

- Respetar el orden temporal: no entrenar con información de meses posteriores al período que se predice ni usar etiquetas de prueba.
- Comparar modelos con Gini, calculado como `2 × ROC AUC − 1`; un valor mayor es mejor.
- Usar como validación principal el corte temporal acordado: entrenar con enero–septiembre de 2026 y validar con octubre–noviembre de 2026. No sustituir esta comparación por una partición aleatoria.
- Ajustar configuraciones solo en cortes walk-forward internos hasta septiembre; no usar el holdout octubre–noviembre para búsqueda ni early stopping.
- Comparar variantes sin `id_cliente`, con historial causal y con ID crudo. Las variables de historial solo pueden usar observaciones del mismo mes y meses anteriores; no usar `objetivo` para construirlas.
- Tratar el ID crudo como diagnóstico. Solo puede entrar a búsqueda o ser candidato ganador después de superar los cortes temporales internos y un corte con grupos de clientes separados.
- Registrar por corrida los hashes SHA-256 de fuentes, split, features, modelo y salidas en un manifiesto JSON.
- Generar una probabilidad entre `0` y `1` para cada fila de prueba.
- La entrega debe contener exactamente `id_cliente,prediccion`, preservar el `id_cliente` y el orden de filas de `data/test.csv`, y omitir `mes`.
- Tratar `data/` como la ubicación de los archivos de competencia. No cambiar sus datos fuente al desarrollar experimentos.

## Datos disponibles

- `data/train.csv`: predictores y `objetivo`.
- `data/test.csv`: predictores de diciembre, sin `objetivo`.
- `data/sample_submission.csv`: ejemplo de formato de entrega; sus valores son de muestra.
- `data/metaData.csv`: descripciones, tipos y notas de las columnas.
- `DATASET_DESCRIPTION.md`: descripción del conjunto y de las reglas de entrega.
- `experiments/index.json`: índice de las corridas; cada corrida conserva métricas, submission y `manifest.json`.

En la inspección inicial, entrenamiento contiene 110.100 filas de enero a noviembre y prueba 9.900 filas de diciembre. No se encontraron valores faltantes ni claves duplicadas de `id_cliente` y `mes`. Los identificadores de cliente se repiten entre meses, por lo que la observación y la unidad de evaluación son cliente-mes.
