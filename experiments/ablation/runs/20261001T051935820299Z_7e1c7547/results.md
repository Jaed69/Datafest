# Ablación rolling de mes e historial

Lote: `20261001T051935820299Z_7e1c7547`

## Protocolo

- A y C se ejecutan sin mes; B y D usan AAAAMM o `month_index` (enero=0 a noviembre=10). Todas excluyen `id_cliente`.
- Rolling principal: ≤Ago→Sep, ≤Sep→Oct y ≤Oct→Nov. Se informa Gini mensual, media simple y desviación estándar poblacional (`ddof=0`).
- Configuración baseline común por algoritmo; iteraciones elegidas en folds internos hasta agosto con A y C. Sin selección ni early stopping en septiembre-noviembre.
- Iteraciones congeladas: catboost: 192, lightgbm: 58.
- Frozen: modelo entrenado hasta septiembre para octubre y noviembre. Pooled es secundaria.

## Rolling principal

| Modelo | Variante | Mes | Gini Sep | Gini Oct | Gini Nov | Media | Std. |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| catboost | A | none | 0.24713 | 0.26508 | 0.23575 | 0.24932 | 0.01207 |
| catboost | B_absolute | absolute | 0.24633 | 0.26756 | 0.23456 | 0.24948 | 0.01365 |
| catboost | B_month_index | month_index | 0.24419 | 0.26171 | 0.23373 | 0.24655 | 0.01154 |
| catboost | C | none | 0.22701 | 0.26520 | 0.23372 | 0.24197 | 0.01665 |
| catboost | D_absolute | absolute | 0.24091 | 0.26499 | 0.23891 | 0.24827 | 0.01185 |
| catboost | D_month_index | month_index | 0.23128 | 0.26544 | 0.23840 | 0.24504 | 0.01471 |
| lightgbm | A | none | 0.23931 | 0.26290 | 0.23509 | 0.24577 | 0.01223 |
| lightgbm | B_absolute | absolute | 0.24193 | 0.27045 | 0.23368 | 0.24869 | 0.01575 |
| lightgbm | B_month_index | month_index | 0.24193 | 0.27045 | 0.23368 | 0.24869 | 0.01575 |
| lightgbm | C | none | 0.23867 | 0.26412 | 0.23158 | 0.24479 | 0.01397 |
| lightgbm | D_absolute | absolute | 0.24337 | 0.26773 | 0.24203 | 0.25104 | 0.01181 |
| lightgbm | D_month_index | month_index | 0.24337 | 0.26773 | 0.24203 | 0.25104 | 0.01181 |

## Frozen hasta septiembre

| Modelo | Variante | Gini Oct | Gini Nov | Media | Std. | Pooled secundario |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| catboost | A | 0.26508 | 0.23614 | 0.25061 | 0.01447 | 0.25138 |
| catboost | B_absolute | 0.26756 | 0.23045 | 0.24900 | 0.01855 | 0.25009 |
| catboost | B_month_index | 0.26171 | 0.23601 | 0.24886 | 0.01285 | 0.24962 |
| catboost | C | 0.26520 | 0.23654 | 0.25087 | 0.01433 | 0.25166 |
| catboost | D_absolute | 0.26499 | 0.23267 | 0.24883 | 0.01616 | 0.25012 |
| catboost | D_month_index | 0.26544 | 0.22870 | 0.24707 | 0.01837 | 0.24808 |
| lightgbm | A | 0.26290 | 0.23402 | 0.24846 | 0.01444 | 0.24939 |
| lightgbm | B_absolute | 0.27045 | 0.22457 | 0.24751 | 0.02294 | 0.24886 |
| lightgbm | B_month_index | 0.27045 | 0.22457 | 0.24751 | 0.02294 | 0.24886 |
| lightgbm | C | 0.26412 | 0.23069 | 0.24740 | 0.01671 | 0.24855 |
| lightgbm | D_absolute | 0.26773 | 0.22362 | 0.24567 | 0.02206 | 0.24708 |
| lightgbm | D_month_index | 0.26773 | 0.22362 | 0.24567 | 0.02206 | 0.24708 |
