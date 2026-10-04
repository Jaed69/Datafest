# NVIDIA Kumo Tabular — 20261003_kumo_colab_t4_rolling

## Executive summary

Mejor Kumo: **small; E=1; context=60000; uniform, variante A**, mean rolling Gini **0.24730** (Δ vs LightGBM D: **-0.00374**). Sep/Oct/Nov: 0.23938/0.25848/0.24403; std temporal 0.00813.
Mejor Kumo-D para diversidad: **small; E=1; context=60000; uniform**, mean Gini 0.24229; correlaciones por fold: 202609 Pearson 0.945/Spearman 0.828, 202610 Pearson 0.950/Spearman 0.829, 202611 Pearson 0.943/Spearman 0.789.
Correlaciones de la mejor variante global (A, small; E=1; context=60000; uniform) con LightGBM D: 202609 Pearson 0.957/Spearman 0.854, 202610 Pearson 0.957/Spearman 0.848, 202611 Pearson 0.951/Spearman 0.812.
Conclusión operativa: **ensemble-only (best fixed blend is exploratory; paired bootstrap required)**.
Costo: 7.1 min de inferencia acumulada, pico VRAM observado 2.76 GiB; hardware {'platform': 'Linux-6.6.122+-x86_64-with-glibc2.39', 'processor': 'x86_64', 'cpu_count': 2, 'gpu': 'Tesla T4', 'gpu_total_gib': 14.563}.
Mejor blend evaluado: variante A, small; E=1; context=60000; uniform con probability, 50% LGBM / 50% Kumo, mean Gini 0.25423 (Δ vs LGBM-D +0.00319).

## Protocolo y API

Rolling: ≤202608→202609, ≤202609→202610, ≤202610→202611. Gini=2×ROC AUC−1; media simple y desviación poblacional. No se usaron splits aleatorios ni diciembre.
Se usó `sdm.models.KumoTabular(task='classification', size=..., device=...)`, `fit` con etiquetas solo del train y `predict` por lotes. Salida softmax multiclase; la columna `1` es P(clase positiva) y se usa continua, sin threshold. No se garantiza calibración.
Features categóricas y booleanas se pasan con el tipo inferido desde train; el recipe oficial las alinea, convierte a numéricas y aplica sus transformaciones. `mes` se adapta al `month_index` de D existente (enero=0). Los tipos/recipe se ajustan solo con contexto de entrenamiento.

## Tabla principal

| Modelo | Variante | Config | Sep | Oct | Nov | Mean | Std | Δ vs LGBM-D |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | D | existing rolling benchmark | nan | nan | nan | 0.25104 | nan | +0.00000 |
| CatBoost | A | existing rolling benchmark | nan | nan | nan | 0.24932 | nan | -0.00172 |
| CatBoost | B AAAAMM | existing rolling benchmark | nan | nan | nan | 0.24948 | nan | -0.00156 |
| LightGBM | B | existing rolling benchmark | nan | nan | nan | 0.24869 | nan | -0.00235 |
| KumoTabular | A | small; E=1; context=60000; uniform | 0.23938 | 0.25848 | 0.24403 | 0.24730 | 0.00813 | -0.00374 |
| KumoTabular | D | small; E=1; context=60000; uniform | 0.23580 | 0.25087 | 0.24022 | 0.24229 | 0.00633 | -0.00875 |
| KumoTabular | D | small; E=1; context=8192; uniform | 0.22065 | 0.22940 | 0.21674 | 0.22226 | 0.00529 | -0.02878 |
| KumoTabular | D | small; E=1; context=8192; recency-aware | 0.22358 | 0.24656 | 0.21185 | 0.22733 | 0.01441 | -0.02371 |
| KumoTabular | D | small; E=1; context=8192; balanced-by-month | 0.21893 | 0.24495 | 0.19503 | 0.21964 | 0.02038 | -0.03140 |
| KumoTabular | D | small; E=4; context=8192; uniform | 0.23202 | 0.24812 | 0.21327 | 0.23114 | 0.01424 | -0.01990 |
| KumoTabular | D | small; E=8; context=8192; uniform | 0.23212 | 0.25143 | 0.21720 | 0.23358 | 0.01401 | -0.01746 |
| KumoTabular | D | medium; E=1; context=60000; uniform | 0.19802 | 0.17652 | 0.16290 | 0.17914 | 0.01446 | -0.07190 |
| KumoTabular | D | large; E=1; context=60000; uniform | 0.21381 | 0.17507 | 0.19267 | 0.19385 | 0.01584 | -0.05719 |

## ROC AUC y Average Precision por mes

| Config | ROC AUC Sep | ROC AUC Oct | ROC AUC Nov | AP Sep | AP Oct | AP Nov |
|---|---:|---:|---:|---:|---:|---:|
| small; E=1; context=60000; uniform / A | 0.61969 | 0.62924 | 0.62202 | 0.22545 | 0.25115 | 0.22814 |
| small; E=1; context=60000; uniform / D | 0.61790 | 0.62544 | 0.62011 | 0.22491 | 0.25013 | 0.22892 |
| small; E=1; context=8192; uniform / D | 0.61033 | 0.61470 | 0.60837 | 0.21563 | 0.24036 | 0.22369 |
| small; E=1; context=8192; recency-aware / D | 0.61179 | 0.62328 | 0.60592 | 0.21242 | 0.24745 | 0.21388 |
| small; E=1; context=8192; balanced-by-month / D | 0.60947 | 0.62247 | 0.59752 | 0.22259 | 0.24570 | 0.21791 |
| small; E=4; context=8192; uniform / D | 0.61601 | 0.62406 | 0.60664 | 0.21900 | 0.24649 | 0.22220 |
| small; E=8; context=8192; uniform / D | 0.61606 | 0.62572 | 0.60860 | 0.22150 | 0.24850 | 0.22607 |
| medium; E=1; context=60000; uniform / D | 0.59901 | 0.58826 | 0.58145 | 0.20119 | 0.20668 | 0.19011 |
| large; E=1; context=60000; uniform / D | 0.60691 | 0.58754 | 0.59633 | 0.20629 | 0.21579 | 0.20794 |

## Comparación con benchmarks

Los cuatro valores históricos son rolling y se copian sin recalcularlos. No se mezclan métricas frozen ni pooled.

## Ensemble

| Variante | Config Kumo | Método | LGBM | Kumo | Sep | Oct | Nov | Mean | Std |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | small; E=1; context=60000; uniform | probability | 0.75 | 0.25 | 0.24517 | 0.26807 | 0.24718 | 0.25347 | 0.01036 |
| A | small; E=1; context=60000; uniform | rank | 0.75 | 0.25 | 0.24413 | 0.26829 | 0.24594 | 0.25278 | 0.01099 |
| A | small; E=1; context=60000; uniform | probability | 0.50 | 0.50 | 0.24531 | 0.26724 | 0.25014 | 0.25423 | 0.00941 |
| A | small; E=1; context=60000; uniform | rank | 0.50 | 0.50 | 0.24387 | 0.26702 | 0.24879 | 0.25323 | 0.00996 |
| A | small; E=1; context=60000; uniform | probability | 0.25 | 0.75 | 0.24400 | 0.26416 | 0.24937 | 0.25251 | 0.00852 |
| A | small; E=1; context=60000; uniform | rank | 0.25 | 0.75 | 0.24225 | 0.26378 | 0.24811 | 0.25138 | 0.00909 |
| D | small; E=1; context=60000; uniform | probability | 0.75 | 0.25 | 0.24476 | 0.26652 | 0.24701 | 0.25276 | 0.00977 |
| D | small; E=1; context=60000; uniform | rank | 0.75 | 0.25 | 0.24362 | 0.26654 | 0.24552 | 0.25189 | 0.01038 |
| D | small; E=1; context=60000; uniform | probability | 0.50 | 0.50 | 0.24403 | 0.26362 | 0.24910 | 0.25225 | 0.00830 |
| D | small; E=1; context=60000; uniform | rank | 0.50 | 0.50 | 0.24275 | 0.26305 | 0.24672 | 0.25084 | 0.00878 |
| D | small; E=1; context=60000; uniform | probability | 0.25 | 0.75 | 0.24140 | 0.25860 | 0.24736 | 0.24912 | 0.00713 |
| D | small; E=1; context=60000; uniform | rank | 0.25 | 0.75 | 0.23993 | 0.25758 | 0.24477 | 0.24743 | 0.00745 |
| D | small; E=8; context=8192; uniform | probability | 0.75 | 0.25 | 0.24310 | 0.26812 | 0.24015 | 0.25046 | 0.01255 |
| D | small; E=8; context=8192; uniform | rank | 0.75 | 0.25 | 0.24397 | 0.26783 | 0.23945 | 0.25042 | 0.01245 |
| D | small; E=8; context=8192; uniform | probability | 0.50 | 0.50 | 0.24147 | 0.26535 | 0.23520 | 0.24734 | 0.01299 |
| D | small; E=8; context=8192; uniform | rank | 0.50 | 0.50 | 0.24242 | 0.26506 | 0.23417 | 0.24722 | 0.01306 |
| D | small; E=8; context=8192; uniform | probability | 0.25 | 0.75 | 0.23879 | 0.25971 | 0.22755 | 0.24201 | 0.01333 |
| D | small; E=8; context=8192; uniform | rank | 0.25 | 0.75 | 0.23821 | 0.25908 | 0.22598 | 0.24109 | 0.01367 |
| D | small; E=4; context=8192; uniform | probability | 0.75 | 0.25 | 0.24421 | 0.26682 | 0.23828 | 0.24977 | 0.01230 |
| D | small; E=4; context=8192; uniform | rank | 0.75 | 0.25 | 0.24453 | 0.26664 | 0.23894 | 0.25004 | 0.01196 |
| D | small; E=4; context=8192; uniform | probability | 0.50 | 0.50 | 0.24294 | 0.26330 | 0.23332 | 0.24652 | 0.01250 |
| D | small; E=4; context=8192; uniform | rank | 0.50 | 0.50 | 0.24274 | 0.26282 | 0.23254 | 0.24603 | 0.01258 |
| D | small; E=4; context=8192; uniform | probability | 0.25 | 0.75 | 0.23960 | 0.25690 | 0.22509 | 0.24053 | 0.01300 |
| D | small; E=4; context=8192; uniform | rank | 0.25 | 0.75 | 0.23845 | 0.25652 | 0.22323 | 0.23940 | 0.01361 |
| D | small; E=1; context=8192; recency-aware | probability | 0.75 | 0.25 | 0.24613 | 0.26917 | 0.23871 | 0.25134 | 0.01297 |
| D | small; E=1; context=8192; recency-aware | rank | 0.75 | 0.25 | 0.24249 | 0.26777 | 0.23878 | 0.24968 | 0.01288 |
| D | small; E=1; context=8192; recency-aware | probability | 0.50 | 0.50 | 0.24409 | 0.26685 | 0.23221 | 0.24771 | 0.01437 |
| D | small; E=1; context=8192; recency-aware | rank | 0.50 | 0.50 | 0.23887 | 0.26433 | 0.23198 | 0.24506 | 0.01392 |
| D | small; E=1; context=8192; recency-aware | probability | 0.25 | 0.75 | 0.23566 | 0.26046 | 0.22283 | 0.23965 | 0.01562 |
| D | small; E=1; context=8192; recency-aware | rank | 0.25 | 0.75 | 0.23193 | 0.25669 | 0.22370 | 0.23744 | 0.01402 |
| D | small; E=1; context=8192; uniform | probability | 0.75 | 0.25 | 0.23975 | 0.26574 | 0.24100 | 0.24883 | 0.01197 |
| D | small; E=1; context=8192; uniform | rank | 0.75 | 0.25 | 0.24109 | 0.26414 | 0.24135 | 0.24886 | 0.01081 |
| D | small; E=1; context=8192; uniform | probability | 0.50 | 0.50 | 0.23555 | 0.25871 | 0.23679 | 0.24369 | 0.01064 |
| D | small; E=1; context=8192; uniform | rank | 0.50 | 0.50 | 0.23575 | 0.25665 | 0.23732 | 0.24324 | 0.00951 |
| D | small; E=1; context=8192; uniform | probability | 0.25 | 0.75 | 0.22917 | 0.24730 | 0.22897 | 0.23515 | 0.00859 |
| D | small; E=1; context=8192; uniform | rank | 0.25 | 0.75 | 0.22828 | 0.24493 | 0.22854 | 0.23392 | 0.00779 |
| D | small; E=1; context=8192; balanced-by-month | probability | 0.75 | 0.25 | 0.24429 | 0.26901 | 0.23664 | 0.24998 | 0.01381 |
| D | small; E=1; context=8192; balanced-by-month | rank | 0.75 | 0.25 | 0.24246 | 0.26782 | 0.23754 | 0.24927 | 0.01327 |
| D | small; E=1; context=8192; balanced-by-month | probability | 0.50 | 0.50 | 0.24058 | 0.26708 | 0.22856 | 0.24541 | 0.01609 |
| D | small; E=1; context=8192; balanced-by-month | rank | 0.50 | 0.50 | 0.23756 | 0.26367 | 0.22735 | 0.24286 | 0.01529 |
| D | small; E=1; context=8192; balanced-by-month | probability | 0.25 | 0.75 | 0.23201 | 0.26129 | 0.21465 | 0.23599 | 0.01925 |
| D | small; E=1; context=8192; balanced-by-month | rank | 0.25 | 0.75 | 0.22926 | 0.25543 | 0.21225 | 0.23231 | 0.01776 |
| D | large; E=1; context=60000; uniform | probability | 0.75 | 0.25 | 0.25310 | 0.24367 | 0.23919 | 0.24532 | 0.00580 |
| D | large; E=1; context=60000; uniform | rank | 0.75 | 0.25 | 0.24822 | 0.25537 | 0.23950 | 0.24770 | 0.00649 |
| D | large; E=1; context=60000; uniform | probability | 0.50 | 0.50 | 0.24637 | 0.22327 | 0.23208 | 0.23391 | 0.00952 |
| D | large; E=1; context=60000; uniform | rank | 0.50 | 0.50 | 0.24665 | 0.23428 | 0.22986 | 0.23693 | 0.00711 |
| D | large; E=1; context=60000; uniform | probability | 0.25 | 0.75 | 0.23138 | 0.19950 | 0.21517 | 0.21535 | 0.01302 |
| D | large; E=1; context=60000; uniform | rank | 0.25 | 0.75 | 0.23518 | 0.20711 | 0.21428 | 0.21886 | 0.01191 |
| D | medium; E=1; context=60000; uniform | probability | 0.75 | 0.25 | 0.24003 | 0.24247 | 0.22358 | 0.23536 | 0.00839 |
| D | medium; E=1; context=60000; uniform | rank | 0.75 | 0.25 | 0.24038 | 0.26069 | 0.23582 | 0.24563 | 0.01081 |
| D | medium; E=1; context=60000; uniform | probability | 0.50 | 0.50 | 0.22494 | 0.21888 | 0.20413 | 0.21598 | 0.00874 |
| D | medium; E=1; context=60000; uniform | rank | 0.50 | 0.50 | 0.23228 | 0.24222 | 0.22033 | 0.23161 | 0.00895 |
| D | medium; E=1; context=60000; uniform | probability | 0.25 | 0.75 | 0.21066 | 0.19625 | 0.18307 | 0.19666 | 0.01127 |
| D | medium; E=1; context=60000; uniform | rank | 0.25 | 0.75 | 0.21780 | 0.21353 | 0.19554 | 0.20896 | 0.00964 |

### Diversidad por fold

| Variante | Config | Mes | Pearson | Spearman |
|---|---|---:|---:|---:|
| A | small; E=1; context=60000; uniform | 202609 | 0.95668 | 0.85382 |
| A | small; E=1; context=60000; uniform | 202610 | 0.95692 | 0.84847 |
| A | small; E=1; context=60000; uniform | 202611 | 0.95086 | 0.81208 |
| D | small; E=1; context=60000; uniform | 202609 | 0.94530 | 0.82823 |
| D | small; E=1; context=60000; uniform | 202610 | 0.94986 | 0.82919 |
| D | small; E=1; context=60000; uniform | 202611 | 0.94322 | 0.78859 |
| D | small; E=8; context=8192; uniform | 202609 | 0.90841 | 0.75921 |
| D | small; E=8; context=8192; uniform | 202610 | 0.92564 | 0.78661 |
| D | small; E=8; context=8192; uniform | 202611 | 0.92590 | 0.77171 |
| D | small; E=4; context=8192; uniform | 202609 | 0.89617 | 0.74236 |
| D | small; E=4; context=8192; uniform | 202610 | 0.91155 | 0.80488 |
| D | small; E=4; context=8192; uniform | 202611 | 0.91414 | 0.74625 |
| D | small; E=1; context=8192; recency-aware | 202609 | 0.85776 | 0.75378 |
| D | small; E=1; context=8192; recency-aware | 202610 | 0.90637 | 0.76065 |
| D | small; E=1; context=8192; recency-aware | 202611 | 0.83640 | 0.79625 |
| D | small; E=1; context=8192; uniform | 202609 | 0.90551 | 0.75273 |
| D | small; E=1; context=8192; uniform | 202610 | 0.82864 | 0.76679 |
| D | small; E=1; context=8192; uniform | 202611 | 0.88066 | 0.70803 |
| D | small; E=1; context=8192; balanced-by-month | 202609 | 0.87081 | 0.71481 |
| D | small; E=1; context=8192; balanced-by-month | 202610 | 0.89403 | 0.75651 |
| D | small; E=1; context=8192; balanced-by-month | 202611 | 0.82843 | 0.67346 |
| D | large; E=1; context=60000; uniform | 202609 | 0.66437 | 0.56326 |
| D | large; E=1; context=60000; uniform | 202610 | 0.70365 | 0.63447 |
| D | large; E=1; context=60000; uniform | 202611 | 0.72004 | 0.59745 |
| D | medium; E=1; context=60000; uniform | 202609 | 0.72790 | 0.65021 |
| D | medium; E=1; context=60000; uniform | 202610 | 0.56825 | 0.52630 |
| D | medium; E=1; context=60000; uniform | 202611 | 0.58348 | 0.49348 |

## Context-size analysis

NVIDIA no documenta un máximo duro de inferencia; su benchmark expone `max_context_size` y su artículo describe entrenamiento con contextos de hasta 60,000 filas. El límite configurado para los experimentos de contexto máximo fue 60,000 filas; el máximo observado fue 60,000. Se usaron contextos por estimator de [8192, 60000]; 27 de 27 folds-config usaron sampling determinista al quedar el train por encima del contexto. Tiempos promedio por fold-config: large: 50.8 s/fold-config en promedio, medium: 29.2 s/fold-config en promedio, small: 8.9 s/fold-config en promedio. El intento exploratorio local de 100,600 filas excedió la VRAM de la GTX 1650; no se presenta como límite del modelo. El código oficial soporta `fit`/`predict` con KV cache y contextos de estimator en batch; esta arena promedió 1, 4 u 8 estimators con contextos muestreados de forma determinista. `predict` procesa batches de 256.

## Leakage audit

- `model.fit` recibe únicamente etiquetas de `mes <= train_through`; nunca recibe `objetivo` del mes validado.
- `objetivo` no entra a X y no se usa target histórico.
- `id_cliente` no entra a X; sólo alinea predicciones y audita recurrencia. `context_query_client_overlap` por fold queda en `fold_results.csv` y `manifest.json`; la recurrencia de un cliente entre contexto pasado y query posterior está permitida.
- Las features de historial son covariables cronológicas: observaciones previas, primera aparición y duración. No consultan etiquetas. Primera aparición y duraciones se calculan hasta el mes de cada fila.

## Limitaciones

Se ejecutaron large, medium, small en Tesla T4; Medium/Large se limitaron a D, E=1 y el límite de contexto máximo configurado. Las mediciones por fold, contexto real, checkpoint y memoria están en `fold_results.csv`; hardware y versiones en `manifest.json`. Los estimators se ejecutan secuencialmente (`estimator_batch_size=1`).
El softmax no se calibró y no se hizo búsqueda de hiperparámetros. El ganador entre mezclas usa únicamente pesos fijos 75/25, 50/50 y 25/75, pero sigue siendo la mejor fila observada entre configuraciones y folds compartidos; es exploratorio y debe validarse con bootstrap pareado/futuros meses antes de usarse.
