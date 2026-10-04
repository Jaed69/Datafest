# NVIDIA Kumo Tabular — 20261003_kumo_small_rolling

## Executive summary

Mejor Kumo: **small; E=1; context=32768; uniform, variante A**, mean rolling Gini **0.24056** (Δ vs LightGBM D: **-0.01048**). Sep/Oct/Nov: 0.23279/0.25632/0.23257; std temporal 0.01115.
Mejor Kumo-D para diversidad: **small; E=1; context=32768; uniform**, mean Gini 0.23901; correlaciones por fold: 202609 Pearson 0.932/Spearman 0.785, 202610 Pearson 0.949/Spearman 0.827, 202611 Pearson 0.928/Spearman 0.777.
Conclusión operativa: **ensemble-only (marginal blend gain; bootstrap required)**.
Costo: 25.6 min de inferencia acumulada, pico VRAM observado 1.38 GiB; hardware {'platform': 'Linux-7.0.0-34-generic-x86_64-with-glibc2.39', 'processor': 'x86_64', 'cpu_count': 12, 'gpu': 'NVIDIA GeForce GTX 1650', 'gpu_total_gib': 3.628}.
Mejor blend evaluado: small; E=1; context=32768; uniform con probability, 75% LGBM / 25% Kumo, mean Gini 0.25187 (Δ vs LGBM-D +0.00083).

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
| KumoTabular | A | small; E=1; context=32768; uniform | 0.23279 | 0.25632 | 0.23257 | 0.24056 | 0.01115 | -0.01048 |
| KumoTabular | D | small; E=1; context=32768; uniform | 0.23152 | 0.25527 | 0.23025 | 0.23901 | 0.01151 | -0.01203 |
| KumoTabular | D | small; E=1; context=8192; uniform | 0.22066 | 0.22934 | 0.21684 | 0.22228 | 0.00523 | -0.02876 |
| KumoTabular | D | small; E=1; context=8192; recency-aware | 0.22362 | 0.24653 | 0.21181 | 0.22732 | 0.01442 | -0.02372 |
| KumoTabular | D | small; E=1; context=8192; balanced-by-month | 0.21904 | 0.24510 | 0.19505 | 0.21973 | 0.02044 | -0.03131 |
| KumoTabular | D | small; E=4; context=8192; uniform | 0.23198 | 0.24816 | 0.21328 | 0.23114 | 0.01425 | -0.01990 |
| KumoTabular | D | small; E=8; context=8192; uniform | 0.23213 | 0.25146 | 0.21723 | 0.23361 | 0.01401 | -0.01743 |
| KumoTabular | D | medium; E=1; context=8192; uniform | 0.23135 | 0.23022 | 0.21197 | 0.22451 | 0.00888 | -0.02653 |
| KumoTabular | D | large; E=1; context=8192; uniform | 0.23225 | 0.23801 | 0.22433 | 0.23153 | 0.00561 | -0.01951 |

## ROC AUC y Average Precision por mes

| Config | ROC AUC Sep | ROC AUC Oct | ROC AUC Nov | AP Sep | AP Oct | AP Nov |
|---|---:|---:|---:|---:|---:|---:|
| small; E=1; context=32768; uniform / A | 0.61639 | 0.62816 | 0.61629 | 0.22479 | 0.25098 | 0.22515 |
| small; E=1; context=32768; uniform / D | 0.61576 | 0.62763 | 0.61512 | 0.22168 | 0.25121 | 0.22644 |
| small; E=1; context=8192; uniform / D | 0.61033 | 0.61467 | 0.60842 | 0.21568 | 0.24033 | 0.22369 |
| small; E=1; context=8192; recency-aware / D | 0.61181 | 0.62327 | 0.60590 | 0.21244 | 0.24746 | 0.21385 |
| small; E=1; context=8192; balanced-by-month / D | 0.60952 | 0.62255 | 0.59753 | 0.22262 | 0.24577 | 0.21792 |
| small; E=4; context=8192; uniform / D | 0.61599 | 0.62408 | 0.60664 | 0.21899 | 0.24649 | 0.22225 |
| small; E=8; context=8192; uniform / D | 0.61607 | 0.62573 | 0.60861 | 0.22147 | 0.24850 | 0.22602 |
| medium; E=1; context=8192; uniform / D | 0.61567 | 0.61511 | 0.60598 | 0.21682 | 0.23809 | 0.22423 |
| large; E=1; context=8192; uniform / D | 0.61612 | 0.61901 | 0.61217 | 0.21714 | 0.24235 | 0.22420 |

## Comparación con benchmarks

Los cuatro valores históricos son rolling y se copian sin recalcularlos. No se mezclan métricas frozen ni pooled.

## Ensemble

| Config Kumo D | Método | LGBM | Kumo | Sep | Oct | Nov | Mean | Std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small; E=1; context=32768; uniform | probability | 0.75 | 0.25 | 0.24348 | 0.26835 | 0.24380 | 0.25187 | 0.01165 |
| small; E=1; context=32768; uniform | rank | 0.75 | 0.25 | 0.24320 | 0.26787 | 0.24309 | 0.25139 | 0.01165 |
| small; E=1; context=32768; uniform | probability | 0.50 | 0.50 | 0.24175 | 0.26677 | 0.24264 | 0.25039 | 0.01159 |
| small; E=1; context=32768; uniform | rank | 0.50 | 0.50 | 0.24109 | 0.26556 | 0.24174 | 0.24947 | 0.01139 |
| small; E=1; context=32768; uniform | probability | 0.25 | 0.75 | 0.23832 | 0.26274 | 0.23875 | 0.24660 | 0.01141 |
| small; E=1; context=32768; uniform | rank | 0.25 | 0.75 | 0.23649 | 0.26130 | 0.23680 | 0.24487 | 0.01162 |
| small; E=8; context=8192; uniform | probability | 0.75 | 0.25 | 0.24309 | 0.26813 | 0.24017 | 0.25046 | 0.01255 |
| small; E=8; context=8192; uniform | rank | 0.75 | 0.25 | 0.24398 | 0.26783 | 0.23947 | 0.25043 | 0.01244 |
| small; E=8; context=8192; uniform | probability | 0.50 | 0.50 | 0.24148 | 0.26536 | 0.23524 | 0.24736 | 0.01298 |
| small; E=8; context=8192; uniform | rank | 0.50 | 0.50 | 0.24242 | 0.26509 | 0.23420 | 0.24723 | 0.01306 |
| small; E=8; context=8192; uniform | probability | 0.25 | 0.75 | 0.23879 | 0.25971 | 0.22760 | 0.24203 | 0.01331 |
| small; E=8; context=8192; uniform | rank | 0.25 | 0.75 | 0.23823 | 0.25911 | 0.22601 | 0.24112 | 0.01367 |
| large; E=1; context=8192; uniform | probability | 0.75 | 0.25 | 0.24052 | 0.26610 | 0.24238 | 0.24967 | 0.01164 |
| large; E=1; context=8192; uniform | rank | 0.75 | 0.25 | 0.24363 | 0.26517 | 0.24229 | 0.25036 | 0.01049 |
| large; E=1; context=8192; uniform | probability | 0.50 | 0.50 | 0.23864 | 0.26101 | 0.24015 | 0.24660 | 0.01021 |
| large; E=1; context=8192; uniform | rank | 0.50 | 0.50 | 0.24185 | 0.25959 | 0.23958 | 0.24701 | 0.00894 |
| large; E=1; context=8192; uniform | probability | 0.25 | 0.75 | 0.23645 | 0.25150 | 0.23470 | 0.24089 | 0.00754 |
| large; E=1; context=8192; uniform | rank | 0.25 | 0.75 | 0.23745 | 0.25065 | 0.23233 | 0.24014 | 0.00772 |
| small; E=4; context=8192; uniform | probability | 0.75 | 0.25 | 0.24419 | 0.26684 | 0.23831 | 0.24978 | 0.01230 |
| small; E=4; context=8192; uniform | rank | 0.75 | 0.25 | 0.24453 | 0.26666 | 0.23893 | 0.25004 | 0.01197 |
| small; E=4; context=8192; uniform | probability | 0.50 | 0.50 | 0.24289 | 0.26330 | 0.23333 | 0.24650 | 0.01250 |
| small; E=4; context=8192; uniform | rank | 0.50 | 0.50 | 0.24273 | 0.26282 | 0.23255 | 0.24603 | 0.01257 |
| small; E=4; context=8192; uniform | probability | 0.25 | 0.75 | 0.23955 | 0.25691 | 0.22511 | 0.24052 | 0.01300 |
| small; E=4; context=8192; uniform | rank | 0.25 | 0.75 | 0.23841 | 0.25654 | 0.22323 | 0.23940 | 0.01362 |
| small; E=1; context=8192; recency-aware | probability | 0.75 | 0.25 | 0.24617 | 0.26916 | 0.23871 | 0.25135 | 0.01296 |
| small; E=1; context=8192; recency-aware | rank | 0.75 | 0.25 | 0.24250 | 0.26777 | 0.23878 | 0.24968 | 0.01288 |
| small; E=1; context=8192; recency-aware | probability | 0.50 | 0.50 | 0.24412 | 0.26681 | 0.23218 | 0.24771 | 0.01436 |
| small; E=1; context=8192; recency-aware | rank | 0.50 | 0.50 | 0.23887 | 0.26431 | 0.23194 | 0.24504 | 0.01392 |
| small; E=1; context=8192; recency-aware | probability | 0.25 | 0.75 | 0.23569 | 0.26045 | 0.22281 | 0.23965 | 0.01562 |
| small; E=1; context=8192; recency-aware | rank | 0.25 | 0.75 | 0.23194 | 0.25672 | 0.22366 | 0.23744 | 0.01404 |
| medium; E=1; context=8192; uniform | probability | 0.75 | 0.25 | 0.23983 | 0.26497 | 0.24025 | 0.24835 | 0.01175 |
| medium; E=1; context=8192; uniform | rank | 0.75 | 0.25 | 0.24491 | 0.26420 | 0.24047 | 0.24986 | 0.01030 |
| medium; E=1; context=8192; uniform | probability | 0.50 | 0.50 | 0.23900 | 0.25881 | 0.23461 | 0.24414 | 0.01053 |
| medium; E=1; context=8192; uniform | rank | 0.50 | 0.50 | 0.24317 | 0.25676 | 0.23451 | 0.24481 | 0.00916 |
| medium; E=1; context=8192; uniform | probability | 0.25 | 0.75 | 0.23790 | 0.24788 | 0.22552 | 0.23710 | 0.00915 |
| medium; E=1; context=8192; uniform | rank | 0.25 | 0.75 | 0.23887 | 0.24514 | 0.22484 | 0.23628 | 0.00849 |
| small; E=1; context=8192; uniform | probability | 0.75 | 0.25 | 0.23975 | 0.26577 | 0.24104 | 0.24885 | 0.01197 |
| small; E=1; context=8192; uniform | rank | 0.75 | 0.25 | 0.24109 | 0.26413 | 0.24138 | 0.24887 | 0.01080 |
| small; E=1; context=8192; uniform | probability | 0.50 | 0.50 | 0.23554 | 0.25874 | 0.23684 | 0.24371 | 0.01064 |
| small; E=1; context=8192; uniform | rank | 0.50 | 0.50 | 0.23578 | 0.25664 | 0.23740 | 0.24327 | 0.00947 |
| small; E=1; context=8192; uniform | probability | 0.25 | 0.75 | 0.22919 | 0.24727 | 0.22907 | 0.23518 | 0.00855 |
| small; E=1; context=8192; uniform | rank | 0.25 | 0.75 | 0.22831 | 0.24488 | 0.22861 | 0.23393 | 0.00774 |
| small; E=1; context=8192; balanced-by-month | probability | 0.75 | 0.25 | 0.24431 | 0.26905 | 0.23662 | 0.24999 | 0.01384 |
| small; E=1; context=8192; balanced-by-month | rank | 0.75 | 0.25 | 0.24250 | 0.26785 | 0.23756 | 0.24930 | 0.01327 |
| small; E=1; context=8192; balanced-by-month | probability | 0.50 | 0.50 | 0.24064 | 0.26714 | 0.22854 | 0.24544 | 0.01612 |
| small; E=1; context=8192; balanced-by-month | rank | 0.50 | 0.50 | 0.23762 | 0.26376 | 0.22737 | 0.24292 | 0.01532 |
| small; E=1; context=8192; balanced-by-month | probability | 0.25 | 0.75 | 0.23213 | 0.26142 | 0.21466 | 0.23607 | 0.01929 |
| small; E=1; context=8192; balanced-by-month | rank | 0.25 | 0.75 | 0.22935 | 0.25557 | 0.21228 | 0.23240 | 0.01780 |

### Diversidad por fold

| Config | Mes | Pearson | Spearman |
|---|---:|---:|---:|
| small; E=1; context=32768; uniform | 202609 | 0.93227 | 0.78534 |
| small; E=1; context=32768; uniform | 202610 | 0.94853 | 0.82686 |
| small; E=1; context=32768; uniform | 202611 | 0.92775 | 0.77713 |
| small; E=8; context=8192; uniform | 202609 | 0.90839 | 0.75923 |
| small; E=8; context=8192; uniform | 202610 | 0.92560 | 0.78662 |
| small; E=8; context=8192; uniform | 202611 | 0.92586 | 0.77168 |
| large; E=1; context=8192; uniform | 202609 | 0.90627 | 0.76536 |
| large; E=1; context=8192; uniform | 202610 | 0.87744 | 0.79098 |
| large; E=1; context=8192; uniform | 202611 | 0.90455 | 0.75147 |
| small; E=4; context=8192; uniform | 202609 | 0.89614 | 0.74226 |
| small; E=4; context=8192; uniform | 202610 | 0.91158 | 0.80483 |
| small; E=4; context=8192; uniform | 202611 | 0.91409 | 0.74635 |
| small; E=1; context=8192; recency-aware | 202609 | 0.85767 | 0.75388 |
| small; E=1; context=8192; recency-aware | 202610 | 0.90635 | 0.76087 |
| small; E=1; context=8192; recency-aware | 202611 | 0.83637 | 0.79632 |
| medium; E=1; context=8192; uniform | 202609 | 0.88198 | 0.72402 |
| medium; E=1; context=8192; uniform | 202610 | 0.84209 | 0.79022 |
| medium; E=1; context=8192; uniform | 202611 | 0.87375 | 0.69395 |
| small; E=1; context=8192; uniform | 202609 | 0.90560 | 0.75284 |
| small; E=1; context=8192; uniform | 202610 | 0.82853 | 0.76640 |
| small; E=1; context=8192; uniform | 202611 | 0.88068 | 0.70799 |
| small; E=1; context=8192; balanced-by-month | 202609 | 0.87079 | 0.71472 |
| small; E=1; context=8192; balanced-by-month | 202610 | 0.89414 | 0.75667 |
| small; E=1; context=8192; balanced-by-month | 202611 | 0.82834 | 0.67335 |

## Context-size analysis

NVIDIA no fija un máximo de inferencia; el benchmark oficial expone `max_context_size` opcional y el artículo técnico describe entrenamiento con contextos de hasta 60,000 filas. En esta GPU, el barrido Small con 8,192 filas requirió ~15–18 s por fold para E=1 y ~0.46 GiB de VRAM; 32,768 filas requirieron ~75–80 s y ~1.38 GiB. La prueba de 100,600 filas no completó en el presupuesto práctico y emitió una asignación fallida de 208 MB cerca de 3.6/4 GiB. La arena usó 32,768 para Small A/D E=1 y 8,192 para las variantes de muestreo, ensembles Small y Medium/Large. En todos los contextos menores que el train se usó selección uniforme determinista, salvo las tres políticas D Small comparadas explícitamente. `fit` reutiliza KV cache; `predict` procesa batches de 256.

## Leakage audit

- `model.fit` recibe únicamente etiquetas de `mes <= train_through`; nunca recibe `objetivo` del mes validado.
- `objetivo` no entra a X y no se usa target histórico.
- `id_cliente` no entra a X; sólo alinea predicciones y audita recurrencia. `context_query_client_overlap` por fold queda en `fold_results.csv` y `manifest.json`; la recurrencia de un cliente entre contexto pasado y query posterior está permitida.
- Las features de historial son covariables cronológicas: observaciones previas, primera aparición y duración. No consultan etiquetas. Primera aparición y duraciones se calculan hasta el mes de cada fila.

## Limitaciones

Se ejecutaron Small, Medium y Large en NVIDIA GeForce GTX 1650; Medium/Large se limitaron a D, E=1 y contexto 8,192 para esta arena. Las mediciones por fold, contexto real, checkpoint y memoria están en `fold_results.csv`; hardware y versiones en `manifest.json`. Los estimators se ejecutan secuencialmente (`estimator_batch_size=1`).
El softmax no se calibró y no se hizo búsqueda de hiperparámetros. El mayor contexto Small estable medido fue 32,768; 100,600 no completó. El blend líder supera al baseline por un margen pequeño y no constituye evidencia de significancia; conviene validarlo con bootstrap pareado antes de promoverlo.
