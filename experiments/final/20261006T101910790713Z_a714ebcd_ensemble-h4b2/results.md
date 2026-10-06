# Final ensemble 20261006T101910790713Z_a714ebcd_ensemble-h4b2

- Model: H4b2 = equal-weight rank-average of LightGBM D (B0) and CatBoost A with `dias_ultima_interaccion` -> within-month percentile rank (H1b).
- Iterations: {'lightgbm': 58, 'catboost': 192} (frozen, no early stopping); seed 42; trained on all Jan-Nov 2026 rows, scored December 2026.
- CatBoost rank feature: per month in train, within December only for test.
- Selection: rolling mean Gini 0.25245 vs B0 0.25104 (+0.0014, CI [-0.0032, +0.0059]); not significant, chosen by blind judges.

## Output mapping

The blend is a mean of percentile ranks over the whole December test set. To submit probabilities, the blend is quantile-mapped onto the sorted LightGBM December probabilities (k-th smallest blend gets the k-th smallest LightGBM probability; ties share the first value of the tie group). Order, hence AUC/Gini, equals the blend's exactly; the marginal distribution equals LightGBM's.

## Pre-submission checks

- rows: 9900 (sample_submission order and columns validated); finite: True; within [0,1]: True
- overall: mean 0.14781, p10 0.06318, p50 0.14504, p90 0.22904
- groups: 8061 survivors, 1839 new clients (id > max train id), 0 other

| Group | Rows | Mean | p10 | p50 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nov survivors | 8061 | 0.14184 | 0.06280 | 0.14360 | 0.19837 |
| New clients | 1839 | 0.17400 | 0.10514 | 0.15180 | 0.26779 |

## Spearman vs B0 submission (`experiments/final/20261006T094413527919Z_a714ebcd_lightgbm_D_absolute/submission.csv`)

- overall: 0.96278
- Nov survivors: 0.96327
- new clients: 0.96034
