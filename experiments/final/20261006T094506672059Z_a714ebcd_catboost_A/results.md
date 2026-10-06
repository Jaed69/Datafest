# Final fit 20261006T094506672059Z_a714ebcd_catboost_A

- Model: catboost; features: A; iterations: 192 (no early stopping).
- Trained on all labelled months (Jan-Nov 2026), scored December 2026.

## Prediction distribution

- rows: 9900
- mean: 0.14066
- std: 0.05933
- min: 0.02909
- max: 0.42633

| Quantile | Value |
| ---: | ---: |
| 0.01 | 0.04616 |
| 0.05 | 0.05360 |
| 0.25 | 0.11880 |
| 0.5 | 0.13756 |
| 0.75 | 0.15569 |
| 0.95 | 0.25032 |
| 0.99 | 0.36516 |

## Agreement with another submission

- compared with: `experiments/final/20261006T094413527919Z_a714ebcd_lightgbm_D_absolute/submission.csv`
- Spearman correlation: 0.85884
