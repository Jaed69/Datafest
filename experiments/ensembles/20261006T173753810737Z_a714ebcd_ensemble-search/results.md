# Ensemble decision `20261006T173753810737Z_a714ebcd_ensemble-search`

Rolling Sep/Oct/Nov out-of-fold predictions (train <=Aug->Sep, <=Sep->Oct, <=Oct->Nov). Candidate list fixed a priori; every blend is a weighted mean of within-month percentile ranks. Paired client bootstrap: 2000 resamples, one shared client draw across months and models; delta CIs are unadjusted percentile 95%; p is the two-sided centered bootstrap p with plus-one correction, Holm-adjusted across the family (fixed candidates + the two LOMO procedures).

Reproduction checks (rank-averaging the stored members against the stored columns): H4b2 mean 0.25245 vs stored 0.25245, E1 mean 0.25349 vs stored 0.25349

## Candidates

| Candidate | Sep | Oct | Nov | Mean | Std | dMean vs H4b2 [95% CI] | p Holm | P(best) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| H4b2 | 0.24715 | 0.26804 | 0.24216 | 0.25245 | 0.01121 | - | - | 0.022 |
| H4b2+N1 | 0.24867 | 0.26727 | 0.24701 | 0.25432 | 0.00918 | +0.0019 [-0.0006, +0.0043] | 1.000 | 0.061 |
| E1 | 0.24876 | 0.26897 | 0.24273 | 0.25349 | 0.01122 | +0.0010 [-0.0017, +0.0037] | 1.000 | 0.102 |
| E1+N1 | 0.24961 | 0.26775 | 0.24708 | 0.25482 | 0.00921 | +0.0024 [-0.0010, +0.0057] | 1.000 | 0.264 |
| B0+N1 | 0.24779 | 0.26709 | 0.24911 | 0.25466 | 0.00880 | +0.0022 [-0.0018, +0.0064] | 1.000 | 0.359 |
| 0.5*rank(H4b2)+0.5*rank(N1) | 0.24910 | 0.26623 | 0.24832 | 0.25455 | 0.00826 | +0.0021 [-0.0014, +0.0056] | 1.000 | 0.192 |
| STACK_LOMO | 0.24830 | 0.26374 | 0.23938 | 0.25047 | 0.01006 | -0.0020 [-0.0065, +0.0027] | 1.000 | - |
| BESTOF_LOMO | 0.24779 | 0.26623 | 0.24273 | 0.25225 | 0.01010 | -0.0002 [-0.0036, +0.0031] | 1.000 | - |

`STACK_LOMO` and `BESTOF_LOMO` are out-of-selection estimates (see below); P(best) is only defined for the fixed candidates (share of bootstrap replicates in which the candidate has the highest mean Gini; ties split).

## Per-month deltas vs H4b2 (95% CI)

| Candidate | dSep | dOct | dNov |
| --- | --- | --- | --- |
| H4b2+N1 | +0.0015 [-0.0028, +0.0056] | -0.0008 [-0.0046, +0.0031] | +0.0048 [+0.0001, +0.0098] |
| E1 | +0.0016 [-0.0034, +0.0064] | +0.0009 [-0.0032, +0.0050] | +0.0006 [-0.0040, +0.0051] |
| E1+N1 | +0.0025 [-0.0037, +0.0081] | -0.0003 [-0.0054, +0.0048] | +0.0049 [-0.0014, +0.0114] |
| B0+N1 | +0.0006 [-0.0067, +0.0076] | -0.0009 [-0.0068, +0.0050] | +0.0069 [-0.0005, +0.0148] |
| 0.5*rank(H4b2)+0.5*rank(N1) | +0.0019 [-0.0044, +0.0078] | -0.0018 [-0.0072, +0.0036] | +0.0062 [-0.0006, +0.0131] |
| STACK_LOMO | +0.0011 [-0.0071, +0.0090] | -0.0043 [-0.0133, +0.0048] | -0.0028 [-0.0096, +0.0041] |
| BESTOF_LOMO | +0.0006 [-0.0067, +0.0076] | -0.0018 [-0.0072, +0.0036] | +0.0006 [-0.0040, +0.0051] |

## Leave-one-month-out rank-weight stack (non-negative, sum 1, grid step 0.1)

Members: B0, H1b, catF1, N1, xgb. Weights are fitted on two months (maximising their mean Gini) and the third is scored; the held-out month is never used for fitting.

| Held-out month | B0 | H1b | catF1 | N1 | xgb | Held-out Gini | H4b2 Gini |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 202609 | 0.40 | 0.00 | 0.00 | 0.60 | 0.00 | 0.24830 | 0.24715 |
| 202610 | 0.10 | 0.00 | 0.10 | 0.80 | 0.00 | 0.26374 | 0.26804 |
| 202611 | 0.20 | 0.00 | 0.50 | 0.00 | 0.30 | 0.23938 | 0.24216 |

LOMO stack mean Gini 0.25047 vs H4b2 0.25245 (-0.0020). Descriptive in-sample weights fitted on all three months (not an estimate of anything): B0=0.30, H1b=0.00, catF1=0.20, N1=0.50, xgb=0.00.

## Best-of-list selection bias (LOMO)

Choose the candidate with the best mean on two months, score it on the third.

| Held-out month | Chosen on the other two | Held-out Gini | H4b2 Gini | Delta |
| --- | --- | ---: | ---: | ---: |
| 202609 | B0+N1 | 0.24779 | 0.24715 | +0.0006 |
| 202610 | 0.5*rank(H4b2)+0.5*rank(N1) | 0.26623 | 0.26804 | -0.0018 |
| 202611 | E1 | 0.24273 | 0.24216 | +0.0006 |

Best-of LOMO mean Gini 0.25225 vs H4b2 0.25245 (-0.0002).

## How often each candidate is best

Point-estimate wins per month, and share of bootstrap replicates in which the candidate is best (per month and on the mean).

| Candidate | Wins (of 3 months) | P(best) Sep | P(best) Oct | P(best) Nov | P(best) mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4b2 | 0 | 0.091 | 0.222 | 0.011 | 0.022 |
| H4b2+N1 | 0 | 0.032 | 0.050 | 0.011 | 0.061 |
| E1 | 1 | 0.265 | 0.466 | 0.009 | 0.102 |
| E1+N1 | 1 | 0.232 | 0.083 | 0.081 | 0.264 |
| B0+N1 | 1 | 0.106 | 0.147 | 0.590 | 0.359 |
| 0.5*rank(H4b2)+0.5*rank(N1) | 0 | 0.274 | 0.032 | 0.298 | 0.192 |

## December sanity

- Buildable from existing December outputs (members B0, H1b, N1): H4b2, H4b2+N1, B0+N1, 0.5*rank(H4b2)+0.5*rank(N1).
- Not buildable without a new December fit (need cat+F1 or xgb): E1, E1+N1.
- Top 2 buildable by rolling mean: B0+N1, 0.5*rank(H4b2)+0.5*rank(N1).
- Id order checks: {'members_same_order_as_sample': True, 'neural_same_order_as_sample': True}.

| Pair | Spearman |
| --- | ---: |
| H4b2 vs H4b2+N1 | 0.9877 |
| H4b2 vs B0+N1 | 0.9693 |
| H4b2 vs 0.5*rank(H4b2)+0.5*rank(N1) | 0.9757 |
| H4b2+N1 vs B0+N1 | 0.9880 |
| H4b2+N1 vs 0.5*rank(H4b2)+0.5*rank(N1) | 0.9978 |
| B0+N1 vs 0.5*rank(H4b2)+0.5*rank(N1) | 0.9889 |

| Candidate | Spearman vs current H4b2 submission |
| --- | ---: |
| H4b2 | 1.0000 |
| H4b2+N1 | 0.9877 |
| B0+N1 | 0.9693 |
| 0.5*rank(H4b2)+0.5*rank(N1) | 0.9757 |

No submission file was written by this run.
