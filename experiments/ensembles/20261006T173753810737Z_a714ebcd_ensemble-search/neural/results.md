# Neural tabular candidates (T9)

Rolling folds: train <=Aug->Sep, <=Sep->Oct, <=Oct->Nov (same as repo). Features: repo `D_absolute` matrix (history + month), categoricals as pandas categories, booleans as 0/1; `Drank` variants replace `dias_ultima_interaccion` by its within-month percentile rank (exploratory). pytabkit defaults (RealMLP_TD_Classifier, TabM_D_Classifier), `val_metric_name='1-auc_ovr'`, device cuda; internal best-epoch/early-stopping hold-out = pytabkit default random 20% of the TRAINING rows only; seeds per candidate: {'N1/D': [42, 43, 44], 'N1/Drank': [42, 43, 44], 'N2/D': [42, 43, 44], 'N2/Drank': [42, 43, 44]}; seed-averaged by mean probability. GBDT members come from `oof_predictions.csv` (run a714ebcd), joined by (id_cliente, mes). Blends = equal-weight per-month rank average of B0 + H1b_catboost (the H4b2 members) plus the neural member(s). Paired client bootstrap, 2000 resamples, one shared client draw across months and models (repo `paired_bootstrap`, seed 42).

Champion reproduction from its members: mean Gini 0.25245 vs stored 0.25245.

| Candidate | Sep | Oct | Nov | Mean | Std | Spearman vs B0 | dMean vs B0 [95% CI] | dMean vs H4b2 [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BL_H4b2+N1 | 0.24868 | 0.26727 | 0.24701 | 0.25432 | 0.00918 | 0.956 | +0.0033 [-0.0023, +0.0088] | +0.0019 [-0.0006, +0.0043] |
| N1 | 0.24890 | 0.26142 | 0.25075 | 0.25369 | 0.00551 | 0.888 | +0.0026 [-0.0062, +0.0111] | +0.0012 [-0.0054, +0.0082] |
| BL_H4b2+N1+N2 | 0.25054 | 0.26491 | 0.24392 | 0.25312 | 0.00876 | 0.942 | +0.0021 [-0.0044, +0.0084] | +0.0007 [-0.0029, +0.0041] |
| BL_H4b2+N1r | 0.24720 | 0.26734 | 0.24394 | 0.25282 | 0.01035 | 0.954 | +0.0018 [-0.0039, +0.0072] | +0.0004 [-0.0020, +0.0028] |
| H4b2_rank_lgbm_catboost | 0.24715 | 0.26804 | 0.24216 | 0.25245 | 0.01121 | 0.970 | +0.0014 [-0.0032, +0.0059] | - |
| BL_H4b2+N1r+N2r | 0.24964 | 0.26489 | 0.24147 | 0.25200 | 0.00971 | 0.942 | +0.0010 [-0.0057, +0.0073] | -0.0005 [-0.0038, +0.0030] |
| BL_H4b2+N2 | 0.24992 | 0.26500 | 0.23938 | 0.25143 | 0.01051 | 0.948 | +0.0004 [-0.0058, +0.0062] | -0.0010 [-0.0041, +0.0019] |
| BL_H4b2+N2r | 0.25049 | 0.26474 | 0.23905 | 0.25143 | 0.01051 | 0.948 | +0.0004 [-0.0058, +0.0062] | -0.0010 [-0.0041, +0.0020] |
| B0_lgbm_D | 0.24337 | 0.26773 | 0.24203 | 0.25104 | 0.01181 | 1.000 | - | -0.0014 [-0.0059, +0.0032] |
| N1r | 0.24511 | 0.26213 | 0.24388 | 0.25037 | 0.00833 | 0.884 | -0.0007 [-0.0094, +0.0077] | -0.0021 [-0.0088, +0.0047] |
| H1b_catboost | 0.24770 | 0.26364 | 0.23818 | 0.24984 | 0.01050 | 0.887 | -0.0012 [-0.0101, +0.0076] | -0.0026 [-0.0071, +0.0020] |
| B1_catboost_A | 0.24713 | 0.26508 | 0.23575 | 0.24932 | 0.01207 | 0.882 | -0.0017 [-0.0105, +0.0071] | -0.0031 [-0.0088, +0.0022] |
| N2r | 0.24864 | 0.24896 | 0.22268 | 0.24009 | 0.01231 | 0.846 | -0.0110 [-0.0217, -0.0009] | -0.0124 [-0.0212, -0.0040] |
| N2 | 0.24521 | 0.24787 | 0.22248 | 0.23852 | 0.01139 | 0.838 | -0.0125 [-0.0233, -0.0025] | -0.0139 [-0.0228, -0.0055] |

## Per-month deltas vs H4b2 (95% CI)

| Candidate | dSep | dOct | dNov | p Holm (vs H4b2) |
| --- | --- | --- | --- | ---: |
| BL_H4b2+N1 | +0.0015 [-0.0028, +0.0056] | -0.0008 [-0.0046, +0.0031] | +0.0048 [+0.0001, +0.0098] | 1.000 |
| N1 | +0.0018 [-0.0108, +0.0135] | -0.0066 [-0.0174, +0.0040] | +0.0086 [-0.0047, +0.0222] | 1.000 |
| BL_H4b2+N1+N2 | +0.0034 [-0.0031, +0.0093] | -0.0031 [-0.0089, +0.0026] | +0.0018 [-0.0049, +0.0081] | 1.000 |
| BL_H4b2+N1r | +0.0000 [-0.0041, +0.0043] | -0.0007 [-0.0045, +0.0033] | +0.0018 [-0.0026, +0.0062] | 1.000 |
| BL_H4b2+N1r+N2r | +0.0025 [-0.0036, +0.0080] | -0.0031 [-0.0086, +0.0023] | -0.0007 [-0.0069, +0.0051] | 1.000 |
| BL_H4b2+N2 | +0.0028 [-0.0027, +0.0083] | -0.0030 [-0.0079, +0.0017] | -0.0028 [-0.0082, +0.0026] | 1.000 |
| BL_H4b2+N2r | +0.0033 [-0.0019, +0.0086] | -0.0033 [-0.0081, +0.0013] | -0.0031 [-0.0085, +0.0023] | 1.000 |
| B0_lgbm_D | -0.0038 [-0.0116, +0.0039] | -0.0003 [-0.0077, +0.0064] | -0.0001 [-0.0089, +0.0085] | 1.000 |
| N1r | -0.0020 [-0.0138, +0.0102] | -0.0059 [-0.0168, +0.0056] | +0.0017 [-0.0102, +0.0142] | 1.000 |
| H1b_catboost | +0.0005 [-0.0074, +0.0081] | -0.0044 [-0.0112, +0.0030] | -0.0040 [-0.0128, +0.0049] | 1.000 |
| B1_catboost_A | -0.0000 [-0.0099, +0.0096] | -0.0030 [-0.0118, +0.0057] | -0.0064 [-0.0176, +0.0043] | 1.000 |
| N2r | +0.0015 [-0.0136, +0.0163] | -0.0191 [-0.0323, -0.0061] | -0.0195 [-0.0346, -0.0040] | 0.066 |
| N2 | -0.0019 [-0.0178, +0.0138] | -0.0202 [-0.0339, -0.0068] | -0.0197 [-0.0350, -0.0043] | 0.019 |

## Per-month deltas vs B0 (95% CI)

| Candidate | dSep | dOct | dNov | p Holm (vs B0) |
| --- | --- | --- | --- | ---: |
| BL_H4b2+N1 | +0.0053 [-0.0039, +0.0149] | -0.0005 [-0.0089, +0.0087] | +0.0050 [-0.0059, +0.0157] | 1.000 |
| N1 | +0.0055 [-0.0098, +0.0203] | -0.0063 [-0.0199, +0.0077] | +0.0087 [-0.0082, +0.0253] | 1.000 |
| BL_H4b2+N1+N2 | +0.0072 [-0.0035, +0.0179] | -0.0028 [-0.0125, +0.0076] | +0.0019 [-0.0101, +0.0142] | 1.000 |
| BL_H4b2+N1r | +0.0038 [-0.0057, +0.0133] | -0.0004 [-0.0092, +0.0090] | +0.0019 [-0.0089, +0.0121] | 1.000 |
| H4b2_rank_lgbm_catboost | +0.0038 [-0.0039, +0.0116] | +0.0003 [-0.0064, +0.0077] | +0.0001 [-0.0085, +0.0089] | 1.000 |
| BL_H4b2+N1r+N2r | +0.0063 [-0.0044, +0.0171] | -0.0028 [-0.0128, +0.0076] | -0.0006 [-0.0123, +0.0114] | 1.000 |
| BL_H4b2+N2 | +0.0065 [-0.0037, +0.0169] | -0.0027 [-0.0115, +0.0067] | -0.0026 [-0.0140, +0.0088] | 1.000 |
| BL_H4b2+N2r | +0.0071 [-0.0027, +0.0175] | -0.0030 [-0.0119, +0.0064] | -0.0030 [-0.0143, +0.0088] | 1.000 |
| N1r | +0.0017 [-0.0136, +0.0166] | -0.0056 [-0.0197, +0.0091] | +0.0018 [-0.0143, +0.0178] | 1.000 |
| H1b_catboost | +0.0043 [-0.0113, +0.0198] | -0.0041 [-0.0172, +0.0103] | -0.0038 [-0.0209, +0.0134] | 1.000 |
| B1_catboost_A | +0.0038 [-0.0112, +0.0189] | -0.0027 [-0.0166, +0.0116] | -0.0063 [-0.0239, +0.0104] | 1.000 |
| N2r | +0.0053 [-0.0126, +0.0228] | -0.0188 [-0.0343, -0.0033] | -0.0194 [-0.0382, -0.0004] | 0.456 |
| N2 | +0.0018 [-0.0167, +0.0196] | -0.0199 [-0.0360, -0.0042] | -0.0195 [-0.0386, -0.0007] | 0.260 |

## Fit timings (seconds per fit, RTX 5060 Laptop)

|                 |   count |   mean |    min |    max |
|:----------------|--------:|-------:|-------:|-------:|
| ('N1', 'D')     |       9 | 1509.6 | 1061.3 | 1768.9 |
| ('N1', 'Drank') |       9 | 1441   | 1008.4 | 1696.3 |
| ('N2', 'D')     |       9 |  277.2 |   80.6 |  393.5 |
| ('N2', 'Drank') |       9 |  274.9 |  133.7 |  335.9 |

## Single-seed spread (model-init noise)

Each row is one seed (no averaging); the candidates above are the 3-seed mean-probability averages.

| candidate   |   seed |     sep |     oct |     nov |    mean |
|:------------|-------:|--------:|--------:|--------:|--------:|
| N1          |     42 | 0.2452  | 0.26876 | 0.23241 | 0.24879 |
| N1          |     43 | 0.23574 | 0.2583  | 0.24934 | 0.24779 |
| N1          |     44 | 0.25267 | 0.25488 | 0.25417 | 0.25391 |
| N1r         |     42 | 0.23699 | 0.25518 | 0.23806 | 0.24341 |
| N1r         |     43 | 0.24182 | 0.25722 | 0.23592 | 0.24498 |
| N1r         |     44 | 0.24492 | 0.26243 | 0.24608 | 0.25114 |
| N2          |     42 | 0.24612 | 0.24188 | 0.22861 | 0.23887 |
| N2          |     43 | 0.24507 | 0.24099 | 0.21838 | 0.23481 |
| N2          |     44 | 0.22682 | 0.24925 | 0.21293 | 0.22967 |
| N2r         |     42 | 0.24748 | 0.25017 | 0.22136 | 0.23967 |
| N2r         |     43 | 0.24543 | 0.23985 | 0.22551 | 0.23693 |
| N2r         |     44 | 0.24477 | 0.24977 | 0.2146  | 0.23638 |

| candidate   |    mean |     std |     min |     max |
|:------------|--------:|--------:|--------:|--------:|
| N1          | 0.25016 | 0.00328 | 0.24779 | 0.25391 |
| N1r         | 0.24651 | 0.00409 | 0.24341 | 0.25114 |
| N2          | 0.23445 | 0.00461 | 0.22967 | 0.23887 |
| N2r         | 0.23766 | 0.00176 | 0.23638 | 0.23967 |

## Notes and disclosures

- Pre-specified candidates: N1 (RealMLP_TD, features D), N2 (TabM_D, features D) and the three blends BL_H4b2+N1, BL_H4b2+N2, BL_H4b2+N1+N2
  (equal weights over B0, H1b_catboost and the neural member(s); the first blend therefore gives each of the three members weight 1/3).
  The `*r` rows (within-month rank of `dias_ultima_interaccion`) were added after seeing the first results and are exploratory.
- Neural seeds were 42, 43, 44 for every candidate and fold (full 3 seeds; RealMLP fits took ~18-38 min each, run as parallel processes).
  The GBDT members are single-seed (as in the repo's stored OOF); seed-averaging alone lifts the neural candidates by about
  +0.003-0.004 over their mean single-seed score, so part of N1's edge over the single-seed GBDTs is a seed-ensemble effect.
- Early stopping / best-epoch selection uses pytabkit's default random 20% hold-out drawn from the TRAINING rows only (so the scored month is
  never seen and nothing is tuned on Sep/Oct/Nov); the model never trains on that 20%. Clients repeat across months, so the hold-out is
  not client-disjoint (epoch selection may be slightly optimistic).
- Seed-to-seed std of a single N1 fit's rolling mean Gini is ~0.003-0.004, as large as the differences vs H4b2: none of the deltas versus H4b2
  excludes zero on the rolling mean (only N2/N2r are significantly worse). Holm p-values are across all candidates listed.
- Parallel processes occasionally failed at start-up (CUDA init / RAM); failed jobs were simply re-run with the same seed. GPU kernels are not bit-deterministic.
