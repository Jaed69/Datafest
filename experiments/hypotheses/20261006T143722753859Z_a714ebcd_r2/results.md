# Round-2 hypothesis evaluation `20261006T143722753859Z_a714ebcd_r2`

Rolling folds: train <=Aug->Sep, <=Sep->Oct, <=Oct->Nov (reported). Selection used only inner folds with validation months <= August (train <=Apr->May ... <=Jul->Aug). LightGBM 58 / CatBoost 192 iterations frozen for feature add-ons; XGBoost 100 rounds (median early-stopped rounds on inner folds [106, 4, 95, 216]). Paired client bootstrap (2000 resamples, shared draw across months and models) against B0 (LightGBM D) and H4b2 (rank-average champion); p = Holm-adjusted two-sided p over all candidates, per baseline.

B0 reproduction: mean Gini differs from the ablation reference 0.25104430 by +0.00e+00.

| Candidate | Sep | Oct | Nov | Mean | Std | Inner<=Aug | dMean vs B0 [95% CI] p Holm | dMean vs H4b2 [95% CI] p Holm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| E1_rank_inner_selected | 0.24876 | 0.26897 | 0.24273 | 0.25349 | 0.01122 | - | +0.0024 [-0.0022, +0.0071] p=1.000 | +0.0010 [-0.0017, +0.0037] p=1.000 |
| H4b2_rank_lgbm_catboost | 0.24715 | 0.26804 | 0.24216 | 0.25245 | 0.01121 | - | +0.0014 [-0.0032, +0.0059] p=1.000 | - |
| cat_A_H1b+F1_calendar | 0.25068 | 0.26502 | 0.23796 | 0.25122 | 0.01105 | 0.25976 | +0.0002 [-0.0089, +0.0092] p=1.000 | -0.0012 [-0.0073, +0.0046] p=1.000 |
| B0_lgbm_D | 0.24337 | 0.26773 | 0.24203 | 0.25104 | 0.01181 | 0.26298 | - | -0.0014 [-0.0059, +0.0032] p=1.000 |
| E3_rank_tuned_plus_xgb | 0.24692 | 0.27026 | 0.23477 | 0.25065 | 0.01473 | - | -0.0004 [-0.0080, +0.0073] p=1.000 | -0.0018 [-0.0066, +0.0030] p=1.000 |
| lgbm_D+F1_calendar | 0.24532 | 0.26913 | 0.23728 | 0.25058 | 0.01352 | 0.26250 | -0.0005 [-0.0050, +0.0041] p=1.000 | -0.0019 [-0.0074, +0.0038] p=1.000 |
| cat_A_H1b+UNION | 0.24744 | 0.26670 | 0.23596 | 0.25004 | 0.01268 | 0.25920 | -0.0010 [-0.0102, +0.0084] p=1.000 | -0.0024 [-0.0086, +0.0037] p=1.000 |
| cat_A_H1b+F1g_calendar_flags | 0.24744 | 0.26670 | 0.23596 | 0.25004 | 0.01268 | 0.25920 | -0.0010 [-0.0102, +0.0084] p=1.000 | -0.0024 [-0.0086, +0.0037] p=1.000 |
| lgbm_D+F2_ratios | 0.24130 | 0.26815 | 0.24061 | 0.25002 | 0.01282 | 0.26107 | -0.0010 [-0.0078, +0.0057] p=1.000 | -0.0024 [-0.0092, +0.0042] p=1.000 |
| E2_rank_tuned | 0.24770 | 0.26732 | 0.23500 | 0.25001 | 0.01329 | - | -0.0010 [-0.0092, +0.0073] p=1.000 | -0.0024 [-0.0076, +0.0027] p=1.000 |
| H1b_catboost | 0.24770 | 0.26364 | 0.23818 | 0.24984 | 0.01050 | 0.25777 | -0.0012 [-0.0101, +0.0076] p=1.000 | -0.0026 [-0.0071, +0.0020] p=1.000 |
| tuned_cat_A_H1b+F1_calendar | 0.24597 | 0.26405 | 0.23801 | 0.24934 | 0.01089 | 0.26146 | -0.0017 [-0.0103, +0.0070] p=1.000 | -0.0031 [-0.0088, +0.0026] p=1.000 |
| xgb_D | 0.24289 | 0.27173 | 0.23259 | 0.24907 | 0.01657 | 0.26422 | -0.0020 [-0.0101, +0.0062] p=1.000 | -0.0034 [-0.0098, +0.0033] p=1.000 |
| tuned_H1b_catboost | 0.24264 | 0.26393 | 0.24028 | 0.24895 | 0.01064 | 0.26026 | -0.0021 [-0.0107, +0.0065] p=1.000 | -0.0035 [-0.0093, +0.0020] p=1.000 |
| cat_A_H1b+F4_target_enc | 0.24194 | 0.26975 | 0.23427 | 0.24865 | 0.01524 | 0.25684 | -0.0024 [-0.0115, +0.0065] p=1.000 | -0.0038 [-0.0104, +0.0025] p=1.000 |
| lgbm_D+F3_rank_all | 0.24465 | 0.26507 | 0.23561 | 0.24845 | 0.01232 | 0.25851 | -0.0026 [-0.0095, +0.0037] p=1.000 | -0.0040 [-0.0098, +0.0019] p=1.000 |
| tuned_B0_lgbm_D | 0.24746 | 0.26715 | 0.22990 | 0.24817 | 0.01521 | 0.26074 | -0.0029 [-0.0113, +0.0055] p=1.000 | -0.0043 [-0.0106, +0.0022] p=1.000 |
| xgb_D+F1g_calendar_flags | 0.23881 | 0.27128 | 0.23428 | 0.24813 | 0.01648 | 0.26046 | -0.0029 [-0.0110, +0.0051] p=1.000 | -0.0043 [-0.0108, +0.0023] p=1.000 |
| lgbm_D+F1g_calendar_flags | 0.23956 | 0.26762 | 0.23594 | 0.24771 | 0.01416 | 0.26232 | -0.0033 [-0.0081, +0.0016] p=1.000 | -0.0047 [-0.0104, +0.0011] p=1.000 |
| lgbm_D+F5_cohort | 0.24346 | 0.26777 | 0.23154 | 0.24759 | 0.01508 | 0.26016 | -0.0035 [-0.0111, +0.0038] p=1.000 | -0.0049 [-0.0119, +0.0020] p=1.000 |
| xgb_D+F2_ratios | 0.24787 | 0.26617 | 0.22790 | 0.24731 | 0.01563 | 0.26193 | -0.0037 [-0.0118, +0.0043] p=1.000 | -0.0051 [-0.0116, +0.0017] p=1.000 |
| xgb_D+F3r_rank_replace | 0.24630 | 0.26779 | 0.22741 | 0.24716 | 0.01650 | 0.26092 | -0.0039 [-0.0122, +0.0047] p=1.000 | -0.0053 [-0.0121, +0.0017] p=1.000 |
| lgbm_D+F3r_rank_replace | 0.24890 | 0.26034 | 0.23156 | 0.24693 | 0.01183 | 0.26035 | -0.0041 [-0.0114, +0.0031] p=1.000 | -0.0055 [-0.0119, +0.0010] p=1.000 |
| cat_A_H1b+F3_rank_all | 0.24746 | 0.26175 | 0.23097 | 0.24673 | 0.01258 | 0.25592 | -0.0043 [-0.0135, +0.0052] p=1.000 | -0.0057 [-0.0120, +0.0005] p=1.000 |
| cat_A_H1b+F2_ratios | 0.24107 | 0.26438 | 0.23471 | 0.24672 | 0.01276 | 0.25732 | -0.0043 [-0.0140, +0.0049] p=1.000 | -0.0057 [-0.0120, +0.0004] p=1.000 |
| xgb_D+F1_calendar | 0.23937 | 0.27027 | 0.22906 | 0.24623 | 0.01751 | 0.26135 | -0.0048 [-0.0124, +0.0028] p=1.000 | -0.0062 [-0.0125, +0.0001] p=1.000 |
| xgb_D+F3_rank_all | 0.24237 | 0.27242 | 0.22375 | 0.24618 | 0.02005 | 0.26301 | -0.0049 [-0.0130, +0.0032] p=1.000 | -0.0063 [-0.0126, +0.0003] p=1.000 |
| lgbm_D+F4_target_enc | 0.23919 | 0.26813 | 0.23016 | 0.24583 | 0.01620 | 0.26214 | -0.0052 [-0.0125, +0.0018] p=1.000 | -0.0066 [-0.0135, +0.0002] p=1.000 |
| xgb_D+F5_cohort | 0.23895 | 0.27041 | 0.22659 | 0.24532 | 0.01844 | 0.25924 | -0.0057 [-0.0141, +0.0023] p=1.000 | -0.0071 [-0.0140, +0.0001] p=1.000 |
| cat_A_H1b+F5_cohort | 0.24387 | 0.26324 | 0.22852 | 0.24521 | 0.01421 | 0.25593 | -0.0058 [-0.0148, +0.0037] p=1.000 | -0.0072 [-0.0134, -0.0008] p=0.705 |
| cat_A_H1b+F3r_rank_replace | 0.24828 | 0.25866 | 0.22694 | 0.24463 | 0.01321 | 0.25417 | -0.0064 [-0.0161, +0.0034] p=1.000 | -0.0078 [-0.0146, -0.0010] p=0.705 |
| xgb_D+F4_target_enc | 0.23656 | 0.26936 | 0.22682 | 0.24424 | 0.01820 | 0.26321 | -0.0068 [-0.0148, +0.0014] p=1.000 | -0.0082 [-0.0146, -0.0019] p=0.341 |

## Interpretation

- Best rolling mean: `E1_rank_inner_selected` (0.25349).
- Significantly better than B0 after Holm: none.
- Significantly better than H4b2 after Holm: none.
- Inner-fold (<=Aug) picks: {'lgbm_D': 'B0_lgbm_D', 'cat_A_H1b': 'cat_A_H1b+F1_calendar', 'xgb_D': 'xgb_D'}. Helper unions: {'lgbm_D': [], 'cat_A_H1b': ['F1g_calendar_flags'], 'xgb_D': []}.
- Optuna: `tuned_B0_lgbm_D` objective(Jun-Aug) 0.26074 vs base config 0.25792; `tuned_H1b_catboost` objective(Jun-Aug) 0.26026 vs base config 0.25403; `tuned_cat_A_H1b+F1_calendar` objective(Jun-Aug) 0.26146 vs base config 0.25500. The Optuna objective is the search target itself, so it is optimistic.
- `gratificacion_month` (Jul, Dec) cannot be validated: no gratificacion month exists in Sep-Nov. `cts_month` is exercised only by November, with May as the sole training example. Compare F1 with F1g.
- Selection bias: 32 candidates were scored on Sep-Nov and ranked by their mean; the top rolling mean is the maximum of many noisy estimates and is optimistic. Selections themselves (which feature set, which helpers, XGBoost rounds, Optuna configs) used inner folds <= August only. Holm over the whole family is the honest significance read. Blend members are picked by inner-fold means, not by Sep-Nov.
