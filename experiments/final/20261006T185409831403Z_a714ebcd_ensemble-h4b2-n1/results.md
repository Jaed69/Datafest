# Final ensemble 20261006T185409831403Z_a714ebcd_ensemble-h4b2-n1

- Model: H4b2+N1 = equal-weight 3-way mean of December percentile ranks of B0 LightGBM D, CatBoost A (`dias_ultima_interaccion` -> within-December rank) and RealMLP N1 (pytabkit, seeds 42/43/44, seed-averaged probability).
- No refit: members are the stored December test predictions listed under Sources.
- Output mapping: order-preserving quantile mapping of the blend onto the sorted B0 LightGBM December probabilities (ties share the first value of the tie group). Order, hence AUC/Gini, equals the blend's; the marginal distribution equals LightGBM's.

## Decision provenance

- Chosen over H4b2 and B0 by two blind judges (rolling mean Gini: BL_H4b2+N1 0.25432 vs H4b2 0.25245 vs B0 0.25104; deltas not significant, CI includes 0).
- Evidence: `experiments/ensembles/20261006T173753810737Z_a714ebcd_ensemble-search/results.md`
- Evidence: `experiments/ensembles/20261006T173753810737Z_a714ebcd_ensemble-search/neural/results.md`

## Gate table

| Gate | Result | Evidence |
| --- | --- | --- |
| G1 Spearman N1 vs B0 (Dec) in [0.80, 0.95] | PASS | overall 0.8561; Nov survivors (8061) 0.8673; new clients (1839) 0.8417; OOF reference 0.888 |
| G2 seeds 42/43/44 (exact set) | PASS | found [42, 43, 44]; vs B0 s42: 0.8233, s43: 0.8502, s44: 0.8457; pairwise 42-43: 0.9327, 42-44: 0.9209, 43-44: 0.9400; max abs diff mean(seeds) vs N1 column 1.49e-08 (tolerance 1e-07, float32 CSV resolution) |
| G3 N1 sanity | PASS | finite True; in [0,1] True; std 0.0563 |
| G4 final submission | PASS | rows 9900; validator True; finite True; in [0,1] True; Spearman vs current H4b2 submission 0.9877 (min 0.95) |

### G3 segment means (probability scale)

| Member | Nov survivors | New clients |
| --- | ---: | ---: |
| N1 | 0.1826 | 0.2041 |
| B0 LightGBM | 0.1415 | 0.1755 |
| CatBoost A rank | 0.1355 | 0.1644 |

## Submission

- Rows: 9900; columns id_cliente,prediccion; sample_submission order validated.
- Mean 0.14781, p10 0.06318, p50 0.14504, p90 0.22904.
- SHA-256 of submission.csv: `bfdfd1a0584f9dcd9c0530f4c144755dca621de5a8e6b495a3c47d4acc0751ba`

## Sources (SHA-256 recorded in manifest.json)

- H4b2 member predictions: `experiments\final\20261006T101910790713Z_a714ebcd_ensemble-h4b2\member_predictions.csv`
- Neural N1 test predictions: `experiments\ensembles\20261006T173753810737Z_a714ebcd_ensemble-search\neural\test_neural.csv`
- Neural per-seed predictions (copied): `experiments\final\20261006T185409831403Z_a714ebcd_ensemble-h4b2-n1\neural_seeds`
- Compared with: `experiments/final/20261006T101910790713Z_a714ebcd_ensemble-h4b2/submission.csv`
