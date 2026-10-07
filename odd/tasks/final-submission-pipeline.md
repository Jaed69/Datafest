# Feature: final-submission-pipeline

## Objective
Produce one validated December 2026 submission (`id_cliente,prediccion`) for the BCP Datafest #CodingChallenge (metric Gini = 2*AUC-1, deadline 2026-10-07), then improve it through hypothesis-driven experiments.

## Problem
No pipeline fits the best validated model (LightGBM D, rolling mean Gini 0.25104) on Jan-Nov 2026 and scores `data/test.csv`. Existing `submission.csv` files come from CatBoost runs trained only up to September.

## Why
Only one final solution per team is accepted. A safe, reproducible baseline submission must exist before further experimentation.

## Scope
- Final-fit pipeline: train on all Jan-Nov rows, score December, validate the submission format.
- Hypothesis experiments on the existing rolling harness (Sep/Oct/Nov) with paired client bootstrap.
- Final model selection and submission.

## Constraints
- `NON_NEGOTIABLE.md`: strict temporal order, no random splits, raw `id_cliente` only diagnostic, SHA-256 manifests per run.
- Python 3.12 via `uv`; existing pinned dependencies.
- Strict TDD: RED -> GREEN -> REFACTOR. Runner: `uv run pytest -q`. Source: user global CLAUDE.md ("Strict TDD Mode: enabled").

## Key data facts
- Discrete-time hazard: `objetivo=1` is first conversion; client leaves afterwards. Prevalence ~15%.
- Almost all features static per client; only `dias_ultima_interaccion` varies over time.
- `dias_ultima_interaccion` vs `dias_ultima_transaccion`: equal in 100% of Jan rows, 9.4% Nov, 0.26% test (corr 0.55 train vs -0.003 test). Structural drift.
- Test = 8,061 Nov survivors + 1,839 new clients.

## Tasks
- [x] T0 Final-fit pipeline + validated baseline submission (LightGBM D and CatBoost A fit Jan-Nov -> December). Route: delegated (writer trigger: 2+ non-trivial files).
  - Evidence: commits b715eb4 (submissions), dc30616 (pipeline). `uv run pytest -q tests/test_final_fit.py`: 14 passed (RED: ModuleNotFoundError first). Full suite: 44 passed, 1 pre-existing Windows failure (`test_diagnostics::...SIGALRM`).
  - Submissions validated (9,900 rows, sample order, [0,1], no NaN): LightGBM D mean 0.148; CatBoost A mean 0.141; Spearman 0.859.
  - Review: first combined candidate hit `lens_context_budget_exceeded` (CSV artifacts); split into artifacts + code commits. Code slice assessed medium, `under_budget` (379 lines) -> pending in slice.
- [x] T1 H1 drift: drop / per-month rank of `dias_ultima_interaccion` vs keep.
- [x] T2 H2 recency: time-decay sample weights / shorter train window.
- [x] T3 H3 robustness: WoE logistic baseline; monotone constraints.
- [x] T4 H4 ensemble: rank-average LGBM + CatBoost + logistic with seed bagging.
  - Route: delegated (writer trigger). Evidence: commits 5495620 (code), bfc451d (results). `uv run pytest -q`: 55 passed, 1 pre-existing Windows failure. RED: ModuleNotFoundError first; GREEN 11 passed.
  - Run: experiments/hypotheses/20261006T100749205414Z_a714ebcd (2000 bootstrap, Holm over 17). B0 reproduced exactly (0.25104).
  - Result: no candidate beats B0 significantly. Best H4b2 rank(LGBM D + CatBoost H1b) 0.25245 (+0.0014, CI [-0.0032,+0.0059]). Significantly worse: dropping both dias_ultima_* (-0.022/-0.026), logistic WoE (-0.038). Drift handling neutral for CatBoost, -0.005 for LGBM. Recency weights, monotone, seed bagging: no gain.
  - Review: native review (reliability lens) over b715eb4..bfc451d approved and acknowledged (lineage review-11d03d7a1c4314e6, authority burned). Advisory warnings (non-blocking): bootstrap alignment assumption (verified: OOF is one wide table, rows aligned), bootstrap untested, final_fit partial artifacts on failure, logistic warnings swallowed, tautological test at tests/test_hypotheses.py:51.
- [x] T5 Blind dual judging of candidates (mean Gini, stability, bootstrap significance).
  - Route: two independent read-only judges (jd-judge-a/b). Both chose H4b2 (rank-average LGBM D + CatBoost A with within-month rank of dias_ultima_interaccion). Rationale: best mean (0.25245) and lowest std, drift diversification, already implemented. Gain vs B0 is noise (CI spans 0) -> chosen for robustness, B0 kept as fallback. Kumo blend rejected (no bootstrap, no Jan-Nov fit, GPU). No extra members (all extra variants hurt Nov).
- [x] T6 Final fit of the winner + submission checks.
  - Route: delegated (writer trigger). `uv run pytest -q`: 59 passed, 1 pre-existing Windows failure. RED: ModuleNotFoundError first.
  - Output order-preserving quantile mapping of the blend onto LightGBM December probabilities (keeps blend AUC, values in [0,1]).
  - Final submission: experiments/final/20261006T101910790713Z_a714ebcd_ensemble-h4b2/submission.csv (9,900 rows, sample order, [0,1], no NaN, mean 0.148).
  - Checks: Spearman vs B0 0.963 overall, 0.963 Nov survivors, 0.960 new clients. Mean score survivors 0.142, new clients 0.174.

### Round 2 (user request 2026-10-06: push for best Gini with new variables, seasonality, advanced models, GPU)
Local GPU observed: RTX 5060 Laptop 8 GB (user reports an RTX 5080 elsewhere).
- [x] T7 Feature candidates on the rolling harness: pay-day calendar features (weekday/weekend/Peruvian holiday of `dia_preferido_pago` in the month), ratios/interactions, within-month ranks of all numerics, time-aware out-of-fold target encoding. Route: delegated.
- [x] T8 GBDT upgrades: XGBoost (GPU), Optuna tuning of LightGBM/CatBoost restricted to inner temporal folds <= August. Route: delegated (same writer as T7).
  - Evidence: commits 3d6d7a7 (results), 113607e (code). 88 passed + 1 known Windows failure. Run experiments/hypotheses/20261006T143722753859Z_a714ebcd_r2 (32 candidates, bootstrap vs B0 and H4b2).
  - Result: nothing beats B0 or H4b2. F1 calendar neutral; gratificacion/CTS flags hurt (unvalidatable for December); ratios, within-month ranks, causal TE, cohort features and XGBoost 0.002-0.008 below. Optuna overfits inner folds (tuned LGBM 0.24817 < B0 0.25104). Best new: E1 rank(B0 + CatBoost+F1) 0.25349 (+0.0010 vs H4b2, CI spans 0).
  - Review: approved and acknowledged (lineage review-8bc39e7853389466). Advisory: cache key omits features.py/ablation.py and library versions (verified harmless here: files unchanged, B0 reproduced exactly), nonatomic cache write, duplicate union candidate, unguarded ratio.
- [x] T9 Neural models on GPU. Route: delegated, out-of-tree venv (scratchpad/neural) to avoid concurrent writers; torch 2.11.0+cu128 on RTX 5060 (sm_120 verified).
  - N1 RealMLP-TD (3 seeds) 0.25369, Spearman 0.888 vs B0, lowest month std (0.0055). N2 TabM 0.23852 (significantly worse, rejected). H4b2+N1 0.25432 (+0.0019 vs H4b2, CI [-0.0006,+0.0043]). Seed noise of N1 ~0.003. December N1/N2 predictions produced (~38 min/seed final fit).
- [x] T10 Ensemble search with selection-bias control (leave-one-month-out selection estimate).
  - Evidence: commits b75691b (run), 8ddc4db (code). 102 passed + 1 known failure. Run experiments/ensembles/20261006T173753810737Z_a714ebcd_ensemble-search.
  - Fixed blends with N1: +0.0019 to +0.0024 vs H4b2 (CIs span 0, Holm p=1). Selection-aware: best-of-list LOMO -0.0002, LOMO stack -0.0020 (unstable weights). N1 blends lower std (~0.009) and better Nov.
  - Review: 4-lens native review approved and acknowledged (lineage review-938c41ce8b46a0ec). Advisory: partial-seed averaging in neural scripts (gated in T12), hardcoded local paths in scripts/neural, delta-table/global-month warnings.
- [x] T11 Blind dual judging of round-2 candidates vs H4b2.
  - Both judges independently chose H4b2+N1 (equal 3-way rank average): a-priori fixed blend (little selection optimism), lower variance, better Nov, 2/3 validated GBDT members dilute neural risk, bounded downside (Dec Spearman 0.988 vs H4b2). Fallback H4b2 if December gates fail.
- [x] T12 Final fit of the round-2 winner + submission checks.
  - Route: delegated (writer trigger). 111 passed + 1 known failure; 9 new tests (RED ModuleNotFoundError first).
  - Gates (all PASS): G1 N1 vs B0 Dec Spearman 0.856 (survivors 0.867, new 0.842); G2 exactly seeds 42/43/44 (mean reproduced to 1.5e-8), seed vs B0 0.823-0.850, pairwise 0.921-0.940; G3 N1 sane (std 0.056); G4 validator OK, Spearman vs H4b2 submission 0.9877.
  - FINAL submission: experiments/final/20261006T185409831403Z_a714ebcd_ensemble-h4b2-n1/submission.csv, SHA-256 bfdfd1a0584f9dcd9c0530f4c144755dca621de5a8e6b495a3c47d4acc0751ba (independently re-validated).
  - Expected December Gini ~0.25 (rolling 0.25432); gain vs H4b2 is a variance-reduction bet, not a proven improvement.

## Acceptance criteria
- Submission has exactly 9,900 rows, columns `id_cliente,prediccion`, same order as `data/sample_submission.csv`, predictions finite in [0,1].
- Each experiment reports rolling Gini per fold, mean, and a paired bootstrap delta vs baseline.

## Delivery
Strategy: ask-on-risk. Forecast: ~300-600 authored lines over all tasks.

## Progress
- 2026-10-06: branch `feat/final-submission-pipeline` created; diagnosis done.

- 2026-10-06: T0 done; baseline submissions available as fallback.

- 2026-10-06: T1-T6 done. Final submission ready.
- 2026-10-06: Round 2 (T7-T12) done. New final: H4b2+N1. Fallbacks: H4b2, then B0.

## Next step
Team decides the single representative who emails the final submission to marcaempleadora@bcp.com.pe before 2026-10-07. Push / PR are the team's decision.
