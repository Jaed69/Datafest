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
- [ ] T0 Final-fit pipeline + validated baseline submission (LightGBM D and CatBoost A fit Jan-Nov -> December). Route: delegated (writer trigger: 2+ non-trivial files).
- [ ] T1 H1 drift: drop / per-month rank of `dias_ultima_interaccion` vs keep.
- [ ] T2 H2 recency: time-decay sample weights / shorter train window.
- [ ] T3 H3 robustness: WoE logistic baseline; monotone constraints.
- [ ] T4 H4 ensemble: rank-average LGBM + CatBoost + logistic with seed bagging.
- [ ] T5 Blind dual judging of candidates (mean Gini, stability, bootstrap significance).
- [ ] T6 Final fit of the winner + submission checks.

## Acceptance criteria
- Submission has exactly 9,900 rows, columns `id_cliente,prediccion`, same order as `data/sample_submission.csv`, predictions finite in [0,1].
- Each experiment reports rolling Gini per fold, mean, and a paired bootstrap delta vs baseline.

## Delivery
Strategy: ask-on-risk. Forecast: ~300-600 authored lines over all tasks.

## Progress
- 2026-10-06: branch `feat/final-submission-pipeline` created; diagnosis done.

## Next step
T0.
