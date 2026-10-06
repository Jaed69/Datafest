import numpy as np
import optuna
import pandas as pd
import pytest

from datafest import round2
from datafest.hypotheses import Spec, fold_ginis, predict_folds
from datafest.models import fit_model
from datafest.round2 import (
    INNER_END,
    candidate_specs,
    inner_cuts,
    select_by_inner,
    suggest_params,
    tune_model,
    union_of_helpers,
)
from datafest.splitting import make_inner_temporal_folds

MONTHS = [202601, 202602, 202603, 202604, 202605, 202606, 202607, 202608, 202609, 202610, 202611]


def _synthetic(rows_per_month=250, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for month in MONTHS:
        a = rng.normal(size=rows_per_month)
        b = rng.normal(size=rows_per_month)
        y = (a + 0.5 * rng.normal(size=rows_per_month) > 0.8).astype(int)
        parts.append(pd.DataFrame({"id_cliente": np.arange(rows_per_month) + len(parts) * 10_000, "mes": month,
                                   "a": a, "b": b, "objetivo": y}))
    return pd.concat(parts, ignore_index=True)


def test_inner_cuts_never_reach_september_and_match_the_existing_inner_folds():
    frame = _synthetic(rows_per_month=10)
    cuts = inner_cuts(MONTHS)
    assert [valid for _, valid in cuts] == [202605, 202606, 202607, 202608]
    assert max(valid for _, valid in cuts) <= INNER_END < 202609
    assert all(train < valid for train, valid in cuts)
    folds = make_inner_temporal_folds(frame, end_month=INNER_END)
    assert cuts == tuple((f.train_through, f.valid_month) for f in folds)


def test_inner_cuts_ignore_months_after_the_end_month():
    assert inner_cuts(MONTHS + [202612]) == inner_cuts(MONTHS)


def test_predict_folds_honours_custom_cuts_and_iteration_override():
    frame = _synthetic()
    matrix = frame[["a", "b"]]
    spec = Spec("t", "x", "d", model="lightgbm", features="m", iterations=7)
    cuts = inner_cuts(MONTHS)
    out = predict_folds(spec, matrix, frame, cuts=cuts)
    assert sorted(out["mes"].unique()) == [202605, 202606, 202607, 202608]
    assert len(out) == int(frame["mes"].isin([202605, 202606, 202607, 202608]).sum())
    assert all(g > 0.2 for g in fold_ginis(out))


def test_fit_model_supports_xgboost_with_categoricals_and_early_stopping():
    rng = np.random.default_rng(1)
    n = 600
    x = pd.DataFrame({"a": rng.normal(size=n), "region": rng.choice(["w", "e", "n"], size=n)})
    y = ((x["a"] + (x["region"] == "w") * 0.8 + rng.normal(size=n)) > 0.7).astype(int)
    model = fit_model("xgboost", x[:400], y[:400], valid_x=x[400:], valid_y=y[400:], iterations=200,
                      early_stopping_rounds=10)
    assert 1 <= model.best_iteration <= 200
    prediction = model.predict_proba(x[400:])
    assert prediction.shape == (200,) and ((prediction >= 0) & (prediction <= 1)).all()


def test_suggest_params_stay_inside_declared_ranges_for_every_model():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seen = {}
    for model in ("lightgbm", "catboost"):
        draws = []
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        study.optimize(lambda trial, m=model, d=draws: d.append(suggest_params(m, trial)) or 0.0, n_trials=25)
        seen[model] = draws
    for config, iterations in seen["lightgbm"]:
        assert 8 <= config["num_leaves"] <= 128 and 20 <= iterations <= 400
        assert 0.01 <= config["learning_rate"] <= 0.2 and 0.5 <= config["subsample"] <= 1.0
    for config, iterations in seen["catboost"]:
        assert 4 <= config["depth"] <= 8 and 100 <= iterations <= 500
        assert 0.02 <= config["learning_rate"] <= 0.2


def test_tuning_only_scores_inner_folds_up_to_august(monkeypatch):
    frame = _synthetic()
    labels = frame["objetivo"].astype(float)
    labels[frame["mes"].ge(202609)] = np.nan  # Sep-Nov labels are unavailable to the search
    frame = frame.assign(objetivo=labels)
    matrix = frame[["a", "b"]]
    seen_cuts = []
    real = round2.predict_folds

    def spy(spec, mat, frm, cuts=None):
        seen_cuts.append(tuple(cuts))
        return real(spec, mat, frm, cuts=cuts)

    monkeypatch.setattr(round2, "predict_folds", spy)
    base = Spec("t", "x", "d", model="lightgbm", features="m")
    cuts = inner_cuts(MONTHS)[1:]
    result = tune_model("lightgbm", base, matrix, frame, n_trials=4, seed=3, cuts=cuts)
    assert len(result.trials) == 4
    assert seen_cuts and all(c == cuts for c in seen_cuts)
    assert max(valid for c in seen_cuts for _, valid in c) <= INNER_END
    assert result.best_value == pytest.approx(result.trials["value"].max())
    assert 20 <= result.iterations <= 400


def test_tuning_is_deterministic_given_the_seed():
    frame = _synthetic()
    matrix = frame[["a", "b"]]
    base = Spec("t", "x", "d", model="lightgbm", features="m")
    cuts = inner_cuts(MONTHS)[1:]
    first = tune_model("lightgbm", base, matrix, frame, n_trials=3, seed=5, cuts=cuts)
    second = tune_model("lightgbm", base, matrix, frame, n_trials=3, seed=5, cuts=cuts)
    assert first.config_extra == second.config_extra and first.best_value == second.best_value


def test_select_by_inner_and_union_of_helpers_use_only_inner_scores():
    inner = {"base": 0.20, "F1": 0.21, "F2": 0.19, "F3": 0.205}
    assert select_by_inner(inner, ["base", "F1", "F2", "F3"]) == "F1"
    assert union_of_helpers(inner, "base", ["F1", "F2", "F3"]) == ["F1", "F3"]
    assert union_of_helpers({"base": 0.3, "F1": 0.1}, "base", ["F1"]) == []


def test_candidate_specs_cover_each_family_for_both_gradient_boosters():
    specs = candidate_specs()
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    for family in ("F1_calendar", "F1g_calendar_flags", "F2_ratios", "F3_rank_all", "F4_target_enc", "F5_cohort"):
        assert f"lgbm_D+{family}" in names and f"cat_A_H1b+{family}" in names
    assert "B0_lgbm_D" in names and "H1b_catboost" in names
    cat = next(s for s in specs if s.name == "cat_A_H1b+F2_ratios")
    assert cat.model == "catboost" and cat.transform == "rank_interaccion"
    replace = next(s for s in specs if s.name == "cat_A_H1b+F3r_rank_replace")
    assert replace.transform == "none"


def _client_frame(months=(202601, 202602, 202603), n=40, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for month in months:
        rows.append(pd.DataFrame({
            "id_cliente": np.arange(n), "mes": month, "edad": rng.integers(20, 70, n),
            "ingresos": rng.uniform(2e4, 1e5, n), "ratio_deuda_ingresos": rng.uniform(0.05, 0.9, n),
            "antiguedad_cuenta_meses": rng.integers(1, 150, n), "numero_productos": rng.integers(1, 5, n),
            "saldo_promedio": rng.uniform(300, 9e4, n), "dias_ultima_transaccion": rng.integers(0, 300, n),
            "antiguedad_direccion_meses": rng.integers(1, 200, n), "visitas_web_ultimos_90_dias": rng.integers(0, 20, n),
            "distancia_sucursal_km": rng.uniform(0, 30, n), "dia_preferido_pago": rng.integers(1, 29, n),
            "dias_ultima_interaccion": rng.integers(0, 300, n),
            "ocupacion": rng.choice(list("abcde"), n), "region": rng.choice(list("vwxyz"), n),
            "canal_adquisicion": rng.choice(list("abc"), n), "banda_riesgo": rng.choice(["low", "high"], n),
            "dispositivo_principal": rng.choice(["ios", "android"], n), "objetivo": rng.integers(0, 2, n),
        }))
    return pd.concat(rows, ignore_index=True)


def test_apply_families_keeps_rows_aligned_and_replace_family_drops_raw_numerics():
    frame = _client_frame()
    base = frame[["edad", "ingresos", "dias_ultima_interaccion", "ocupacion"]].copy()
    for family in round2.FAMILY_NAMES:
        out = round2.apply_families([family], frame, base)
        assert len(out) == len(frame) and out.index.equals(pd.RangeIndex(len(frame)))
        assert out.notna().all().all() or family == "F4_target_enc"
    replaced = round2.apply_families(["F3r_rank_replace"], frame, base)
    assert "edad" not in replaced.columns and "rk_edad" in replaced.columns and "ocupacion" in replaced.columns


def test_union_deduplicates_overlapping_families():
    frame = _client_frame()
    base = frame[["edad"]].copy()
    out = round2.apply_families(["F1g_calendar_flags", "F1_calendar"], frame, base)
    assert out.columns.is_unique


def test_target_encoding_family_is_nan_in_first_month_and_causal_in_the_matrix():
    frame = _client_frame()
    base = frame[["edad"]].copy()
    first = round2.apply_families(["F4_target_enc"], frame, base)
    te_cols = [c for c in first.columns if c.startswith("te_")]
    assert len(te_cols) == 15
    assert first.loc[frame["mes"].eq(202601), te_cols].isna().all().all()
    flipped = frame.assign(objetivo=np.where(frame["mes"].ge(202603), 1 - frame["objetivo"], frame["objetivo"]))
    second = round2.apply_families(["F4_target_enc"], flipped, base)
    pd.testing.assert_frame_equal(first, second)  # flipping March (own month) labels changes no encoding


def test_compare_to_baselines_reports_both_baselines_without_self_rows():
    frame = _synthetic(rows_per_month=120, seed=4)
    folds = frame[frame["mes"].isin([202609, 202610, 202611])][["id_cliente", "mes", "objetivo"]].reset_index(drop=True)
    rng = np.random.default_rng(0)
    results = {}
    for name, noise in (("B0_lgbm_D", 1.0), ("H4b2_rank_lgbm_catboost", 0.8), ("cand", 0.5)):
        score = folds["objetivo"] * 0.5 + rng.normal(scale=noise, size=len(folds))
        results[name] = folds.assign(prediccion=1 / (1 + np.exp(-score)))
    table = round2.compare_to_baselines(results, ("B0_lgbm_D", "H4b2_rank_lgbm_catboost"), replicates=30, seed=1)
    assert set(table["candidate"]) == set(results)
    by_name = table.set_index("candidate")
    assert np.isnan(by_name.loc["B0_lgbm_D", "delta_mean_b0"]) and not np.isnan(by_name.loc["B0_lgbm_D", "delta_mean_h4"])
    assert np.isnan(by_name.loc["H4b2_rank_lgbm_catboost", "delta_mean_h4"])
    assert not np.isnan(by_name.loc["cand", ["delta_mean_b0", "delta_mean_h4", "p_holm_b0", "p_holm_h4"]]).any()
