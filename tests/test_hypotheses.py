import numpy as np
import pandas as pd
import pytest

from datafest.hypotheses import (
    holm_adjust,
    monotone_vector,
    rank_average,
    recent_window_mask,
    time_decay_weights,
    within_month_rank,
)
from datafest.models import fit_model


def test_time_decay_newest_month_has_weight_one_and_half_life_halves():
    months = pd.Series([202603, 202606, 202609, 202609])
    weights = time_decay_weights(months, half_life=3)
    assert weights.tolist() == pytest.approx([0.25, 0.5, 1.0, 1.0])


def test_time_decay_handles_year_boundary():
    weights = time_decay_weights(pd.Series([202512, 202601, 202602]), half_life=1)
    assert weights.tolist() == pytest.approx([0.25, 0.5, 1.0])


def test_time_decay_rejects_non_positive_half_life():
    with pytest.raises(ValueError):
        time_decay_weights(pd.Series([202601]), half_life=0)


def test_within_month_rank_is_bounded_and_per_month_only():
    values = pd.Series([10.0, 20.0, 30.0, 1000.0, 2000.0])
    months = pd.Series([1, 1, 1, 2, 2])
    ranks = within_month_rank(values, months)
    assert ranks.min() > 0 and ranks.max() <= 1
    assert ranks.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0, 0.5, 1.0])
    # Other months do not influence a month's ranks.
    shifted = within_month_rank(values.where(months.eq(1), values * 1e6), months)
    assert shifted[:3].tolist() == pytest.approx(ranks[:3].tolist())


def test_rank_average_per_month_and_monotone_invariant():
    months = pd.Series([1, 1, 1, 2, 2, 2])
    a = np.array([0.1, 0.5, 0.9, 0.2, 0.3, 0.1])
    b = np.array([0.9, 0.5, 0.1, 0.8, 0.9, 0.7])
    blend = rank_average([a, b], months)
    assert blend.shape == a.shape and blend.min() > 0 and blend.max() <= 1
    transformed = rank_average([np.exp(5 * a), np.log(b + 1.0)], months)
    assert transformed.tolist() == pytest.approx(blend.tolist())
    # Month 2 values never change month 1 blend.
    other = rank_average([a, b], pd.Series([1, 1, 1, 2, 2, 2]))
    assert other[:3].tolist() == pytest.approx(blend[:3].tolist())


def test_rank_average_single_member_is_its_within_month_rank():
    months = pd.Series([1, 1, 2, 2])
    a = np.array([0.3, 0.1, 0.5, 0.9])
    assert rank_average([a], months).tolist() == pytest.approx([1.0, 0.5, 0.5, 1.0])


def test_monotone_vector_aligns_with_feature_order():
    columns = ["edad", "numero_productos", "banda_riesgo_ord", "x", "dias_ultima_transaccion"]
    directions = {"banda_riesgo_ord": -1, "numero_productos": 1, "dias_ultima_transaccion": -1}
    assert monotone_vector(columns, directions) == [0, 1, -1, 0, -1]


def test_monotone_vector_fails_when_constrained_feature_missing():
    with pytest.raises(ValueError):
        monotone_vector(["edad"], {"numero_productos": 1})


def test_recent_window_mask_keeps_last_n_months_through_cut():
    months = pd.Series([202602, 202603, 202608, 202609])
    mask = recent_window_mask(months, train_through=202608, n_months=6)
    assert mask.tolist() == [False, True, True, False]


def test_holm_adjust_is_monotone_and_capped():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_fit_model_accepts_sample_weight():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = (x["a"] + rng.normal(size=200) > 0).astype(int)
    weights = np.linspace(0.1, 1.0, 200)
    for name in ("lightgbm", "catboost"):
        model = fit_model(name, x, y, iterations=5, early_stopping_rounds=None, sample_weight=weights)
        assert model.predict_proba(x).shape == (200,)
