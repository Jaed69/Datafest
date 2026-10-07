import numpy as np
import pandas as pd
import pytest

from datafest.round2_features import (
    behavioral_ratio_features,
    causal_target_encoding,
    cohort_features,
    pair_keys,
    within_month_rank_features,
)


def _frame():
    return pd.DataFrame({
        "mes": [202601, 202601, 202601, 202602, 202602, 202603, 202603],
        "g": ["a", "a", "b", "a", "b", "a", "b"],
        "y": [1, 0, 0, 1, 0, 0, 1],
    })


def test_target_encoding_first_month_is_nan_and_later_months_use_prior_months_only():
    frame = _frame()
    out = causal_target_encoding(frame["mes"], frame["y"], frame[["g"]], smoothing=2.0)
    col = out["te_g"]
    assert col[:3].isna().all()
    # Feb, group a: prior months -> a has 2 rows, 1 positive; prior mean = 1/3.
    prior = 1 / 3
    assert col[3] == pytest.approx((1 + 2 * prior) / (2 + 2))
    # Feb, group b: 1 row, 0 positives.
    assert col[4] == pytest.approx((0 + 2 * prior) / (1 + 2))
    # Mar, group a: Jan+Feb a rows = 3, positives 2; prior mean = 2/5.
    assert col[5] == pytest.approx((2 + 2 * (2 / 5)) / (3 + 2))


def test_target_encoding_never_uses_own_or_later_month_labels():
    rng = np.random.default_rng(0)
    months = np.repeat([202601, 202602, 202603, 202604], 50)
    keys = pd.DataFrame({"g": rng.choice(list("abc"), size=200), "h": rng.choice(list("xy"), size=200)})
    y = pd.Series(rng.integers(0, 2, size=200))
    base = causal_target_encoding(pd.Series(months), y, keys, smoothing=5.0)
    for cut in (202602, 202603, 202604):
        changed = y.copy()
        changed[months >= cut] = 1 - changed[months >= cut]  # flip own and later labels
        other = causal_target_encoding(pd.Series(months), changed, keys, smoothing=5.0)
        earlier_or_equal = months <= cut
        pd.testing.assert_frame_equal(base[earlier_or_equal], other[earlier_or_equal])


def test_target_encoding_unknown_labels_in_last_month_are_ignored():
    frame = _frame()
    labels = frame["y"].astype(float)
    labels[frame["mes"].eq(202603)] = np.nan  # unlabeled (test-like) month
    out = causal_target_encoding(frame["mes"], labels, frame[["g"]], smoothing=2.0)
    reference = causal_target_encoding(frame["mes"], frame["y"], frame[["g"]], smoothing=2.0)
    pd.testing.assert_series_equal(out["te_g"], reference["te_g"])


def test_pair_keys_builds_all_two_way_combinations():
    frame = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "c": ["m", "n"]})
    keys = pair_keys(frame, ["a", "b", "c"])
    assert list(keys.columns) == ["a", "b", "c", "a__b", "a__c", "b__c"]
    assert keys["a__b"].tolist() == ["x|p", "y|q"]


def test_within_month_rank_features_are_per_month_and_prefixed():
    frame = pd.DataFrame({"mes": [1, 1, 1, 2, 2], "v": [3.0, 1.0, 2.0, 10.0, 20.0], "w": [1, 1, 1, 5, 5]})
    out = within_month_rank_features(frame, ["v", "w"])
    assert list(out.columns) == ["rk_v", "rk_w"]
    assert out["rk_v"].tolist() == pytest.approx([1.0, 1 / 3, 2 / 3, 0.5, 1.0])
    assert out["rk_w"].tolist() == pytest.approx([2 / 3] * 3 + [0.75] * 2)  # ties -> average rank


def test_behavioral_ratios_match_hand_computation():
    frame = pd.DataFrame({
        "mes": [202601, 202601], "edad": [30, 20], "ingresos": [1000.0, 2000.0],
        "ratio_deuda_ingresos": [0.25, 0.5], "antiguedad_cuenta_meses": [23, 11],
        "numero_productos": [2, 0], "saldo_promedio": [500.0, 100.0],
        "visitas_web_ultimos_90_dias": [6, 4], "dias_ultima_transaccion": [24, 5],
        "dias_ultima_interaccion": [20, 5],
    })
    out = behavioral_ratio_features(frame)
    assert out.loc[0, "saldo_ingresos"] == pytest.approx(0.5)
    assert out.loc[0, "ingreso_neto"] == pytest.approx(750.0)
    assert out.loc[0, "antiguedad_share_vida_adulta"] == pytest.approx(23 / (12 * 12))
    assert out.loc[1, "antiguedad_share_vida_adulta"] == pytest.approx(11 / 24)
    assert out.loc[0, "productos_por_anio_antiguedad"] == pytest.approx(2 / (24 / 12))
    assert out.loc[1, "visitas_por_producto"] == pytest.approx(4.0)  # zero products guarded
    assert out.loc[0, "recencia_ratio"] == pytest.approx(24 / 24)
    # |gap| = [4, 0] -> within-month percentile ranks [1.0, 0.5]
    assert out["gap_interaccion_rank"].tolist() == pytest.approx([1.0, 0.5])
    assert np.isfinite(out.to_numpy()).all()


def test_cohort_features_rank_inside_first_seen_cohort_and_month():
    frame = pd.DataFrame({
        "id_cliente": [1, 2, 3, 1, 2, 3, 4],
        "mes": [202601, 202601, 202601, 202602, 202602, 202602, 202602],
        "v": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 5.0],
    })
    out = cohort_features(frame, ["v"])
    # Clients 1-3 are the Jan cohort; client 4 is a Feb cohort of one.
    assert out["cohort_rk_v"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0, 1 / 3, 2 / 3, 1.0, 1.0])
    assert out["cohort_survival"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert out["cohort_size_initial"].tolist() == [3, 3, 3, 3, 3, 3, 1]


def test_cohort_survival_uses_membership_up_to_the_current_month_only():
    frame = pd.DataFrame({
        "id_cliente": [1, 2, 3, 1, 2, 1],
        "mes": [202601, 202601, 202601, 202602, 202602, 202603],
        "v": [1.0] * 6,
    })
    out = cohort_features(frame, ["v"])
    assert out["cohort_survival"].tolist() == pytest.approx([1.0, 1.0, 1.0, 2 / 3, 2 / 3, 1 / 3])
    truncated = cohort_features(frame[frame["mes"].le(202602)].reset_index(drop=True), ["v"])
    assert truncated["cohort_survival"].tolist() == pytest.approx(out["cohort_survival"].tolist()[:5])
