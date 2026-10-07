import numpy as np
import pandas as pd
import pytest

from datafest.ensemble_search import (
    Candidate,
    blend_scores,
    bootstrap_ginis,
    fit_rank_weights,
    join_oof,
    lomo_best_of,
    lomo_stack,
    probability_of_best,
    simplex_grid,
)
from datafest.hypotheses import rank_average, within_month_rank

MONTHS = (202609, 202610, 202611)


def _synthetic(n_per_month=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for month in MONTHS:
        y = rng.integers(0, 2, n_per_month)
        signal = y + rng.normal(0, 1.0, n_per_month)
        rows.append(pd.DataFrame({
            "id_cliente": np.arange(n_per_month), "mes": month, "objetivo": y,
            "good": signal, "ok": y * 0.5 + rng.normal(0, 1.0, n_per_month),
            "noise": rng.normal(0, 1.0, n_per_month),
        }))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------ join
def test_join_oof_merges_on_keys_and_keeps_target():
    a = pd.DataFrame({"id_cliente": [1, 2], "mes": [1, 1], "objetivo": [0, 1], "m1": [0.1, 0.9]})
    b = pd.DataFrame({"id_cliente": [2, 1], "mes": [1, 1], "objetivo": [1, 0], "m2": [0.7, 0.3]})
    joined = join_oof([a, b])
    assert joined[["id_cliente", "m1", "m2"]].to_dict("list") == {"id_cliente": [1, 2], "m1": [0.1, 0.9], "m2": [0.3, 0.7]}
    assert joined["objetivo"].tolist() == [0, 1]


def test_join_oof_rejects_mismatched_row_sets():
    a = pd.DataFrame({"id_cliente": [1, 2], "mes": [1, 1], "objetivo": [0, 1], "m1": [0.1, 0.9]})
    b = pd.DataFrame({"id_cliente": [1, 3], "mes": [1, 1], "objetivo": [0, 1], "m2": [0.3, 0.7]})
    with pytest.raises(ValueError, match="row set"):
        join_oof([a, b])
    with pytest.raises(ValueError, match="row set"):
        join_oof([a, b.iloc[:1]])


def test_join_oof_rejects_mismatched_target_duplicates_and_column_clash():
    a = pd.DataFrame({"id_cliente": [1, 2], "mes": [1, 1], "objetivo": [0, 1], "m1": [0.1, 0.9]})
    flipped = pd.DataFrame({"id_cliente": [1, 2], "mes": [1, 1], "objetivo": [1, 1], "m2": [0.3, 0.7]})
    with pytest.raises(ValueError, match="objetivo"):
        join_oof([a, flipped])
    duplicated = pd.concat([a, a], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        join_oof([duplicated])
    clash = a.rename(columns={"m1": "m1"}).copy()
    with pytest.raises(ValueError, match="column"):
        join_oof([a, clash])


# --------------------------------------------------------------------- weights
def test_simplex_grid_rows_are_non_negative_and_sum_to_one():
    grid = simplex_grid(4, 0.25)
    assert (grid >= 0).all()
    assert grid.sum(axis=1) == pytest.approx(np.ones(len(grid)))
    assert len(grid) == 35  # C(4 + 4 - 1, 4 - 1)
    assert len({tuple(row) for row in grid.tolist()}) == len(grid)


def test_simplex_grid_requires_step_that_divides_one():
    with pytest.raises(ValueError):
        simplex_grid(3, 0.3)


def test_fit_rank_weights_is_non_negative_sums_to_one_and_prefers_informative_member():
    data = _synthetic()
    ranks = np.column_stack([within_month_rank(data[c], data["mes"]) for c in ("good", "ok", "noise")])
    weights = fit_rank_weights(ranks, data["objetivo"].to_numpy(), data["mes"].to_numpy(), step=0.1)
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[2]


def test_lomo_stack_never_uses_the_held_out_month():
    data = _synthetic()
    members = ["good", "ok", "noise"]
    baseline = lomo_stack(data, members, MONTHS, step=0.1)
    # Replace every held-out-month value (members and target) with different garbage.
    for held_out in MONTHS:
        corrupted = data.copy()
        mask = corrupted["mes"].eq(held_out)
        rng = np.random.default_rng(99)
        corrupted.loc[mask, "objetivo"] = rng.integers(0, 2, mask.sum())
        for column in members:
            corrupted.loc[mask, column] = rng.normal(size=mask.sum())
        result = lomo_stack(corrupted, members, MONTHS, step=0.1)
        assert result.weights[held_out] == pytest.approx(baseline.weights[held_out])
    # And the other folds' weights *do* react to a month they are fitted on.
    mutated = data.copy()
    mask = mutated["mes"].eq(MONTHS[0])
    mutated.loc[mask, "objetivo"] = 1 - mutated.loc[mask, "objetivo"]
    assert not np.allclose(lomo_stack(mutated, members, MONTHS, step=0.1).weights[MONTHS[1]],
                           baseline.weights[MONTHS[1]])


def test_lomo_stack_scores_cover_every_row_with_fold_weights():
    data = _synthetic()
    members = ["good", "ok", "noise"]
    result = lomo_stack(data, members, MONTHS, step=0.1)
    assert set(result.weights) == set(MONTHS)
    assert len(result.scores) == len(data)
    for month, weights in result.weights.items():
        assert weights.sum() == pytest.approx(1.0) and (weights >= 0).all()
        mask = data["mes"].eq(month).to_numpy()
        ranks = np.column_stack([within_month_rank(data.loc[mask, c], data.loc[mask, "mes"]) for c in members])
        assert result.scores[mask] == pytest.approx(ranks @ weights)
    assert set(result.gini_by_month) == set(MONTHS)


# ------------------------------------------------------------------- blending
def test_blend_scores_equal_groups_match_rank_average():
    data = _synthetic()
    candidate = Candidate("eq", ((1.0, ("good",)), (1.0, ("ok",)), (1.0, ("noise",))))
    expected = rank_average([data[c].to_numpy() for c in ("good", "ok", "noise")], data["mes"])
    assert blend_scores(data, candidate) == pytest.approx(expected)


def test_blend_scores_group_is_reranked_within_month_before_mixing():
    data = _synthetic()
    candidate = Candidate("two", ((0.5, ("good", "ok")), (0.5, ("noise",))))
    inner = rank_average([data["good"].to_numpy(), data["ok"].to_numpy()], data["mes"])
    expected = 0.5 * within_month_rank(pd.Series(inner), data["mes"]) + 0.5 * within_month_rank(data["noise"], data["mes"])
    assert blend_scores(data, candidate) == pytest.approx(expected)


def test_blend_scores_rejects_non_positive_weight_sum():
    data = _synthetic()
    with pytest.raises(ValueError):
        blend_scores(data, Candidate("bad", ((0.0, ("good",)),)))


# ------------------------------------------------------------- best-of via LOMO
def test_lomo_best_of_chooses_on_other_months_and_scores_held_out():
    table = pd.DataFrame(
        {202609: [0.30, 0.20, 0.10], 202610: [0.10, 0.25, 0.20], 202611: [0.50, 0.00, 0.40]},
        index=["a", "b", "c"],
    )
    result = lomo_best_of(table)
    # Hold out 202609: other-month means a=0.30, b=0.125, c=0.30 -> tie a/c -> first (a).
    # Hold out 202610: a=0.40, b=0.10, c=0.25 -> a. Hold out 202611: a=0.20, b=0.225, c=0.15 -> b.
    assert result.chosen == {202609: "a", 202610: "a", 202611: "b"}
    assert result.held_out_gini == {202609: 0.30, 202610: 0.10, 202611: 0.00}
    assert result.mean == pytest.approx((0.30 + 0.10 + 0.00) / 3)
    assert result.wins == {"a": 2, "b": 1, "c": 0}


def test_lomo_best_of_choice_ignores_the_held_out_month():
    table = pd.DataFrame({202609: [0.1, 0.2], 202610: [0.3, 0.1], 202611: [0.2, 0.3]}, index=["a", "b"])
    changed = table.copy()
    changed.loc["a", 202609] = 99.0  # would flip the held-out pick only if it leaked
    assert lomo_best_of(table).chosen[202609] == lomo_best_of(changed).chosen[202609]


# ------------------------------------------------------------------- bootstrap
def test_bootstrap_ginis_shape_determinism_and_probability_of_best():
    data = _synthetic(n_per_month=300)
    scores = {"good": data["good"].to_numpy(), "ok": data["ok"].to_numpy(), "noise": data["noise"].to_numpy()}
    first = bootstrap_ginis(data, scores, MONTHS, replicates=60, seed=1)
    second = bootstrap_ginis(data, scores, MONTHS, replicates=60, seed=1)
    assert first.shape == (60, 3, 3)
    assert np.array_equal(first, second)
    probabilities = probability_of_best(first, ["good", "ok", "noise"])
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["good"] > probabilities["noise"]
