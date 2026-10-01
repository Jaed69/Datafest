import pandas as pd

from datafest.features import add_history_features, features_for_variant
from datafest.splitting import (
    make_group_diagnostic_split,
    make_inner_temporal_folds,
    make_outer_temporal_split,
)


def sample_rows():
    return pd.DataFrame(
        {
            "id_cliente": [1, 2, 1, 3, 2, 4],
            "mes": [202601, 202601, 202602, 202602, 202603, 202603],
            "edad": [30, 45, 30, 26, 45, 51],
            "objetivo": [0, 0, 1, 0, 0, 0],
        }
    )


def test_outer_split_uses_only_preceding_months_for_training():
    rows = pd.DataFrame(
        {
            "id_cliente": [1, 2, 1, 3],
            "mes": [202609, 202610, 202610, 202611],
            "objetivo": [0, 0, 1, 0],
        }
    )
    train, valid = make_outer_temporal_split(rows)

    assert train["mes"].max() == 202609
    assert valid["mes"].tolist() == [202610, 202610, 202611]
    assert set(train["id_cliente"]) & set(valid["id_cliente"])


def test_inner_folds_are_expanding_and_temporal():
    rows = pd.DataFrame(
        {"id_cliente": range(1, 10), "mes": range(202601, 202610)}
    )
    folds = make_inner_temporal_folds(rows)

    assert [(fold.train_through, fold.valid_month) for fold in folds] == [
        (202605, 202606),
        (202606, 202607),
        (202607, 202608),
        (202608, 202609),
    ]
    for fold in folds:
        assert fold.train["mes"].max() < fold.valid["mes"].min()


def test_group_diagnostic_split_has_no_client_overlap():
    clients = list(range(1, 501))
    rows = pd.DataFrame(
        {
            "id_cliente": clients * 2,
            "mes": [202608] * 500 + [202609] * 500,
            "objetivo": [0, 1] * 500,
        }
    )
    train, valid = make_group_diagnostic_split(rows, holdout_month=202609)

    assert set(train["id_cliente"]).isdisjoint(valid["id_cliente"])
    assert set(valid["mes"]) == {202609}
    assert 0 < len(valid) < 500


def test_history_features_use_only_current_and_prior_rows():
    rows = sample_rows()
    baseline = add_history_features(rows)
    with_future = pd.concat(
        [
            rows,
            pd.DataFrame(
                {
                    "id_cliente": [1],
                    "mes": [202604],
                    "edad": [30],
                    "objetivo": [1],
                }
            ),
        ],
        ignore_index=True,
    )
    extended = add_history_features(with_future)

    cols = [
        "mes_primera_aparicion",
        "meses_en_riesgo",
        "n_observaciones_previas",
        "cliente_recurrente",
        "meses_desde_entrada",
    ]
    pd.testing.assert_frame_equal(
        baseline[cols].reset_index(drop=True),
        extended.iloc[: len(rows)][cols].reset_index(drop=True),
    )
    assert "objetivo" not in cols


def test_history_feature_counts_prior_observations_not_current_row():
    features = add_history_features(sample_rows())
    client_one = features.loc[features["id_cliente"].eq(1)]

    assert client_one["n_observaciones_previas"].tolist() == [0, 1]
    assert client_one["cliente_recurrente"].tolist() == [False, True]


def test_feature_variants_keep_or_remove_identifier_as_declared():
    rows = sample_rows()

    clean = features_for_variant(rows, "clean")
    history = features_for_variant(rows, "history")
    raw_id = features_for_variant(rows, "id_raw")

    assert "id_cliente" not in clean and "objetivo" not in clean
    assert set(
        [
            "mes_primera_aparicion",
            "meses_en_riesgo",
            "n_observaciones_previas",
            "cliente_recurrente",
            "meses_desde_entrada",
        ]
    ).issubset(history.columns)
    assert "id_cliente" in raw_id and "objetivo" not in raw_id


def test_history_features_do_not_depend_on_target_labels():
    rows = sample_rows()
    flipped = rows.copy()
    flipped["objetivo"] = 1 - flipped["objetivo"]

    pd.testing.assert_frame_equal(
        features_for_variant(rows, "history"),
        features_for_variant(flipped, "history"),
    )
