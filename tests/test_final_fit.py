import json

import numpy as np
import pandas as pd
import pytest

from datafest.ablation import _feature_matrix
from datafest.final_fit import (
    FINAL_ITERATIONS,
    build_final_matrices,
    resolve_variant,
    run_final_fit,
    validate_final_submission,
)
from datafest.lineage import verify_manifest_integrity


def make_competition(tmp_path=None, n_clients=60, months=(202601, 202602, 202603)):
    rng = np.random.default_rng(0)
    rows = []
    for client in range(1, n_clients + 1):
        for month in months:
            if client > 5 * (months.index(month) + 4) and month == months[0]:
                continue
            rows.append({
                "id_cliente": client, "mes": month,
                "edad": int(rng.integers(20, 70)),
                "ocupacion": str(rng.choice(["office", "field"])),
                "dias_ultima_interaccion": int(rng.integers(0, 60)),
            })
    train = pd.DataFrame(rows)
    train["objetivo"] = (rng.random(len(train)) < 0.3).astype(int)
    train.loc[:1, "objetivo"] = [0, 1]
    # A converted client must not reappear later.
    converted = train.loc[train["objetivo"].eq(1)].groupby("id_cliente")["mes"].min()
    train = train[~(train["id_cliente"].map(converted).lt(train["mes"]))].reset_index(drop=True)
    test_clients = [1, 2, 3, 200, 201]
    test = pd.DataFrame({
        "id_cliente": test_clients, "mes": 202604,
        "edad": [30, 40, 50, 25, 35],
        "ocupacion": ["office", "field", "office", "field", "office"],
        "dias_ultima_interaccion": [5, 6, 7, 8, 9],
    })
    return train, test


def test_validator_accepts_exact_sample_layout():
    sample = pd.DataFrame({"id_cliente": [3, 1, 2], "prediccion": [0.15] * 3})
    submission = pd.DataFrame({"id_cliente": [3, 1, 2], "prediccion": [0.1, 0.5, 0.9]})
    validate_final_submission(submission, sample)


@pytest.mark.parametrize(
    "submission, match",
    [
        (pd.DataFrame({"id_cliente": [3, 2, 1], "prediccion": [0.1, 0.2, 0.3]}), "orden"),
        (pd.DataFrame({"id_cliente": [3, 1], "prediccion": [0.1, 0.2]}), "fila"),
        (pd.DataFrame({"id_cliente": [3, 1, 2], "prediccion": [0.1, 1.2, 0.3]}), "prediccion"),
        (pd.DataFrame({"id_cliente": [3, 1, 2], "prediccion": [0.1, np.nan, 0.3]}), "prediccion"),
        (pd.DataFrame({"id_cliente": [3, 1, 2], "prediccion": [0.1, np.inf, 0.3]}), "prediccion"),
        (pd.DataFrame({"id_cliente": [3, 1, 2], "score": [0.1, 0.2, 0.3]}), "id_cliente,prediccion"),
    ],
)
def test_validator_rejects_bad_submissions(submission, match):
    sample = pd.DataFrame({"id_cliente": [3, 1, 2], "prediccion": [0.15] * 3})
    with pytest.raises(ValueError, match=match):
        validate_final_submission(submission, sample)


def test_test_history_features_match_causal_train_history():
    train, test = make_competition()
    train_x, test_x = build_final_matrices(train, test, "D_absolute")

    # Train-side features are unaffected by appending the test month.
    pd.testing.assert_frame_equal(train_x, _feature_matrix(train, "D_absolute"))

    last_train_month = 202603
    for client in (1, 2, 3):
        history = train.loc[train["id_cliente"].eq(client), "mes"]
        row = test_x.iloc[test["id_cliente"].tolist().index(client)]
        assert row["n_observaciones_previas"] == len(history)
        assert row["mes_primera_aparicion"] == history.min()
        assert row["cliente_recurrente"]
    assert last_train_month < test["mes"].min()


def test_new_test_clients_have_no_history():
    train, test = make_competition()
    _, test_x = build_final_matrices(train, test, "D_absolute")
    new = test_x.iloc[test["id_cliente"].tolist().index(200)]
    assert new["n_observaciones_previas"] == 0
    assert new["meses_desde_entrada"] == 0
    assert new["mes_primera_aparicion"] == 202604
    assert not new["cliente_recurrente"]


@pytest.mark.parametrize(
    "key, expects_month, expects_history",
    [("A", False, False), ("D_absolute", True, True), ("D_month_index", False, True)],
)
def test_feature_matrix_has_no_identifier_or_target_leakage(key, expects_month, expects_history):
    train, test = make_competition()
    train_x, test_x = build_final_matrices(train, test, key)

    for matrix in (train_x, test_x):
        assert "id_cliente" not in matrix.columns
        assert "objetivo" not in matrix.columns
        assert ("mes" in matrix.columns) is expects_month
        assert ("n_observaciones_previas" in matrix.columns) is expects_history
    assert list(train_x.columns) == list(test_x.columns)
    assert len(train_x) == len(train) and len(test_x) == len(test)


def test_variant_aliases_and_validated_iterations():
    assert resolve_variant("D") == "D_absolute"
    assert resolve_variant("A") == "A"
    with pytest.raises(ValueError):
        resolve_variant("Z")
    assert FINAL_ITERATIONS == {"lightgbm": 58, "catboost": 192}


def test_run_final_fit_writes_validated_submission_and_manifest(tmp_path):
    train, test = make_competition()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id_cliente": test["id_cliente"], "prediccion": 0.15}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    result = run_final_fit(
        "lightgbm", "D", data_dir=data_dir, experiment_root=tmp_path / "final",
        iterations=5,
    )

    run_dir = tmp_path / "final" / result["run_id"]
    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["id_cliente", "prediccion"]
    assert submission["id_cliente"].tolist() == test["id_cliente"].tolist()
    assert submission["prediccion"].between(0, 1).all()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["iterations"] == 5
    assert manifest["parameters"]["early_stopping"] is False
    assert verify_manifest_integrity(manifest) == []
    assert (run_dir / "results.md").is_file()
