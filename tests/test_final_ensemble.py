import json

import numpy as np
import pandas as pd

from datafest.final_ensemble import (
    quantile_map_to_reference,
    rank_interaction_matrices,
    run_final_ensemble,
)
from datafest.lineage import verify_manifest_integrity
from test_final_fit import make_competition


def test_quantile_map_preserves_order_and_uses_only_reference_values():
    rng = np.random.default_rng(1)
    blend = rng.random(200)
    reference = np.sort(rng.random(200)) * 0.5 + 0.1
    rng.shuffle(reference)
    mapped = quantile_map_to_reference(blend, reference)
    assert np.array_equal(np.argsort(mapped, kind="stable"), np.argsort(blend, kind="stable"))
    assert set(mapped) == set(reference)
    assert mapped.min() == reference.min() and mapped.max() == reference.max()
    assert ((mapped >= 0) & (mapped <= 1)).all()


def test_quantile_map_ties_get_identical_deterministic_values():
    blend = np.array([0.2, 0.9, 0.2, 0.5])
    reference = np.array([0.4, 0.1, 0.3, 0.2])
    mapped = quantile_map_to_reference(blend, reference)
    assert mapped[0] == mapped[2] == 0.1
    assert mapped[3] == 0.3 and mapped[1] == 0.4
    assert np.array_equal(mapped, quantile_map_to_reference(blend, reference))


def test_december_interaction_rank_uses_test_rows_only():
    train, test = make_competition()
    train_x = train[["dias_ultima_interaccion"]].copy()
    test_x = pd.DataFrame({"dias_ultima_interaccion": [50, 10, 30, 20, 40]})
    out_train, out_test = rank_interaction_matrices(train_x, train["mes"], test_x, test["mes"])
    assert out_test["dias_ultima_interaccion"].tolist() == [1.0, 0.2, 0.6, 0.4, 0.8]
    # Train ranks are computed inside each training month.
    first = train["mes"].eq(202601).to_numpy()
    expected = train_x.loc[first, "dias_ultima_interaccion"].rank(pct=True).to_numpy()
    assert np.allclose(out_train.loc[first, "dias_ultima_interaccion"], expected)
    # Different training data must not change the December ranks.
    _, again = rank_interaction_matrices(train_x * 3, train["mes"], test_x, test["mes"])
    assert again["dias_ultima_interaccion"].tolist() == out_test["dias_ultima_interaccion"].tolist()


def test_run_final_ensemble_writes_valid_order_preserving_submission(tmp_path):
    train, test = make_competition()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id_cliente": test["id_cliente"], "prediccion": 0.15}).to_csv(
        data_dir / "sample_submission.csv", index=False)

    result = run_final_ensemble(
        data_dir=data_dir, experiment_root=tmp_path / "final",
        iterations={"lightgbm": 5, "catboost": 5},
    )

    run_dir = tmp_path / "final" / result["run_id"]
    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["id_cliente", "prediccion"]
    assert submission["id_cliente"].tolist() == test["id_cliente"].tolist()
    assert submission["prediccion"].between(0, 1).all()
    members = pd.read_csv(run_dir / "member_predictions.csv")
    blend = members[["rank_lightgbm", "rank_catboost"]].mean(axis=1).to_numpy()
    ordered = submission["prediccion"].to_numpy()[np.argsort(blend, kind="stable")]
    assert (np.diff(ordered) >= 0).all()  # weakly monotone (reference probabilities may tie)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["iterations"] == {"lightgbm": 5, "catboost": 5}
    assert verify_manifest_integrity(manifest) == []
    assert (run_dir / "results.md").is_file()
