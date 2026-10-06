import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from datafest.final_ensemble_n1 import (
    GateFailure,
    align_by_id,
    blend_h4b2_n1,
    run_final_ensemble_n1,
)
from datafest.lineage import sha256_file, verify_manifest_integrity

SEEDS = (42, 43, 44)
N_TRAIN_IDS, N_TEST = 200, 300


def test_blend_is_invariant_to_monotone_transforms_of_each_member():
    rng = np.random.default_rng(3)
    a, b, c = rng.random(500), rng.random(500), rng.random(500)
    months = pd.Series([202612] * 500)
    base = blend_h4b2_n1(a, b, c, months)
    transformed = blend_h4b2_n1(np.exp(5 * a), -1 / (b + 1) * 3, np.log(c + 1e-3) * 7, months)
    assert np.allclose(base, transformed)
    assert np.isclose(base.mean(), (1 + 1 / 500) / 2 * 1.0, atol=1e-3)  # mean of percentile ranks
    # equal weight: swapping members does not change the blend
    assert np.allclose(base, blend_h4b2_n1(c, a, b, months))


def test_align_by_id_reorders_shuffled_input():
    reference = np.array([5, 3, 9, 1])
    ids = np.array([9, 1, 5, 3])
    values = np.array([0.9, 0.1, 0.5, 0.3])
    assert align_by_id(values, ids, reference).tolist() == [0.5, 0.3, 0.9, 0.1]


def test_align_by_id_raises_on_missing_extra_or_duplicate_ids():
    reference = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="missing"):
        align_by_id(np.array([0.1, 0.2]), np.array([1, 2]), reference)
    with pytest.raises(ValueError, match="duplicate"):
        align_by_id(np.array([0.1, 0.2, 0.3]), np.array([1, 2, 2]), reference)
    with pytest.raises(ValueError, match="unexpected"):
        align_by_id(np.array([0.1, 0.2, 0.3, 0.4]), np.array([1, 2, 3, 4]), reference)


def make_inputs(tmp_path, n1_equals_lgb=False, drop_seed=None, perturb_mean=False):
    """Synthetic data dir, H4b2 member file, neural test file and per-seed files."""
    rng = np.random.default_rng(11)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_ids = np.arange(1, N_TRAIN_IDS + 1)
    pd.DataFrame({"id_cliente": np.repeat(train_ids, 2), "mes": np.tile([202610, 202611], len(train_ids)),
                  "objetivo": 0}).to_csv(data_dir / "train.csv", index=False)
    # survivors (ids 1..150) and new clients (ids 201..350 -> 150 clients), sample order is sorted by id
    ids = np.concatenate([np.arange(1, 151), np.arange(201, 201 + N_TEST - 150)])
    pd.DataFrame({"id_cliente": ids, "mes": 202612}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id_cliente": ids, "prediccion": 0.15}).to_csv(data_dir / "sample_submission.csv", index=False)

    base = rng.random(N_TEST)
    lgb = 0.02 + 0.3 * base
    cat = 0.02 + 0.3 * np.clip(base + 0.1 * rng.standard_normal(N_TEST), 0, 1)
    truth = base + 0.15 * rng.standard_normal(N_TEST)
    seeds = {}
    for s in SEEDS:
        raw = truth + 0.05 * rng.standard_normal(N_TEST)
        seeds[s] = (1 / (1 + np.exp(-(raw * 4 - 2)))).astype(np.float32)
    if n1_equals_lgb:
        seeds = {s: lgb.astype(np.float32) for s in SEEDS}
    n1 = np.mean([seeds[s] for s in SEEDS], axis=0)
    if perturb_mean:
        n1 = n1 + np.float32(0.01)

    shuffle = rng.permutation(N_TEST)  # member file is stored in a different row order than the sample
    members = pd.DataFrame({"id_cliente": ids, "prob_lightgbm": lgb, "prob_catboost": cat}).iloc[shuffle]
    members.to_csv(tmp_path / "member_predictions.csv", index=False)
    neural = pd.DataFrame({"id_cliente": ids, "N1": n1, "N2": 0.1}).iloc[rng.permutation(N_TEST)]
    neural.to_csv(tmp_path / "test_neural.csv", index=False)
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    for s in SEEDS:
        if s != drop_seed:
            np.save(seed_dir / f"N1_D_s{s}.npy", seeds[s])
    # the previous submission, only used for the Spearman comparison
    prev = lgb * 0.5 + cat * 0.5
    pd.DataFrame({"id_cliente": ids, "prediccion": prev}).to_csv(tmp_path / "prev_submission.csv", index=False)
    provenance = tmp_path / "ensemble_results.md"
    provenance.write_text("# decision\n", encoding="utf-8")
    return dict(
        data_dir=data_dir, experiment_root=tmp_path / "final", h4b2_members=tmp_path / "member_predictions.csv",
        neural_test=tmp_path / "test_neural.csv", neural_seeds_dir=seed_dir,
        compare_with=tmp_path / "prev_submission.csv", provenance=[provenance],
    ), ids, lgb, cat, n1


def test_run_writes_valid_submission_with_gates_and_lineage(tmp_path):
    kwargs, ids, lgb, cat, n1 = make_inputs(tmp_path)
    result = run_final_ensemble_n1(**kwargs)
    run_dir = tmp_path / "final" / result["run_id"]
    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["id_cliente", "prediccion"]
    assert submission["id_cliente"].tolist() == ids.tolist()  # sample order, not member-file order
    assert submission["prediccion"].between(0, 1).all() and np.isfinite(submission["prediccion"]).all()
    # values are LightGBM probabilities only (ties in the blend share one value), ordered like the blend
    assert set(submission["prediccion"].round(12)) <= set(np.round(lgb, 12))  # CSV parsing may differ by 1 ulp
    assert submission["prediccion"].min() == pytest.approx(lgb.min())
    months = pd.Series([202612] * len(ids))
    blend = blend_h4b2_n1(lgb, cat, n1, months)
    assert spearmanr(submission["prediccion"], blend).statistic > 0.999
    # N1 contributes: not equal to the two-member blend order
    assert spearmanr(submission["prediccion"], lgb).statistic < 0.9999

    assert all(g["passed"] for g in result["gates"].values())
    assert result["gates"]["G2"]["seeds_found"] == [42, 43, 44]
    assert result["gates"]["G2"]["seed_mean_max_abs_diff"] < 1e-7
    assert result["submission_sha256"] == sha256_file(run_dir / "submission.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert verify_manifest_integrity(manifest) == []
    input_names = {p["path"].replace("\\", "/").rsplit("/", 1)[-1] for p in manifest["inputs"]}
    assert {"test_neural.csv", "member_predictions.csv", "N1_D_s42.npy", "ensemble_results.md"} <= input_names
    results = (run_dir / "results.md").read_text(encoding="utf-8")
    assert "G1" in results and "G2" in results and "G3" in results and "G4" in results
    assert result["submission_sha256"] in results
    assert (run_dir / "neural_seeds" / "N1_D_s44.npy").is_file()


def test_gate_failure_prevents_writing_submission(tmp_path):
    kwargs, *_ = make_inputs(tmp_path, n1_equals_lgb=True)  # Spearman N1 vs B0 = 1.0, outside [0.80, 0.95]
    with pytest.raises(GateFailure, match="G1"):
        run_final_ensemble_n1(**kwargs)
    final_root = tmp_path / "final"
    assert not final_root.exists() or not list(final_root.rglob("submission.csv"))


def test_missing_seed_fails_g2_explicitly(tmp_path):
    kwargs, *_ = make_inputs(tmp_path, drop_seed=44)
    with pytest.raises(GateFailure, match="G2") as error:
        run_final_ensemble_n1(**kwargs)
    assert "44" in str(error.value)
    assert not list((tmp_path / "final").rglob("submission.csv")) if (tmp_path / "final").exists() else True


def test_absent_seed_directory_is_reported_not_silently_passed(tmp_path):
    kwargs, *_ = make_inputs(tmp_path)
    kwargs["neural_seeds_dir"] = None
    with pytest.raises(GateFailure, match="G2"):
        run_final_ensemble_n1(**kwargs)


def test_seed_mean_not_matching_n1_column_fails_g2(tmp_path):
    kwargs, *_ = make_inputs(tmp_path, perturb_mean=True)
    with pytest.raises(GateFailure, match="G2"):
        run_final_ensemble_n1(**kwargs)


def test_row_order_of_member_and_neural_files_does_not_change_the_submission(tmp_path):
    kwargs, ids, *_ = make_inputs(tmp_path)
    first = run_final_ensemble_n1(**kwargs)
    members = pd.read_csv(kwargs["h4b2_members"]).sort_values("id_cliente")
    members.to_csv(kwargs["h4b2_members"], index=False)
    neural = pd.read_csv(kwargs["neural_test"]).sort_values("id_cliente", ascending=False)
    neural.to_csv(kwargs["neural_test"], index=False)
    second = run_final_ensemble_n1(**kwargs)
    assert first["submission_sha256"] == second["submission_sha256"]
