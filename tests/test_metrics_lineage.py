import json

import numpy as np

from datafest.lineage import (
    build_run_manifest,
    sha256_file,
    verify_manifest_integrity,
    write_json,
)
from datafest.metrics import compute_metrics, gini_score


def test_gini_is_twice_auc_minus_one():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])

    metrics = compute_metrics(y, score)

    assert metrics["gini"] == gini_score(y, score)
    assert np.isclose(metrics["gini"], 2 * metrics["roc_auc"] - 1)
    assert 0 <= metrics["pr_auc"] <= 1


def test_sha256_and_manifest_capture_inputs_model_and_outputs(tmp_path):
    source = tmp_path / "source.csv"
    split = tmp_path / "split.csv"
    code = tmp_path / "pipeline.py"
    model = tmp_path / "model.bin"
    validation_model = tmp_path / "validation_model.bin"
    output = tmp_path / "submission.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    split.write_text("role,row\ntrain,0\n", encoding="utf-8")
    code.write_text("# pipeline\n", encoding="utf-8")
    model.write_bytes(b"model")
    validation_model.write_bytes(b"validation-model")
    output.write_text("id_cliente,prediccion\n1,0.8\n", encoding="utf-8")

    manifest = build_run_manifest(
        run_id="test-run",
        model_name="lightgbm",
        feature_variant="clean",
        input_paths=[source],
        model_path=model,
        output_paths=[output],
        parameters={"seed": 42},
        metrics={"gini": 0.5},
        split_artifacts=[split],
        code_paths=[code],
        validation_model_path=validation_model,
    )

    assert manifest["run_id"] == "test-run"
    assert manifest["inputs"][0]["sha256"] == sha256_file(source)
    assert manifest["model"]["sha256"] == sha256_file(model)
    assert manifest["validation_model"]["sha256"] == sha256_file(validation_model)
    assert manifest["outputs"][0]["sha256"] == sha256_file(output)
    assert manifest["split_artifacts"][0]["sha256"] == sha256_file(split)
    assert manifest["code"][0]["sha256"] == sha256_file(code)
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    assert json.loads(manifest_path.read_text()) == manifest
    assert verify_manifest_integrity(manifest, tmp_path) == []
    output.write_text("id_cliente,prediccion\n1,0.1\n", encoding="utf-8")
    assert any("checksum mismatch" in error for error in verify_manifest_integrity(manifest, tmp_path))
    split.write_text("role,row\nvalidation,0\n", encoding="utf-8")
    assert any("checksum mismatch" in error for error in verify_manifest_integrity(manifest, tmp_path))
