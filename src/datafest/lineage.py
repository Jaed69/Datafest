from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {"path": str(file_path), "bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)}


def repository_revision() -> dict[str, str | bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": True}


def build_run_manifest(
    run_id: str,
    model_name: str,
    feature_variant: str,
    input_paths: list[str | Path],
    model_path: str | Path,
    output_paths: list[str | Path],
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    *,
    split_path: str | Path | None = None,
    split_artifacts: list[str | Path] | None = None,
    feature_paths: list[str | Path] | None = None,
    code_paths: list[str | Path] | None = None,
    validation_model_path: str | Path | None = None,
    environment: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "feature_variant": feature_variant,
        "repository": repository_revision(),
        "code": [file_record(path) for path in (code_paths or [])],
        "inputs": [file_record(path) for path in input_paths],
        "split": file_record(split_path) if split_path is not None else None,
        "split_artifacts": [file_record(path) for path in (split_artifacts or [])],
        "features": [file_record(path) for path in (feature_paths or [])],
        "parameters": parameters,
        "metrics": metrics,
        "model": file_record(model_path),
        "validation_model": file_record(validation_model_path) if validation_model_path else None,
        "outputs": [file_record(path) for path in output_paths],
        "environment": environment or {},
        "extra": extra or {},
    }


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = stream.name
    os.replace(temporary, target)


def verify_file_records(records: list[dict[str, Any]], base_dir: str | Path = ".") -> list[str]:
    root = Path(base_dir)
    errors = []
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"missing: {record['path']}")
        elif path.stat().st_size != record["bytes"]:
            errors.append(f"size mismatch: {record['path']}")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"checksum mismatch: {record['path']}")
    return errors


def verify_manifest_integrity(
    manifest: dict[str, Any], base_dir: str | Path = "."
) -> list[str]:
    records = list(manifest.get("inputs", []))
    records.extend(manifest.get("code", []))
    records.extend(manifest.get("features", []))
    records.extend(manifest.get("outputs", []))
    records.extend(manifest.get("split_artifacts", []))
    if manifest.get("split"):
        records.append(manifest["split"])
    if manifest.get("model"):
        records.append(manifest["model"])
    if manifest.get("validation_model"):
        records.append(manifest["validation_model"])
    return verify_file_records(records, base_dir)
