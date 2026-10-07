"""Final submission H4b2+N1: 3-way rank average of B0 LightGBM D, CatBoost A (rank feature) and RealMLP N1.

No model is refit. The members are assembled from stored December test predictions:
the B0 LightGBM and rank-feature CatBoost probabilities of the H4b2 final run, and the
seed-averaged RealMLP (N1, seeds 42/43/44) probabilities of the neural final fit.

The blend is an equal-weight mean of December percentile ranks (monotone-invariant per
member). Submitted values are the order-preserving quantile mapping of the blend onto the
sorted B0 LightGBM December probabilities (``quantile_map_to_reference``).

Gates are computed before anything is written; a failing gate raises ``GateFailure`` and no
submission, results or manifest is produced:

- G1: December Spearman N1 vs B0 overall in [0.80, 0.95] (segments reported).
- G2: exactly seeds 42/43/44 present, each seed finite, non-constant, Spearman vs B0 in
  [0.80, 0.95], pairwise seed Spearman >= 0.90, and the float32 mean of the per-seed files
  reproduces the N1 column used (max abs difference <= 1e-7, the float32 resolution of the
  CSV round trip).
- G3: N1 has no NaN, is inside [0, 1] and not collapsed (std > 0.01); segment means are
  reported next to the GBDT member means.
- G4: submission passes the repository validator, is finite and within [0, 1], and its
  Spearman against the current H4b2 submission is >= 0.95.
"""
from __future__ import annotations

from datetime import datetime, timezone
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from datafest import __version__
from datafest.final_ensemble import _describe, quantile_map_to_reference
from datafest.final_fit import validate_final_submission
from datafest.hypotheses import rank_average
from datafest.lineage import build_run_manifest, sha256_file, write_json

MODEL_NAME = "ensemble-h4b2-n1"
EXPECTED_SEEDS = (42, 43, 44)
SPEARMAN_RANGE = (0.80, 0.95)
SEED_PAIRWISE_MIN = 0.90
MIN_STD = 0.01
SEED_MEAN_TOLERANCE = 1e-7
FINAL_SPEARMAN_MIN = 0.95


class GateFailure(RuntimeError):
    """Raised when at least one gate fails; nothing has been written."""


def align_by_id(values, ids, reference_ids) -> np.ndarray:
    """Return ``values`` reordered to follow ``reference_ids`` using the ids, never positions."""
    values, ids, reference_ids = np.asarray(values), np.asarray(ids), np.asarray(reference_ids)
    if len(values) != len(ids):
        raise ValueError("values and ids must have the same length")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate ids in input")
    series = pd.Series(values, index=ids)
    missing = set(reference_ids.tolist()) - set(ids.tolist())
    if missing:
        raise ValueError(f"missing ids in input: {sorted(missing)[:5]} ({len(missing)} total)")
    extra = set(ids.tolist()) - set(reference_ids.tolist())
    if extra:
        raise ValueError(f"unexpected ids in input: {sorted(extra)[:5]} ({len(extra)} total)")
    return series.loc[reference_ids].to_numpy()


def blend_h4b2_n1(lgb_prob, cat_prob, n1_prob, months) -> np.ndarray:
    """Equal-weight mean of the three members' within-month percentile ranks."""
    return rank_average([np.asarray(lgb_prob), np.asarray(cat_prob), np.asarray(n1_prob)], months)


def _rho(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(spearmanr(a, b).statistic)


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bool(bounds[0] <= value <= bounds[1])


def _seed_files(seeds_dir: Path | None) -> dict[int, Path]:
    if seeds_dir is None or not Path(seeds_dir).is_dir():
        return {}
    files = {}
    for path in Path(seeds_dir).glob("N1_D_s*.npy"):
        files[int(path.stem.rsplit("_s", 1)[1])] = path
    return dict(sorted(files.items()))


def gate_g1(n1: np.ndarray, lgb: np.ndarray, survivor: np.ndarray, new: np.ndarray) -> dict:
    overall = _rho(n1, lgb)
    return {"passed": _in_range(overall, SPEARMAN_RANGE), "overall": overall,
            "survivors": _rho(n1, lgb, survivor), "new_clients": _rho(n1, lgb, new),
            "range": list(SPEARMAN_RANGE), "n_survivors": int(survivor.sum()), "n_new": int(new.sum())}


def gate_g2(seed_files: dict[int, Path], expected_seeds: tuple[int, ...], n1: np.ndarray, lgb: np.ndarray,
            test_ids: np.ndarray, sample_ids: np.ndarray) -> dict:
    found = sorted(seed_files)
    gate = {"passed": False, "expected_seeds": list(expected_seeds), "seeds_found": found,
            "seed_vs_b0": {}, "pairwise": {}, "seed_mean_max_abs_diff": None, "problems": []}
    if found != sorted(expected_seeds):
        missing = sorted(set(expected_seeds) - set(found))
        gate["problems"].append(f"seed set not confirmed: expected {list(expected_seeds)}, found {found}"
                                + (f", missing {missing}" if missing else ""))
        return gate
    arrays = {}
    for seed, path in seed_files.items():
        raw = np.load(path)
        if raw.shape != (len(sample_ids),):
            gate["problems"].append(f"seed {seed}: shape {raw.shape}, expected ({len(sample_ids)},)")
            continue
        arrays[seed] = raw
    if gate["problems"]:
        return gate
    # Seed files are stored in test.csv row order; align them to the sample by id.
    aligned = {s: align_by_id(a, test_ids, sample_ids) for s, a in arrays.items()}
    for seed, values in aligned.items():
        if not np.isfinite(values).all():
            gate["problems"].append(f"seed {seed}: contains NaN/inf")
            continue
        if float(np.std(values)) <= MIN_STD:
            gate["problems"].append(f"seed {seed}: collapsed (std {float(np.std(values)):.5f})")
        rho = _rho(values, lgb)
        gate["seed_vs_b0"][seed] = rho
        if not _in_range(rho, SPEARMAN_RANGE):
            gate["problems"].append(f"seed {seed}: Spearman vs B0 {rho:.4f} outside {list(SPEARMAN_RANGE)}")
    seeds = sorted(aligned)
    if not gate["problems"]:
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                rho = _rho(aligned[a], aligned[b])
                gate["pairwise"][f"{a}-{b}"] = rho
                if rho < SEED_PAIRWISE_MIN:
                    gate["problems"].append(f"seeds {a}/{b}: pairwise Spearman {rho:.4f} < {SEED_PAIRWISE_MIN}")
        mean = np.mean([aligned[s] for s in seeds], axis=0)  # float32 mean, as the neural final fit does
        diff = float(np.max(np.abs(mean.astype(float) - n1)))
        gate["seed_mean_max_abs_diff"] = diff
        if not diff <= SEED_MEAN_TOLERANCE:
            gate["problems"].append(f"mean of seeds {seeds} differs from N1 column by {diff:.3e} "
                                    f"(> {SEED_MEAN_TOLERANCE:.0e})")
    gate["passed"] = not gate["problems"]
    return gate


def gate_g3(n1: np.ndarray, members: dict[str, np.ndarray], survivor: np.ndarray, new: np.ndarray) -> dict:
    finite = bool(np.isfinite(n1).all())
    std = float(np.std(n1)) if finite else float("nan")
    in_unit = bool(finite and ((n1 >= 0) & (n1 <= 1)).all())
    means = {name: {"survivors": float(v[survivor].mean()), "new_clients": float(v[new].mean())}
             for name, v in {"N1": n1, **members}.items()} if finite else {}
    return {"passed": bool(finite and in_unit and std > MIN_STD), "finite": finite, "in_unit_interval": in_unit,
            "std": std, "means": means}


def gate_g4(submission: pd.DataFrame, sample: pd.DataFrame, compare_with: str | Path | None) -> dict:
    gate = {"passed": False, "rows": int(len(submission)), "validator": False, "finite": False,
            "in_unit_interval": False, "spearman_vs_h4b2": None, "min_spearman": FINAL_SPEARMAN_MIN,
            "compared_with": str(compare_with) if compare_with else None}
    try:
        validate_final_submission(submission, sample)
        gate["validator"] = True
    except ValueError as error:
        gate["validator_error"] = str(error)
    values = submission["prediccion"].to_numpy()
    gate["finite"] = bool(np.isfinite(values).all())
    gate["in_unit_interval"] = bool(gate["finite"] and ((values >= 0) & (values <= 1)).all())
    ok = gate["validator"] and gate["finite"] and gate["in_unit_interval"]
    if compare_with is not None:
        other = pd.read_csv(compare_with)
        aligned = align_by_id(other["prediccion"].to_numpy(), other["id_cliente"].to_numpy(),
                              submission["id_cliente"].to_numpy())
        gate["spearman_vs_h4b2"] = _rho(values, aligned)
        ok = ok and gate["spearman_vs_h4b2"] >= FINAL_SPEARMAN_MIN
    gate["passed"] = bool(ok)
    return gate


def _results_md(run_id: str, gates: dict, submission_sha256: str, submission: pd.DataFrame,
                provenance: list[str], sources: dict) -> str:
    g1, g2, g3, g4 = gates["G1"], gates["G2"], gates["G3"], gates["G4"]
    mark = lambda g: "PASS" if g["passed"] else "FAIL"  # noqa: E731
    seeds = ", ".join(f"s{s}: {r:.4f}" for s, r in g2["seed_vs_b0"].items())
    pairs = ", ".join(f"{k}: {r:.4f}" for k, r in g2["pairwise"].items())
    lines = [
        f"# Final ensemble {run_id}", "",
        "- Model: H4b2+N1 = equal-weight 3-way mean of December percentile ranks of B0 LightGBM D, "
        "CatBoost A (`dias_ultima_interaccion` -> within-December rank) and RealMLP N1 (pytabkit, "
        "seeds 42/43/44, seed-averaged probability).",
        "- No refit: members are the stored December test predictions listed under Sources.",
        "- Output mapping: order-preserving quantile mapping of the blend onto the sorted B0 LightGBM "
        "December probabilities (ties share the first value of the tie group). Order, hence AUC/Gini, "
        "equals the blend's; the marginal distribution equals LightGBM's.", "",
        "## Decision provenance", "",
        "- Chosen over H4b2 and B0 by two blind judges (rolling mean Gini: BL_H4b2+N1 0.25432 vs H4b2 0.25245 "
        "vs B0 0.25104; deltas not significant, CI includes 0).",
        *[f"- Evidence: `{p}`" for p in provenance], "",
        "## Gate table", "",
        "| Gate | Result | Evidence |", "| --- | --- | --- |",
        f"| G1 Spearman N1 vs B0 (Dec) in [0.80, 0.95] | {mark(g1)} | overall {g1['overall']:.4f}; "
        f"Nov survivors ({g1['n_survivors']}) {g1['survivors']:.4f}; new clients ({g1['n_new']}) "
        f"{g1['new_clients']:.4f}; OOF reference 0.888 |",
        f"| G2 seeds 42/43/44 (exact set) | {mark(g2)} | found {g2['seeds_found']}; vs B0 {seeds}; "
        f"pairwise {pairs}; max abs diff mean(seeds) vs N1 column {g2['seed_mean_max_abs_diff']:.2e} "
        f"(tolerance {SEED_MEAN_TOLERANCE:.0e}, float32 CSV resolution) |",
        f"| G3 N1 sanity | {mark(g3)} | finite {g3['finite']}; in [0,1] {g3['in_unit_interval']}; "
        f"std {g3['std']:.4f} |",
        f"| G4 final submission | {mark(g4)} | rows {g4['rows']}; validator {g4['validator']}; finite "
        f"{g4['finite']}; in [0,1] {g4['in_unit_interval']}; Spearman vs current H4b2 submission "
        f"{g4['spearman_vs_h4b2']:.4f} (min {g4['min_spearman']}) |", "",
        "### G3 segment means (probability scale)", "",
        "| Member | Nov survivors | New clients |", "| --- | ---: | ---: |",
    ]
    for name, m in g3["means"].items():
        lines.append(f"| {name} | {m['survivors']:.4f} | {m['new_clients']:.4f} |")
    desc = _describe(submission["prediccion"].to_numpy())
    lines += [
        "", "## Submission", "",
        f"- Rows: {len(submission)}; columns id_cliente,prediccion; sample_submission order validated.",
        f"- Mean {desc['mean']:.5f}, p10 {desc['p10']:.5f}, p50 {desc['p50']:.5f}, p90 {desc['p90']:.5f}.",
        f"- SHA-256 of submission.csv: `{submission_sha256}`", "",
        "## Sources (SHA-256 recorded in manifest.json)", "",
        *[f"- {label}: `{path}`" for label, path in sources.items()],
    ]
    return "\n".join(lines) + "\n"


def run_final_ensemble_n1(
    data_dir: str | Path = "data",
    experiment_root: str | Path = "experiments/final",
    h4b2_members: str | Path = "experiments/final/20261006T101910790713Z_a714ebcd_ensemble-h4b2/member_predictions.csv",
    neural_test: str | Path = "experiments/ensembles/20261006T173753810737Z_a714ebcd_ensemble-search/neural/test_neural.csv",
    neural_seeds_dir: str | Path | None = None,
    compare_with: str | Path | None = None,
    provenance: list[str | Path] | None = None,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> dict:
    data_dir, experiment_root = Path(data_dir), Path(experiment_root)
    h4b2_members, neural_test = Path(h4b2_members), Path(neural_test)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    train_ids = pd.read_csv(train_path, usecols=["id_cliente"])["id_cliente"]
    test = pd.read_csv(test_path, usecols=["id_cliente", "mes"])
    sample = pd.read_csv(sample_path)
    sample_ids = sample["id_cliente"].to_numpy()
    survivor = pd.Series(sample_ids).isin(train_ids).to_numpy()
    new = (sample_ids > train_ids.max())
    months = pd.Series(align_by_id(test["mes"].to_numpy(), test["id_cliente"].to_numpy(), sample_ids))

    members = pd.read_csv(h4b2_members)
    lgb = align_by_id(members["prob_lightgbm"].to_numpy(), members["id_cliente"].to_numpy(), sample_ids)
    cat = align_by_id(members["prob_catboost"].to_numpy(), members["id_cliente"].to_numpy(), sample_ids)
    neural = pd.read_csv(neural_test)
    n1 = align_by_id(neural["N1"].to_numpy(dtype=float), neural["id_cliente"].to_numpy(), sample_ids)

    seed_files = _seed_files(None if neural_seeds_dir is None else Path(neural_seeds_dir))
    gates = {
        "G1": gate_g1(n1, lgb, survivor, new),
        "G2": gate_g2(seed_files, tuple(expected_seeds), n1, lgb, test["id_cliente"].to_numpy(), sample_ids),
        "G3": gate_g3(n1, {"B0 LightGBM": lgb, "CatBoost A rank": cat}, survivor, new),
    }
    blend = blend_h4b2_n1(lgb, cat, n1, months)
    submission = pd.DataFrame({"id_cliente": sample_ids, "prediccion": quantile_map_to_reference(blend, lgb)})
    gates["G4"] = gate_g4(submission, sample, compare_with)
    failed = {name: gate for name, gate in gates.items() if not gate["passed"]}
    if failed:
        detail = "; ".join(f"{name}: {gate.get('problems') or {k: v for k, v in gate.items() if k != 'means'}}"
                           for name, gate in failed.items())
        raise GateFailure(f"gates failed ({', '.join(failed)}), no submission written. {detail}")

    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
              f"_{sha256_file(train_path)[:8]}_{MODEL_NAME}")
    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds_out = run_dir / "neural_seeds"
    seeds_out.mkdir()
    copied = []
    for path in seed_files.values():
        shutil.copy2(path, seeds_out / path.name)
        copied.append(seeds_out / path.name)
    timings = Path(neural_seeds_dir).parent / "final_timings.csv"
    if timings.is_file():
        shutil.copy2(timings, seeds_out / timings.name)
        copied.append(seeds_out / timings.name)

    submission_path = run_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    submission_sha256 = sha256_file(submission_path)
    members_path = run_dir / "member_predictions.csv"
    pd.DataFrame({
        "id_cliente": sample_ids, "prob_lightgbm": lgb, "prob_catboost": cat, "prob_n1": n1,
        "rank_lightgbm": rank_average([lgb], months), "rank_catboost": rank_average([cat], months),
        "rank_n1": rank_average([n1], months), "blend": blend,
    }).to_csv(members_path, index=False)

    provenance = [str(p) for p in (provenance or [])]
    sources = {"H4b2 member predictions": str(h4b2_members), "Neural N1 test predictions": str(neural_test),
               "Neural per-seed predictions (copied)": str(seeds_out), "Compared with": str(compare_with)}
    results_path = run_dir / "results.md"
    results_path.write_text(_results_md(run_id, gates, submission_sha256, submission, provenance, sources),
                            encoding="utf-8")

    h4b2_models = [p for p in (h4b2_members.parent / "model_lightgbm.txt", h4b2_members.parent / "model_catboost.cbm")
                   if p.is_file()]
    code_paths = [Path("src/datafest") / n for n in ("final_ensemble_n1.py", "final_ensemble.py", "final_fit.py",
                  "hypotheses.py", "lineage.py", "data.py")] + [Path("pyproject.toml"), Path("uv.lock")]
    manifest = build_run_manifest(
        run_id, MODEL_NAME, "D_absolute+A(rank_interaccion)+N1(RealMLP D_absolute)",
        input_paths=[train_path, test_path, sample_path, h4b2_members, neural_test, *copied, *h4b2_models,
                     *[Path(p) for p in provenance if Path(p).is_file()],
                     *([Path(compare_with)] if compare_with else [])],
        model_path=members_path, output_paths=[submission_path, results_path],
        parameters={
            "members": ["B0 LightGBM D (H4b2 final run)", "CatBoost A + within-December rank (H4b2 final run)",
                        f"RealMLP N1 seeds {list(expected_seeds)} mean probability"],
            "blend": "equal-weight mean of December percentile ranks (3 members)",
            "output_mapping": "quantile map onto sorted B0 LightGBM December probabilities",
            "refit": False, "neural_seeds_source": str(neural_seeds_dir),
            "decision": "two blind judges selected (b) H4b2+N1",
        },
        metrics={"gates": gates, "submission_sha256": submission_sha256},
        code_paths=[p for p in code_paths if p.is_file()],
        environment={"python": platform.python_version(), "datafest": __version__},
    )
    write_json(run_dir / "manifest.json", manifest)
    return {"run_id": run_id, "run_dir": str(run_dir), "submission": str(submission_path), "rows": len(submission),
            "submission_sha256": submission_sha256, "gates": gates}
