"""Assemble neural OOF predictions, build blends, run paired bootstrap, write results.

Neural candidates are averaged over seeds (mean probability) per fold. GBDT members are read from the
repo's hypotheses OOF table and joined by (id_cliente, mes), never by position.
Blends are equal-weight per-month rank averages (the repo's ``rank_average``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import nn_common as C
from datafest.hypotheses import fold_ginis, paired_bootstrap, rank_average

OOF_REPO = C.REPO / "experiments" / "hypotheses" / "20261006T100749205414Z_a714ebcd" / "oof_predictions.csv"
B0, H4B2, B1, H1B = "B0_lgbm_D", "H4b2_rank_lgbm_catboost", "B1_catboost_A", "H1b_catboost"
REPLICATES = 2000


def neural_oof(out: Path, train: pd.DataFrame, repo: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged neural predictions on the repo OOF rows, aligned by (id_cliente, mes)."""
    month = train["mes"].to_numpy()
    pieces = {}
    for model in C.MODELS:
        for variant in C.VARIANTS:
            name = model if variant == "D" else f"{model}r"
            folds = []
            for _, valid_month in C.ROLLING_CUTS:
                files = sorted((out / "ckpt").glob(f"{model}_{variant}_{valid_month}_s*.npy"))
                if not files:
                    folds = None
                    break
                p = np.mean([np.load(f) for f in files], axis=0)
                part = train.loc[month == valid_month, ["id_cliente", "mes", "objetivo"]].reset_index(drop=True)
                part[name] = p
                part.attrs["n_seeds"] = len(files)
                folds.append(part)
            if folds:
                pieces[name] = pd.concat(folds, ignore_index=True)
    merged = repo[["id_cliente", "mes", "objetivo"]].copy()
    for name, frame in pieces.items():
        joined = merged.merge(frame[["id_cliente", "mes", "objetivo", name]], on=["id_cliente", "mes"], how="left",
                              suffixes=("", "_nn"), validate="one_to_one")
        if joined[name].isna().any() or not (joined["objetivo"] == joined["objetivo_nn"]).all():
            raise ValueError(f"row alignment failure for {name}")
        merged[name] = joined[name].to_numpy()
    return merged


def frame_of(base: pd.DataFrame, pred) -> pd.DataFrame:
    f = base[["id_cliente", "mes", "objetivo"]].copy()
    f["prediccion"] = np.asarray(pred, dtype=float)
    return f


def fmt_ci(row, label):
    return f"{row[f'delta_{label}']:+.4f} [{row[f'ci_low_{label}']:+.4f}, {row[f'ci_high_{label}']:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    train, _ = C.load_raw()
    repo = pd.read_csv(OOF_REPO)
    nn = neural_oof(out, train, repo)
    nn_names = [c for c in nn.columns if c not in ("id_cliente", "mes", "objetivo")]
    nn.round(7).to_csv(out / "oof_neural.csv", index=False)
    months = repo["mes"]

    members = {B0: repo[B0].to_numpy(), H1B: repo[H1B].to_numpy()}
    # reproduction check of the champion from its two members
    repro = rank_average([members[B0], members[H1B]], months)
    g_repro, g_col = np.mean(fold_ginis(frame_of(repo, repro))), np.mean(fold_ginis(frame_of(repo, repo[H4B2])))
    assert abs(g_repro - g_col) < 1e-6, (g_repro, g_col)

    preds: dict[str, np.ndarray] = {B0: repo[B0].to_numpy(), B1: repo[B1].to_numpy(),
                                    H1B: repo[H1B].to_numpy(), H4B2: repo[H4B2].to_numpy()}
    desc = {B0: "LightGBM D (baseline)", B1: "CatBoost A", H1B: "CatBoost A + rank(interaccion)",
            H4B2: "champion: rank-avg B0 + H1b_catboost"}
    for n in nn_names:
        preds[n] = nn[n].to_numpy()
        base = {"N1": "RealMLP_TD", "N2": "TabM_D"}[n[:2]]
        desc[n] = f"{base} on " + ("D + within-month rank(interaccion) [exploratory]" if n.endswith("r") else "D (absolute)")
    blend_defs = {}
    for suffix in ("", "r"):
        n1, n2 = f"N1{suffix}", f"N2{suffix}"
        if n1 in preds:
            blend_defs[f"BL_H4b2+{n1}"] = [B0, H1B, n1]
        if n2 in preds:
            blend_defs[f"BL_H4b2+{n2}"] = [B0, H1B, n2]
        if n1 in preds and n2 in preds:
            blend_defs[f"BL_H4b2+{n1}+{n2}"] = [B0, H1B, n1, n2]
    for name, mem in blend_defs.items():
        preds[name] = rank_average([preds[m] for m in mem], months)
        desc[name] = "rank-avg of " + " + ".join(mem)

    frames = {n: frame_of(repo, p) for n, p in preds.items()}
    boot_b0 = paired_bootstrap(frames, B0, REPLICATES).set_index("candidate")
    boot_champ = paired_bootstrap(frames, H4B2, REPLICATES).set_index("candidate")

    rows = []
    b0_months = {m: repo.loc[repo["mes"] == m, B0].to_numpy() for m in sorted(repo["mes"].unique())}
    for n, f in frames.items():
        g = fold_ginis(f)
        rho = np.mean([spearmanr(preds[n][(repo["mes"] == m).to_numpy()], b0_months[m]).statistic for m in b0_months])
        row = {"candidate": n, "description": desc[n], "gini_sep": g[0], "gini_oct": g[1], "gini_nov": g[2],
               "mean_gini": float(np.mean(g)), "std_gini": float(np.std(g)), "spearman_vs_B0": float(rho)}
        for tag, boot in (("B0", boot_b0), ("H4b2", boot_champ)):
            if n in boot.index:
                for col in boot.columns:
                    row[f"{col}_vs_{tag}"] = boot.loc[n, col]
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_gini", ascending=False).reset_index(drop=True)
    summary.to_csv(out / "summary.csv", index=False)

    seeds_used = {}
    for model in C.MODELS:
        for variant in C.VARIANTS:
            files = list((out / "ckpt").glob(f"{model}_{variant}_*_s*.npy"))
            if files:
                seeds_used[f"{model}/{variant}"] = sorted({int(f.stem.rsplit('_s', 1)[1]) for f in files})
    timings = pd.read_csv(out / "timings.csv") if (out / "timings.csv").exists() else pd.DataFrame()
    lines = [
        "# Neural tabular candidates (T9)", "",
        "Rolling folds: train <=Aug->Sep, <=Sep->Oct, <=Oct->Nov (same as repo). Features: repo `D_absolute` matrix "
        "(history + month), categoricals as pandas categories, booleans as 0/1; `Drank` variants replace "
        "`dias_ultima_interaccion` by its within-month percentile rank (exploratory). "
        "pytabkit defaults (RealMLP_TD_Classifier, TabM_D_Classifier), `val_metric_name='1-auc_ovr'`, device cuda; "
        "internal best-epoch/early-stopping hold-out = pytabkit default random 20% of the TRAINING rows only; "
        f"seeds per candidate: {seeds_used}; seed-averaged by mean probability. GBDT members come from "
        f"`{OOF_REPO.name}` (run a714ebcd), joined by (id_cliente, mes). Blends = equal-weight per-month rank average "
        f"of B0 + H1b_catboost (the H4b2 members) plus the neural member(s). Paired client bootstrap, {REPLICATES} "
        "resamples, one shared client draw across months and models (repo `paired_bootstrap`, seed 42).", "",
        f"Champion reproduction from its members: mean Gini {g_repro:.5f} vs stored {g_col:.5f}.", "",
        "| Candidate | Sep | Oct | Nov | Mean | Std | Spearman vs B0 | dMean vs B0 [95% CI] | dMean vs H4b2 [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, r in summary.iterrows():
        a = "-" if pd.isna(r.get("delta_mean_vs_B0")) else fmt_ci({k.replace("_vs_B0", ""): v for k, v in r.items() if k.endswith("_vs_B0")}, "mean")
        b = "-" if pd.isna(r.get("delta_mean_vs_H4b2")) else fmt_ci({k.replace("_vs_H4b2", ""): v for k, v in r.items() if k.endswith("_vs_H4b2")}, "mean")
        lines.append(f"| {r['candidate']} | {r['gini_sep']:.5f} | {r['gini_oct']:.5f} | {r['gini_nov']:.5f} | "
                     f"{r['mean_gini']:.5f} | {r['std_gini']:.5f} | {r['spearman_vs_B0']:.3f} | {a} | {b} |")
    lines += ["", "## Per-month deltas vs H4b2 (95% CI)", "", "| Candidate | dSep | dOct | dNov | p Holm (vs H4b2) |",
              "| --- | --- | --- | --- | ---: |"]
    for _, r in summary.iterrows():
        if pd.isna(r.get("delta_mean_vs_H4b2")):
            continue
        sub = {k.replace("_vs_H4b2", ""): v for k, v in r.items() if k.endswith("_vs_H4b2")}
        lines.append(f"| {r['candidate']} | {fmt_ci(sub, 'sep')} | {fmt_ci(sub, 'oct')} | {fmt_ci(sub, 'nov')} | {r['p_holm_vs_H4b2']:.3f} |")
    lines += ["", "## Per-month deltas vs B0 (95% CI)", "", "| Candidate | dSep | dOct | dNov | p Holm (vs B0) |",
              "| --- | --- | --- | --- | ---: |"]
    for _, r in summary.iterrows():
        if pd.isna(r.get("delta_mean_vs_B0")):
            continue
        sub = {k.replace("_vs_B0", ""): v for k, v in r.items() if k.endswith("_vs_B0")}
        lines.append(f"| {r['candidate']} | {fmt_ci(sub, 'sep')} | {fmt_ci(sub, 'oct')} | {fmt_ci(sub, 'nov')} | {r['p_holm_vs_B0']:.3f} |")
    if len(timings):
        t = timings.groupby(["model", "variant"])["fit_seconds"].agg(["count", "mean", "min", "max"]).round(1)
        lines += ["", "## Fit timings (seconds per fit, RTX 5060 Laptop)", "", t.to_markdown()]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
