"""Rolling Sep/Oct/Nov evaluation of one neural candidate; resumable per (fold, seed).

Usage: python run_rolling.py --model N1 --variant D --seeds 42,43,44 [--folds 202609,...]
Writes checkpoints to <out>/ckpt/<model>_<variant>_<valid_month>_s<seed>.npy and timings.csv.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import nn_common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.MODELS)
    ap.add_argument("--variant", default="D", choices=list(C.VARIANTS))
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--folds", default="")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "out"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    wanted = {int(f) for f in args.folds.split(",") if f}

    train, test = C.load_raw()
    matrix = C.rolling_matrix(train, args.variant)
    levels = C.category_levels(matrix, C.final_matrices(train, test, args.variant)[1])
    nn_matrix = C.to_nn_frame(matrix, levels)
    month = train["mes"].to_numpy()
    y = train["objetivo"].to_numpy()

    for train_through, valid_month in C.ROLLING_CUTS:
        if wanted and valid_month not in wanted:
            continue
        tr = month <= train_through
        va = month == valid_month
        x_tr = nn_matrix.loc[tr].reset_index(drop=True)
        x_va = nn_matrix.loc[va].reset_index(drop=True)
        for seed in seeds:
            path = out / "ckpt" / f"{args.model}_{args.variant}_{valid_month}_s{seed}.npy"
            if path.exists():
                continue
            t0 = time.time()
            pred = C.fit_predict(args.model, x_tr, y[tr], x_va, seed, args.device)
            dt = time.time() - t0
            np.save(path, pred)
            row = pd.DataFrame([{"model": args.model, "variant": args.variant, "valid_month": valid_month,
                                 "seed": seed, "fit_seconds": round(dt, 1), "n_train": int(tr.sum())}])
            row.to_csv(out / "timings.csv", mode="a", header=not (out / "timings.csv").exists(), index=False)
            from sklearn.metrics import roc_auc_score
            g = 2 * roc_auc_score(y[va], pred) - 1
            print(f"{args.model}/{args.variant} valid={valid_month} seed={seed}: {dt:.0f}s gini={g:.5f}", flush=True)


if __name__ == "__main__":
    main()
