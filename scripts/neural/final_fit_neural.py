"""Final fit of N1/N2: train on ALL Jan-Nov rows, score data/test.csv (December).

Usage:
  python final_fit_neural.py --model N1 --variant D --seeds 42,43,44     # one checkpoint per seed
  python final_fit_neural.py --assemble [--variant D]                      # writes test_neural.csv
Checkpoints: <out>/final/<model>_<variant>_s<seed>.npy. test_neural.csv has id_cliente,N1,N2 in test.csv order
(seed-averaged probabilities). pytabkit holds out a random 20% of the training rows for best-epoch selection
(default behaviour); nothing is tuned on any month.
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
    ap.add_argument("--model", choices=C.MODELS)
    ap.add_argument("--variant", default="D", choices=list(C.VARIANTS))
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "out"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "final").mkdir(parents=True, exist_ok=True)
    train, test = C.load_raw()

    if args.assemble:
        result = pd.DataFrame({"id_cliente": test["id_cliente"].to_numpy()})
        for model in C.MODELS:
            files = sorted((out / "final").glob(f"{model}_{args.variant}_s*.npy"))
            if files:
                result[model] = np.mean([np.load(f) for f in files], axis=0)
                print(f"{model}: {len(files)} seeds")
        assert len(result) == len(test) and result["id_cliente"].equals(test["id_cliente"])
        result.to_csv(Path(args.out).parent / "test_neural.csv", index=False)
        print(result.describe())
        return

    train_x, test_x = C.final_matrices(train, test, args.variant)
    levels = C.category_levels(train_x, test_x)
    x_tr, x_te = C.to_nn_frame(train_x, levels), C.to_nn_frame(test_x, levels)
    y = train["objetivo"].to_numpy()
    for seed in [int(s) for s in args.seeds.split(",")]:
        path = out / "final" / f"{args.model}_{args.variant}_s{seed}.npy"
        if path.exists():
            continue
        t0 = time.time()
        pred = C.fit_predict(args.model, x_tr, y, x_te, seed, args.device)
        np.save(path, pred)
        pd.DataFrame([{"model": args.model, "variant": args.variant, "seed": seed,
                       "fit_seconds": round(time.time() - t0, 1), "n_train": len(x_tr)}]).to_csv(
            out / "final_timings.csv", mode="a", header=not (out / "final_timings.csv").exists(), index=False)
        print(f"final {args.model}/{args.variant} seed={seed}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
