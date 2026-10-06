"""Per-seed rolling Gini of each neural candidate (single-seed spread = model-init noise); prints a markdown table."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import nn_common as C

out = Path(__file__).resolve().parents[1] / "out"
train, _ = C.load_raw()
rows = []
for model in C.MODELS:
    for variant in C.VARIANTS:
        for seed in C.SEEDS:
            g = []
            for _, vm in C.ROLLING_CUTS:
                f = out / "ckpt" / f"{model}_{variant}_{vm}_s{seed}.npy"
                m = train["mes"].eq(vm).to_numpy()
                g.append(2 * roc_auc_score(train.loc[m, "objetivo"], np.load(f)) - 1)
            rows.append({"candidate": model if variant == "D" else model + "r", "seed": seed,
                         "sep": g[0], "oct": g[1], "nov": g[2], "mean": np.mean(g)})
t = pd.DataFrame(rows)
print(t.round(5).to_markdown(index=False))
print()
print(t.groupby("candidate")["mean"].agg(["mean", "std", "min", "max"]).round(5).to_markdown())
