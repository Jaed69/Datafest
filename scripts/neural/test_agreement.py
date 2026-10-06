"""Spearman of the neural December predictions vs the repo's existing final submissions (read-only)."""
import sys
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import nn_common as C
nn = pd.read_csv(Path(__file__).resolve().parents[1] / "test_neural.csv")
for run in sorted((C.REPO / "experiments" / "final").iterdir()):
    sub = pd.read_csv(run / "submission.csv")
    assert sub["id_cliente"].equals(nn["id_cliente"])
    print(run.name, {c: round(spearmanr(nn[c], sub["prediccion"]).statistic, 4) for c in ("N1", "N2")})
print("N1 vs N2", round(spearmanr(nn["N1"], nn["N2"]).statistic, 4))
