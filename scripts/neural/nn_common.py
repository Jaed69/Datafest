"""Shared helpers for the neural tabular candidates N1 (RealMLP-TD) and N2 (TabM-D).

Reuses the repository feature builders read-only (``datafest.ablation._feature_matrix``)
so the neural models see exactly the matrices the GBDT members see.

Protocol (frozen a priori, identical to the repo):
  * rolling folds: train <= Aug -> Sep, <= Sep -> Oct, <= Oct -> Nov
  * no tuning on the scored month; pytabkit's internal early-stopping / best-epoch
    selection uses a random 20% hold-out carved from the TRAINING rows only
    (pytabkit default val_fraction=0.2), with the validation metric set to 1-AUC.
  * fixed seeds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("DATAFEST_REPO", r"C:\Users\Luciel\Documents\GitHub\Datafest"))
sys.path.insert(0, str(REPO / "src"))  # read-only import of the repo package

from datafest.ablation import ROLLING_CUTS, _feature_matrix  # noqa: E402
from datafest.final_fit import build_final_matrices  # noqa: E402
from datafest.features import CATEGORICAL_COLUMNS  # noqa: E402
from datafest.hypotheses import within_month_rank  # noqa: E402

INTERACTION = "dias_ultima_interaccion"
SEEDS = (42, 43, 44)
MODELS = ("N1", "N2")
VARIANTS = {"D": "D_absolute", "Drank": "D_absolute"}  # Drank = D + within-month rank of interaccion


def load_raw():
    train = pd.read_csv(REPO / "data" / "train.csv")
    test = pd.read_csv(REPO / "data" / "test.csv")
    train["objetivo"] = train["objetivo"].astype(int)
    return train, test


def rank_interaction(matrix: pd.DataFrame, months: pd.Series) -> pd.DataFrame:
    out = matrix.copy()
    out[INTERACTION] = within_month_rank(out[INTERACTION], months)
    return out


def rolling_matrix(train: pd.DataFrame, variant: str) -> pd.DataFrame:
    key = VARIANTS[variant]
    matrix = _feature_matrix(train.assign(_source_row=np.arange(len(train))), key)
    if variant == "Drank":
        matrix = rank_interaction(matrix, train["mes"])
    return matrix


def final_matrices(train: pd.DataFrame, test: pd.DataFrame, variant: str):
    key = VARIANTS[variant]
    train_x, test_x = build_final_matrices(train, test, key)
    if variant == "Drank":
        train_x = rank_interaction(train_x, train["mes"])
        test_x = rank_interaction(test_x, test["mes"])
    return train_x, test_x


def to_nn_frame(matrix: pd.DataFrame, levels: dict[str, list]) -> pd.DataFrame:
    """Numeric columns -> float32, booleans -> 0/1, categoricals -> pandas category with fixed levels."""
    out = pd.DataFrame(index=matrix.index)
    for column in matrix.columns:
        if column in CATEGORICAL_COLUMNS:
            out[column] = pd.Categorical(matrix[column].astype(str), categories=levels[column])
        elif pd.api.types.is_bool_dtype(matrix[column]):
            out[column] = matrix[column].astype("int8")
        else:
            out[column] = matrix[column].astype("float32")
    return out


def category_levels(*matrices: pd.DataFrame) -> dict[str, list]:
    """Category vocabularies from the (unlabelled) predictor columns of all rows; carries no label information."""
    levels = {}
    for column in CATEGORICAL_COLUMNS:
        values = pd.concat([m[column].astype(str) for m in matrices if column in m.columns])
        levels[column] = sorted(values.unique())
    return levels


def make_model(name: str, seed: int, device: str = "cuda", verbosity: int = 0):
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier

    # NN_THREADS only limits CPU threads per process (compute resource, not a model setting)
    threads = int(os.environ["NN_THREADS"]) if os.environ.get("NN_THREADS") else None
    kw = dict(device=device, random_state=seed, val_metric_name="1-auc_ovr", verbosity=verbosity, n_threads=threads)
    if name == "N1":
        return RealMLP_TD_Classifier(**kw)
    if name == "N2":
        return TabM_D_Classifier(**kw)
    raise ValueError(name)


def fit_predict(name: str, x_train: pd.DataFrame, y_train: np.ndarray, x_score: pd.DataFrame,
                seed: int, device: str = "cuda") -> np.ndarray:
    cat_cols = [c for c in x_train.columns if c in CATEGORICAL_COLUMNS]
    model = make_model(name, seed, device)
    model.fit(x_train, y_train, cat_col_names=cat_cols)
    return model.predict_proba(x_score)[:, 1]
