from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from datafest.features import CATEGORICAL_COLUMNS


MAX_ROUNDS = 1600
EARLY_STOPPING_ROUNDS = 100
SEED = 42
MODEL_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "lightgbm": [
        {"name": "baseline", "num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "min_child_samples": 20, "reg_lambda": 0.0},
        {"name": "compact", "num_leaves": 15, "max_depth": 5, "learning_rate": 0.03, "min_child_samples": 20, "reg_lambda": 1.0},
        {"name": "regularized", "num_leaves": 31, "max_depth": 7, "learning_rate": 0.03, "min_child_samples": 50, "reg_lambda": 1.0},
        {"name": "wide", "num_leaves": 63, "max_depth": 8, "learning_rate": 0.05, "min_child_samples": 50, "reg_lambda": 5.0},
        {"name": "conservative", "num_leaves": 31, "max_depth": -1, "learning_rate": 0.1, "min_child_samples": 100, "reg_lambda": 5.0},
    ],
    "catboost": [
        {"name": "baseline", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"name": "compact", "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
        {"name": "regularized", "depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 5.0},
        {"name": "deep", "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
        {"name": "fast", "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 8.0},
    ],
    "xgboost": [
        {"name": "baseline", "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 10.0, "reg_lambda": 5.0,
         "subsample": 0.8, "colsample_bytree": 0.8},
    ],
}
_XGBOOST_DEVICE: str | None = None


def xgboost_device() -> str:
    """``cuda`` when XGBoost can train on the GPU in this environment, else ``cpu`` (probed once)."""
    global _XGBOOST_DEVICE
    if _XGBOOST_DEVICE is None:
        try:
            probe = XGBClassifier(n_estimators=1, tree_method="hist", device="cuda", verbosity=0)
            probe.fit(np.random.default_rng(0).random((64, 2)), np.arange(64) % 2)
            _XGBOOST_DEVICE = "cuda"
        except Exception:  # noqa: BLE001 - any CUDA/driver failure means CPU
            _XGBOOST_DEVICE = "cpu"
    return _XGBOOST_DEVICE


@dataclass
class FittedModel:
    model_name: str
    estimator: Any
    feature_columns: list[str]
    categorical_columns: list[str]
    category_maps: dict[str, dict[str, int]]

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = frame.loc[:, self.feature_columns].copy()
        for column in matrix.columns:
            if column in self.categorical_columns:
                values = matrix[column].fillna("__UNKNOWN__").astype(str)
                if self.model_name == "catboost":
                    matrix[column] = values
                else:
                    mapping = self.category_maps[column]
                    codes = values.map(mapping).fillna(len(mapping)).astype("int32")
                    matrix[column] = pd.Categorical(
                        codes, categories=list(range(len(mapping) + 1))
                    )
            elif pd.api.types.is_bool_dtype(matrix[column]):
                matrix[column] = matrix[column].astype("int8")
        return matrix

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._transform(frame)
        return self.estimator.predict_proba(matrix)[:, 1]

    @property
    def best_iteration(self) -> int:
        if self.model_name == "xgboost":
            best = getattr(self.estimator, "best_iteration", None)
            return max(1, int(best) + 1) if best is not None else max(1, int(self.estimator.n_estimators))
        value = getattr(self.estimator, "best_iteration_", None)
        if value is None or int(value) <= 0:
            value = getattr(self.estimator, "tree_count_", None)
        if value is None or int(value) <= 0:
            value = getattr(self.estimator, "n_estimators", None)
        return max(1, int(value or 1))

    def save(self, path: str) -> None:
        if self.model_name == "lightgbm":
            self.estimator.booster_.save_model(path)
        else:
            self.estimator.save_model(path)


def _category_maps(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps = {}
    for column in CATEGORICAL_COLUMNS:
        if column in frame.columns:
            values = sorted(frame[column].fillna("__UNKNOWN__").astype(str).unique())
            maps[column] = {value: index for index, value in enumerate(values)}
    return maps


def fit_model(
    model_name: str,
    train_x: pd.DataFrame,
    train_y,
    *,
    valid_x: pd.DataFrame | None = None,
    valid_y=None,
    config: dict[str, Any] | None = None,
    seed: int = SEED,
    iterations: int = MAX_ROUNDS,
    early_stopping_rounds: int | None = EARLY_STOPPING_ROUNDS,
    sample_weight=None,
) -> FittedModel:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Modelo desconocido: {model_name}")
    config = dict(config or MODEL_CONFIGS[model_name][0])
    config.pop("name", None)
    feature_columns = list(train_x.columns)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    category_maps = _category_maps(train_x)
    wrapper = FittedModel(model_name, None, feature_columns, categorical, category_maps)
    x_fit = wrapper._transform(train_x)
    y_fit = np.asarray(train_y, dtype=int)
    if set(np.unique(y_fit)) != {0, 1}:
        raise ValueError("El entrenamiento debe contener ambas clases")

    eval_set = None
    x_valid = None
    if valid_x is not None:
        if valid_y is None:
            raise ValueError("valid_y es obligatorio cuando se pasa valid_x")
        x_valid = wrapper._transform(valid_x)
        eval_set = [(x_valid, np.asarray(valid_y, dtype=int))]

    if model_name == "lightgbm":
        estimator = LGBMClassifier(
            objective="binary",
            n_estimators=iterations,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
            **config,
        )
        fit_args: dict[str, Any] = {}
        if sample_weight is not None:
            fit_args["sample_weight"] = np.asarray(sample_weight, dtype=float)
        if eval_set:
            fit_args["eval_set"] = eval_set
            fit_args["eval_metric"] = "auc"
            if early_stopping_rounds:
                fit_args["callbacks"] = [
                    lgb.early_stopping(early_stopping_rounds, verbose=False)
                ]
        estimator.fit(x_fit, y_fit, **fit_args)
    elif model_name == "xgboost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            n_estimators=iterations,
            random_state=seed,
            n_jobs=4,
            verbosity=0,
            tree_method="hist",
            device=xgboost_device(),
            enable_categorical=True,
            eval_metric="auc",
            early_stopping_rounds=early_stopping_rounds if eval_set and early_stopping_rounds else None,
            **config,
        )
        fit_args = {"verbose": False}
        if sample_weight is not None:
            fit_args["sample_weight"] = np.asarray(sample_weight, dtype=float)
        if eval_set:
            fit_args["eval_set"] = eval_set
        estimator.fit(x_fit, y_fit, **fit_args)
    else:
        thread_count = config.pop("thread_count", 4)
        estimator = CatBoostClassifier(
            iterations=iterations,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=thread_count,
            **config,
        )
        fit_args = {"cat_features": categorical}
        if sample_weight is not None:
            fit_args["sample_weight"] = np.asarray(sample_weight, dtype=float)
        if eval_set:
            fit_args["eval_set"] = eval_set[0]
            if early_stopping_rounds:
                fit_args["early_stopping_rounds"] = early_stopping_rounds
                fit_args["use_best_model"] = True
        estimator.fit(x_fit, y_fit, **fit_args)

    wrapper.estimator = estimator
    return wrapper
