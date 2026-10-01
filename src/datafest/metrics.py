from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def gini_score(y_true, score) -> float:
    return float(2 * roc_auc_score(y_true, score) - 1)


def compute_metrics(y_true, score) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    prediction = np.asarray(score, dtype=float)
    if len(y) != len(prediction) or len(y) == 0:
        raise ValueError("Las etiquetas y probabilidades deben tener el mismo tamaño no vacío")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Se requieren ambas clases para calcular las métricas")
    if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1")
    auc = float(roc_auc_score(y, prediction))
    return {
        "gini": float(2 * auc - 1),
        "roc_auc": auc,
        "pr_auc": float(average_precision_score(y, prediction)),
    }


def paired_group_bootstrap_gini_delta(
    y_true,
    score_a,
    score_b,
    groups,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    cluster = np.asarray(groups)
    if not (len(y) == len(a) == len(b) == len(cluster)):
        raise ValueError("y, predicciones y grupos deben tener la misma longitud")
    observed = gini_score(y, a) - gini_score(y, b)
    by_group = defaultdict(list)
    for index, value in enumerate(cluster):
        by_group[value].append(index)
    group_keys = np.asarray(list(by_group), dtype=object)
    if len(group_keys) < 2:
        raise ValueError("Se requieren al menos dos grupos para el bootstrap")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_resamples):
        selected = rng.choice(group_keys, size=len(group_keys), replace=True)
        indices = np.concatenate([by_group[group] for group in selected])
        if len(np.unique(y[indices])) < 2:
            continue
        deltas.append(gini_score(y[indices], a[indices]) - gini_score(y[indices], b[indices]))
    if not deltas:
        raise ValueError("El bootstrap no produjo muestras con ambas clases")
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta_gini": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_replicates": int(len(deltas)),
    }
