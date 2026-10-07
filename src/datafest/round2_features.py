"""Round-2 feature families F1-F5 as pure functions plus a family registry.

Each family returns extra columns (and optionally columns to drop) that are
appended to a base feature matrix (LightGBM D or CatBoost A). All functions are
causal: they use only the row's own month, earlier months, or per-month
rankings, and never the target of the row's own month or later.

Row order contract: every function returns a frame with a fresh RangeIndex
aligned positionally with the (reset-index) input frame.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from datafest.calendar_features import calendar_features
from datafest.features import CATEGORICAL_COLUMNS
from datafest.hypotheses import within_month_rank

NUMERIC_COLUMNS = [
    "edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos",
    "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses",
    "visitas_web_ultimos_90_dias", "distancia_sucursal_km", "dia_preferido_pago",
    "dias_ultima_interaccion",
]
TE_SMOOTHING = 50.0


def pair_keys(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Single categorical keys followed by every 2-way combination (``a__b`` -> ``x|y``)."""
    keys = {c: frame[c].fillna("__UNKNOWN__").astype(str) for c in columns}
    for a, b in combinations(columns, 2):
        keys[f"{a}__{b}"] = keys[a] + "|" + keys[b]
    return pd.DataFrame(keys).reset_index(drop=True)


def causal_target_encoding(
    months: pd.Series, labels: pd.Series, keys: pd.DataFrame, smoothing: float = TE_SMOOTHING
) -> pd.DataFrame:
    """Expanding smoothed target mean per key using only months strictly before the row's month.

    ``te = (S + a * prior) / (N + a)`` where S, N are positives / labelled rows of the
    key in earlier months and ``prior`` is the pooled mean of earlier months. Rows in the
    first month have no history and get NaN. Rows whose label is NaN (unlabelled, e.g. the
    test month) contribute nothing; they are only valid as the latest month.
    """
    m = np.asarray(months, dtype="int64")
    y = np.asarray(labels, dtype=float)
    known = ~np.isnan(y)
    pos = np.where(known, y, 0.0)
    cnt = known.astype(float)
    totals = pd.DataFrame({"m": m, "y": pos, "n": cnt}).groupby("m")[["y", "n"]].sum().sort_index()
    before = totals.cumsum() - totals
    prior_by_month = before["y"] / before["n"].where(before["n"] > 0)
    prior = pd.Series(m).map(prior_by_month).to_numpy(dtype=float)
    out = {}
    for column in keys.columns:
        rows = pd.DataFrame({"g": keys[column].fillna("__UNKNOWN__").astype(str).to_numpy(),
                             "m": m, "y": pos, "n": cnt})
        agg = rows.groupby(["g", "m"], sort=True)[["y", "n"]].sum()
        earlier = (agg.groupby(level="g").cumsum() - agg).reset_index()
        merged = rows[["g", "m"]].merge(earlier, on=["g", "m"], how="left")
        encoded = (merged["y"].to_numpy() + smoothing * prior) / (merged["n"].to_numpy() + smoothing)
        out[f"te_{column}"] = np.where(np.isnan(prior), np.nan, encoded)
    return pd.DataFrame(out)


def within_month_rank_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({f"rk_{c}": within_month_rank(frame[c], frame["mes"]) for c in columns})


def behavioral_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    """F2 ratios and interactions of the static behavioural columns."""
    adult_months = np.maximum(12.0 * (frame["edad"].to_numpy(float) - 18.0), 1.0)
    tenure_years = (frame["antiguedad_cuenta_meses"].to_numpy(float) + 1.0) / 12.0
    gap = (frame["dias_ultima_interaccion"] - frame["dias_ultima_transaccion"]).abs()
    out = pd.DataFrame({
        "saldo_ingresos": frame["saldo_promedio"] / frame["ingresos"],
        "ingreso_neto": frame["ingresos"] * (1.0 - frame["ratio_deuda_ingresos"]),
        "antiguedad_share_vida_adulta": frame["antiguedad_cuenta_meses"] / adult_months,
        "productos_por_anio_antiguedad": frame["numero_productos"] / tenure_years,
        "visitas_por_producto": frame["visitas_web_ultimos_90_dias"] / np.maximum(frame["numero_productos"], 1),
        "recencia_ratio": frame["dias_ultima_transaccion"] / (frame["antiguedad_cuenta_meses"] + 1.0),
        "gap_interaccion_rank": within_month_rank(gap, frame["mes"]),
    })
    return out.reset_index(drop=True)


def cohort_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """F5: rank of each feature inside the client's first-seen cohort (same month), plus cohort survival.

    The cohort is the client's first month in the data. ``cohort_survival`` is the share of the
    cohort still present this month (membership only, no labels) and ``cohort_size_initial`` the
    cohort size in its first month; both use only months up to the row's month.
    """
    data = frame.reset_index(drop=True)
    cohort = data.groupby("id_cliente")["mes"].transform("min")
    work = data[["mes"] + columns].assign(cohort=cohort)
    out = {}
    for column in columns:
        out[f"cohort_rk_{column}"] = work.groupby(["cohort", "mes"])[column].rank(pct=True, method="average").to_numpy()
    present = work.groupby(["cohort", "mes"])["mes"].transform("size").to_numpy(dtype=float)
    first = (work["mes"] == work["cohort"]).groupby(work["cohort"]).transform("sum").to_numpy(dtype=float)
    out["cohort_survival"] = present / first
    out["cohort_size_initial"] = first.astype("int64")
    return pd.DataFrame(out)


# ---------------------------------------------------------------- family registry
def te_features(frame: pd.DataFrame) -> pd.DataFrame:
    keys = pair_keys(frame, CATEGORICAL_COLUMNS)
    return causal_target_encoding(frame["mes"], frame["objetivo"], keys)


def _calendar(frame: pd.DataFrame, flags: bool) -> pd.DataFrame:
    return calendar_features(frame["mes"], frame["dia_preferido_pago"], include_flags=flags).reset_index(drop=True)


def family_extras(name: str, frame: pd.DataFrame, matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Extra columns and columns to drop for family ``name`` on top of ``matrix``.

    ``frame`` is the raw labelled frame (it provides ``objetivo`` only for F4, which is
    strictly causal); ``matrix`` is the base feature matrix for the same rows.
    """
    if name == "F1_calendar":
        return _calendar(frame, False), []
    if name == "F1g_calendar_flags":
        return _calendar(frame, True), []
    if name == "F2_ratios":
        return behavioral_ratio_features(frame), []
    if name == "F3_rank_all":
        return within_month_rank_features(frame, NUMERIC_COLUMNS), []
    if name == "F3r_rank_replace":
        return within_month_rank_features(frame, NUMERIC_COLUMNS), [c for c in NUMERIC_COLUMNS if c in matrix.columns]
    if name == "F4_target_enc":
        return te_features(frame), []
    if name == "F5_cohort":
        extra = cohort_features(frame, NUMERIC_COLUMNS)
        if "meses_en_riesgo" not in matrix.columns:
            tenure = frame.groupby("id_cliente")["mes"].transform("min")
            first_ord = (tenure // 100) * 12 + tenure % 100
            cur_ord = (frame["mes"] // 100) * 12 + frame["mes"] % 100
            extra["meses_en_riesgo"] = (cur_ord - first_ord + 1).to_numpy()
        return extra, []
    raise ValueError(f"unknown family {name}")


FAMILY_NAMES = ["F1_calendar", "F1g_calendar_flags", "F2_ratios", "F3_rank_all", "F3r_rank_replace",
                "F4_target_enc", "F5_cohort"]


def apply_family(name: str, frame: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    extra, drop = family_extras(name, frame.reset_index(drop=True), matrix)
    out = matrix.drop(columns=drop).reset_index(drop=True)
    clash = set(extra.columns) & set(out.columns)
    if clash:
        raise ValueError(f"family {name} duplicates columns {sorted(clash)}")
    return pd.concat([out, extra.reset_index(drop=True)], axis=1)
