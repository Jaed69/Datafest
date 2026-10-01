import pandas as pd


HISTORY_FEATURES = [
    "mes_primera_aparicion",
    "meses_en_riesgo",
    "n_observaciones_previas",
    "cliente_recurrente",
    "meses_desde_entrada",
]

CATEGORICAL_COLUMNS = [
    "ocupacion",
    "region",
    "canal_adquisicion",
    "banda_riesgo",
    "dispositivo_principal",
]


def _month_ordinal(month: pd.Series) -> pd.Series:
    values = month.astype("int64")
    return (values // 100) * 12 + values % 100 - 1


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"id_cliente", "mes"}.issubset(frame.columns):
        raise ValueError("Se requieren id_cliente y mes para crear features de historial")
    result = frame.copy()
    result["mes"] = result["mes"].astype("int64")
    original_order = pd.Series(range(len(result)), index=result.index)
    sorted_result = result.assign(_original_order=original_order.values).sort_values(
        ["mes", "id_cliente", "_original_order"], kind="mergesort"
    )
    month_ordinal = _month_ordinal(sorted_result["mes"])
    sorted_result["_month_ordinal"] = month_ordinal
    sorted_result["n_observaciones_previas"] = sorted_result.groupby(
        "id_cliente", sort=False
    ).cumcount()
    first_seen = sorted_result.groupby("id_cliente", sort=False)["_month_ordinal"].transform("min")
    sorted_result["mes_primera_aparicion"] = (
        sorted_result.groupby("id_cliente", sort=False)["mes"].transform("min").astype("int64")
    )
    sorted_result["meses_desde_entrada"] = (month_ordinal - first_seen).astype("int64")
    sorted_result["meses_en_riesgo"] = sorted_result["meses_desde_entrada"] + 1
    sorted_result["cliente_recurrente"] = sorted_result["n_observaciones_previas"].gt(0)
    sorted_result = sorted_result.sort_values("_original_order", kind="mergesort")
    return sorted_result.drop(columns=["_original_order", "_month_ordinal"]).reset_index(drop=True)


def features_for_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant not in {"clean", "history", "id_raw"}:
        raise ValueError(f"Variante de features desconocida: {variant}")
    feature_input = frame.drop(columns=["objetivo"], errors="ignore")
    transformed = add_history_features(feature_input) if variant == "history" else feature_input.copy()
    excluded = {"objetivo"}
    if variant != "id_raw":
        excluded.add("id_cliente")
    if variant != "history":
        excluded.update(HISTORY_FEATURES)
    return transformed.drop(columns=[column for column in excluded if column in transformed.columns])
