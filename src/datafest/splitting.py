from dataclasses import dataclass
import hashlib

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    train_through: int
    valid_month: int
    train: pd.DataFrame
    valid: pd.DataFrame


def make_outer_temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(int(value) for value in frame["mes"].unique())
    if len(months) < 3:
        raise ValueError("Se requieren al menos tres meses para el holdout temporal")
    validation_months = set(months[-2:])
    train = frame.loc[~frame["mes"].isin(validation_months)].copy()
    valid = frame.loc[frame["mes"].isin(validation_months)].copy()
    if train.empty or valid.empty or train["mes"].max() >= valid["mes"].min():
        raise ValueError("Corte temporal inválido: train debe preceder a validación")
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def make_inner_temporal_folds(
    frame: pd.DataFrame, end_month: int | None = None
) -> list[TemporalFold]:
    months = sorted(int(value) for value in frame["mes"].unique())
    if end_month is not None:
        months = [month for month in months if month <= end_month]
    if len(months) < 5:
        raise ValueError("Se requieren al menos cinco meses para crear folds internos")
    validation_months = months[-4:]
    folds = []
    for valid_month in validation_months:
        train_months = [month for month in months if month < valid_month]
        train = frame.loc[frame["mes"].isin(train_months)].copy().reset_index(drop=True)
        valid = frame.loc[frame["mes"].eq(valid_month)].copy().reset_index(drop=True)
        folds.append(TemporalFold(max(train_months), valid_month, train, valid))
    return folds


def _holdout_client(client_id: object, denominator: int = 5) -> bool:
    digest = hashlib.sha256(str(client_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % denominator == 0


def make_group_diagnostic_split(
    frame: pd.DataFrame,
    holdout_month: int,
    holdout_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction debe estar entre 0 y 1")
    # Use a deterministic fifth of customer IDs so the split is stable and auditable.
    if abs(holdout_fraction - 0.2) > 1e-12:
        raise ValueError("El diagnóstico por grupos usa una fracción fija de 20%")
    held_out = frame["id_cliente"].map(_holdout_client)
    train = frame.loc[(frame["mes"] < holdout_month) & ~held_out].copy()
    valid = frame.loc[(frame["mes"] == holdout_month) & held_out].copy()
    if train.empty or valid.empty:
        raise ValueError("El split por grupos produjo train o validación vacíos")
    if set(train["id_cliente"]) & set(valid["id_cliente"]):
        raise ValueError("El split por grupos comparte clientes entre train y validación")
    return train.reset_index(drop=True), valid.reset_index(drop=True)
