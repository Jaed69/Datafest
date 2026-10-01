from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataReport:
    train_rows: int
    test_rows: int
    train_months: tuple[int, ...]
    test_months: tuple[int, ...]
    missing_values: int
    duplicate_client_months: int
    test_rows_with_seen_client: int
    test_rows_with_unseen_client: int


def validate_competition_data(train: pd.DataFrame, test: pd.DataFrame) -> DataReport:
    required_train = {"id_cliente", "mes", "objetivo"}
    required_test = {"id_cliente", "mes"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"train.csv no contiene columnas requeridas: {required_train - set(train.columns)}")
    if not required_test.issubset(test.columns):
        raise ValueError(f"test.csv no contiene columnas requeridas: {required_test - set(test.columns)}")
    if "objetivo" in test.columns:
        raise ValueError("test.csv no debe contener objetivo")

    train_features = [column for column in train.columns if column != "objetivo"]
    if train_features != list(test.columns):
        raise ValueError("Las columnas de train y test deben coincidir, salvo objetivo")

    missing = int(train.isna().sum().sum() + test.isna().sum().sum())
    if missing:
        raise ValueError(f"Se encontraron {missing} valores faltantes")

    dup_train = int(train.duplicated(["id_cliente", "mes"]).sum())
    dup_test = int(test.duplicated(["id_cliente", "mes"]).sum())
    if dup_train or dup_test:
        raise ValueError(f"Se encontraron claves cliente-mes duplicadas: train={dup_train}, test={dup_test}")

    target = set(pd.Series(train["objetivo"]).dropna().unique().tolist())
    if not target.issubset({0, 1, False, True}):
        raise ValueError(f"objetivo debe ser binario; valores encontrados: {target}")
    if len(target) != 2:
        raise ValueError("objetivo debe contener ambas clases")

    train_months = tuple(sorted(int(value) for value in train["mes"].unique()))
    test_months = tuple(sorted(int(value) for value in test["mes"].unique()))
    if not train_months or not test_months:
        raise ValueError("train y test deben contener meses")
    if max(train_months) >= min(test_months):
        raise ValueError("test debe ser posterior a todos los meses de train")
    for month in (*train_months, *test_months):
        if month % 100 < 1 or month % 100 > 12:
            raise ValueError(f"mes debe tener formato AAAAMM válido: {month}")

    converted = train.loc[train["objetivo"].astype(int).eq(1), ["id_cliente", "mes"]]
    if converted["id_cliente"].duplicated().any():
        raise ValueError("Un cliente no puede registrar más de una primera conversión")
    if not converted.empty:
        conversion_month = converted.set_index("id_cliente")["mes"]
        row_conversion_month = train["id_cliente"].map(conversion_month)
        if (row_conversion_month.notna() & train["mes"].gt(row_conversion_month)).any():
            raise ValueError("Un cliente aparece después de su primera conversión")

    known_ids = set(train["id_cliente"])
    seen_rows = int(test["id_cliente"].isin(known_ids).sum())

    return DataReport(
        train_rows=len(train),
        test_rows=len(test),
        train_months=train_months,
        test_months=test_months,
        missing_values=missing,
        duplicate_client_months=dup_train + dup_test,
        test_rows_with_seen_client=seen_rows,
        test_rows_with_unseen_client=len(test) - seen_rows,
    )


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(submission.columns) != ["id_cliente", "prediccion"]:
        raise ValueError("La submission debe tener exactamente id_cliente,prediccion")
    if len(submission) != len(test):
        raise ValueError("La submission debe tener una fila por fila de test")
    if not np.array_equal(submission["id_cliente"].to_numpy(), test["id_cliente"].to_numpy()):
        raise ValueError("Los id_cliente o su orden no coinciden con test")
    prediction = pd.to_numeric(submission["prediccion"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("prediccion debe contener probabilidades finitas entre 0 y 1")
