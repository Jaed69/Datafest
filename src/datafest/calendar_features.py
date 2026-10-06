"""Calendar / seasonality features for the client-month pay day (round 2, F1).

``dia_preferido_pago`` is a day of the month (1-28) and ``mes`` a YYYYMM
integer, so every row maps to one concrete 2026 date. Features describe that
date (weekday, weekend, Peruvian public holiday, distance to the next business
day) and the month it sits in (business days, holidays, length).

Holidays: the 16 national public holidays listed by the Peruvian state portal
for 2026 (https://www.gob.pe/feriados), cross-checked against press listings
(El Comercio, Gestion, Infobae, La Republica; consulted 2026-10-06). Optional
"dias no laborables" (bridge days) are decided by the government during the year
and are NOT included, so they are treated as ordinary business days.

``gratificacion_month`` (Jul, Dec) and ``cts_month`` (May, Nov) mark the legal
Peruvian bonus / CTS deposit months. No gratificacion month falls inside the
Sep-Nov validation folds, so that flag cannot be validated; the test month
(December) is one.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

import pandas as pd

PERU_HOLIDAYS_2026: frozenset[date] = frozenset({
    date(2026, 1, 1),    # Anio Nuevo
    date(2026, 4, 2),    # Jueves Santo
    date(2026, 4, 3),    # Viernes Santo
    date(2026, 5, 1),    # Dia del Trabajo
    date(2026, 6, 7),    # Batalla de Arica y Dia de la Bandera (Sunday)
    date(2026, 6, 29),   # San Pedro y San Pablo
    date(2026, 7, 23),   # Dia de la Fuerza Aerea del Peru
    date(2026, 7, 28),   # Fiestas Patrias
    date(2026, 7, 29),   # Fiestas Patrias
    date(2026, 8, 6),    # Batalla de Junin
    date(2026, 8, 30),   # Santa Rosa de Lima (Sunday)
    date(2026, 10, 8),   # Combate de Angamos
    date(2026, 11, 1),   # Todos los Santos (Sunday)
    date(2026, 12, 8),   # Inmaculada Concepcion
    date(2026, 12, 9),   # Batalla de Ayacucho
    date(2026, 12, 25),  # Navidad
})
GRATIFICACION_MONTHS = frozenset({7, 12})
CTS_MONTHS = frozenset({5, 11})
CALENDAR_COLUMNS = [
    "pay_weekday", "pay_is_weekend", "pay_is_holiday", "pay_days_to_business_day",
    "month_business_days", "month_holidays", "month_days",
]
FLAG_COLUMNS = ["gratificacion_month", "cts_month"]


def is_business_day(day: date, holidays: frozenset[date] = PERU_HOLIDAYS_2026) -> bool:
    return day.weekday() < 5 and day not in holidays


def days_to_business_day(day: date, holidays: frozenset[date] = PERU_HOLIDAYS_2026) -> int:
    """0 when ``day`` is a business day, else days until the next business day."""
    offset = 0
    while not is_business_day(day + timedelta(days=offset), holidays):
        offset += 1
    return offset


def business_days_in_month(year: int, month: int, holidays: frozenset[date] = PERU_HOLIDAYS_2026) -> int:
    last = calendar.monthrange(year, month)[1]
    return sum(is_business_day(date(year, month, d), holidays) for d in range(1, last + 1))


def calendar_features(
    months: pd.Series, pay_day: pd.Series, include_flags: bool = False,
    holidays: frozenset[date] = PERU_HOLIDAYS_2026,
) -> pd.DataFrame:
    """One row per input row, aligned to the input index."""
    months = pd.Series(months).astype("int64")
    pay_day = pd.Series(pay_day).astype("int64")
    if not months.floordiv(100).eq(2026).all():
        raise ValueError("holiday table only covers 2026")
    rows = {}
    for month, day in set(zip(months.tolist(), pay_day.tolist())):
        year, mm = divmod(month, 100)
        pay = date(year, mm, day)
        last = calendar.monthrange(year, mm)[1]
        rows[(month, day)] = {
            "pay_weekday": pay.weekday(),
            "pay_is_weekend": int(pay.weekday() >= 5),
            "pay_is_holiday": int(pay in holidays),
            "pay_days_to_business_day": days_to_business_day(pay, holidays),
            "month_business_days": business_days_in_month(year, mm, holidays),
            "month_holidays": sum(1 for h in holidays if h.year == year and h.month == mm),
            "month_days": last,
        }
    out = pd.DataFrame([rows[key] for key in zip(months.tolist(), pay_day.tolist())], index=months.index)
    out = out[CALENDAR_COLUMNS].astype("int64")
    if include_flags:
        month_of_year = months % 100
        out["gratificacion_month"] = month_of_year.isin(GRATIFICACION_MONTHS).astype("int64")
        out["cts_month"] = month_of_year.isin(CTS_MONTHS).astype("int64")
    return out
