from datetime import date

import pandas as pd
import pytest

from datafest.calendar_features import (
    PERU_HOLIDAYS_2026,
    business_days_in_month,
    calendar_features,
    days_to_business_day,
    is_business_day,
)


def test_holiday_table_has_the_sixteen_official_2026_dates():
    assert len(PERU_HOLIDAYS_2026) == 16
    assert date(2026, 12, 8) in PERU_HOLIDAYS_2026
    assert date(2026, 7, 28) in PERU_HOLIDAYS_2026 and date(2026, 7, 29) in PERU_HOLIDAYS_2026
    assert all(d.year == 2026 for d in PERU_HOLIDAYS_2026)


def test_dec_8_2026_is_a_tuesday_holiday_followed_by_another_holiday():
    assert date(2026, 12, 8).weekday() == 1
    assert not is_business_day(date(2026, 12, 8))
    assert not is_business_day(date(2026, 12, 9))  # Battle of Ayacucho
    # Dec 8 and 9 are holidays, so the next business day is Thursday Dec 10.
    assert days_to_business_day(date(2026, 12, 8)) == 2
    assert days_to_business_day(date(2026, 12, 10)) == 0


def test_weekend_rolls_to_monday_and_friday_holiday_chain():
    assert days_to_business_day(date(2026, 12, 5)) == 2  # Saturday -> Monday
    assert days_to_business_day(date(2026, 12, 25)) == 3  # holiday Friday -> Monday


def test_business_days_in_december_2026_exclude_weekends_and_three_holidays():
    # December 2026 has 23 weekdays; Dec 8, 9 and 25 are weekday holidays.
    assert business_days_in_month(2026, 12) == 20


def test_calendar_features_for_dec_8_pay_day():
    frame = pd.DataFrame({"mes": [202612], "dia_preferido_pago": [8]})
    out = calendar_features(frame["mes"], frame["dia_preferido_pago"])
    row = out.iloc[0]
    assert row["pay_weekday"] == 1
    assert row["pay_is_weekend"] == 0
    assert row["pay_is_holiday"] == 1
    assert row["pay_days_to_business_day"] == 2
    assert row["month_business_days"] == 20
    assert row["month_holidays"] == 3
    assert row["month_days"] == 31


def test_calendar_features_weekend_pay_day_and_row_alignment():
    frame = pd.DataFrame({"mes": [202601, 202601, 202606], "dia_preferido_pago": [3, 4, 7]})
    out = calendar_features(frame["mes"], frame["dia_preferido_pago"])
    # Jan 3 2026 is Saturday, Jan 4 Sunday; Jun 7 2026 is a Sunday holiday.
    assert out["pay_is_weekend"].tolist() == [1, 1, 1]
    assert out["pay_weekday"].tolist() == [5, 6, 6]
    assert out["pay_is_holiday"].tolist() == [0, 0, 1]
    assert out["pay_days_to_business_day"].tolist() == [2, 1, 1]
    assert out.index.equals(frame.index)


def test_gratification_and_cts_flags_are_optional():
    months = pd.Series([202605, 202607, 202611, 202612, 202603])
    days = pd.Series([15] * 5)
    plain = calendar_features(months, days)
    assert "gratificacion_month" not in plain.columns and "cts_month" not in plain.columns
    flagged = calendar_features(months, days, include_flags=True)
    assert flagged["gratificacion_month"].tolist() == [0, 1, 0, 1, 0]
    assert flagged["cts_month"].tolist() == [1, 0, 1, 0, 0]


def test_rejects_months_outside_2026_holiday_table():
    with pytest.raises(ValueError):
        calendar_features(pd.Series([202701]), pd.Series([5]))
