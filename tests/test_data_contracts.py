from pathlib import Path

import pandas as pd
import pytest

from datafest.data import validate_competition_data, validate_submission


ROOT = Path(__file__).resolve().parents[1]


def test_competition_files_match_declared_contract():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    report = validate_competition_data(train, test)

    assert report.train_rows == 110_100
    assert report.test_rows == 9_900
    assert report.train_months == tuple(range(202601, 202612))
    assert report.test_months == (202612,)
    assert report.missing_values == 0
    assert report.duplicate_client_months == 0
    assert report.test_rows_with_seen_client == 8_061
    assert report.test_rows_with_unseen_client == 1_839


def test_submission_preserves_test_ids_order_and_exact_columns():
    test = pd.read_csv(ROOT / "data" / "test.csv")
    expected = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    submission = expected.copy()

    validate_submission(submission, test)


@pytest.mark.parametrize("bad_prediction", [-0.01, 1.01, float("nan")])
def test_submission_rejects_invalid_probabilities(bad_prediction):
    test = pd.DataFrame({"id_cliente": [10, 11]})
    submission = pd.DataFrame(
        {"id_cliente": [10, 11], "prediccion": [0.2, bad_prediction]}
    )

    with pytest.raises(ValueError, match="prediccion"):
        validate_submission(submission, test)


def test_submission_rejects_reordered_or_missing_ids():
    test = pd.DataFrame({"id_cliente": [10, 11]})
    reordered = pd.DataFrame(
        {"id_cliente": [11, 10], "prediccion": [0.2, 0.8]}
    )

    with pytest.raises(ValueError, match="orden"):
        validate_submission(reordered, test)


def test_data_validation_rejects_duplicate_client_month():
    train = pd.DataFrame(
        {
            "id_cliente": [1, 1],
            "mes": [202601, 202601],
            "objetivo": [0, 1],
        }
    )
    test = pd.DataFrame({"id_cliente": [2], "mes": [202602]})

    with pytest.raises(ValueError, match="duplicadas"):
        validate_competition_data(train, test)


def test_data_validation_rejects_rows_after_first_conversion():
    train = pd.DataFrame(
        {
            "id_cliente": [1, 1],
            "mes": [202601, 202602],
            "objetivo": [1, 0],
        }
    )
    test = pd.DataFrame({"id_cliente": [2], "mes": [202603]})

    with pytest.raises(ValueError, match="después de su primera conversión"):
        validate_competition_data(train, test)


def test_data_validation_rejects_test_schema_drift():
    train = pd.DataFrame(
        {"id_cliente": [1, 2], "mes": [202601, 202601], "x": [1, 2], "objetivo": [0, 1]}
    )
    test = pd.DataFrame({"id_cliente": [3], "mes": [202602], "x": [3], "extra": [0]})

    with pytest.raises(ValueError, match="columnas de train y test"):
        validate_competition_data(train, test)
