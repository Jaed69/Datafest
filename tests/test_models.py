import numpy as np
import pandas as pd
import pytest

from datafest.models import MODEL_CONFIGS, fit_model


@pytest.mark.parametrize("model_name", ["lightgbm", "catboost"])
def test_models_train_with_unseen_categories_and_return_probabilities(model_name):
    train_x = pd.DataFrame(
        {
            "edad": np.arange(40),
            "ocupacion": ["office", "field"] * 20,
            "activo_movil": [True, False] * 20,
        }
    )
    train_y = np.array([0, 1] * 20)
    valid_x = pd.DataFrame(
        {
            "edad": [21, 42, 35, 19],
            "ocupacion": ["office", "unseen", "field", "unseen"],
            "activo_movil": [True, False, True, False],
        }
    )
    valid_y = np.array([0, 1, 1, 0])

    model = fit_model(
        model_name,
        train_x,
        train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        config=MODEL_CONFIGS[model_name][0],
        seed=42,
        iterations=50,
        early_stopping_rounds=5,
    )
    prediction = model.predict_proba(valid_x)

    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    assert model.best_iteration >= 1
    assert "unseen" not in model.category_maps["ocupacion"]
