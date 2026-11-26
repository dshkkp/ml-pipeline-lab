
import numpy as np
import pandas as pd
import pytest

from dataset import make_dataset
from pipeline import ( FEATURES, train_model, predict_churn, predict_churn_probability,)
from metrics import evaluate_model
from model_io import save_model, load_model


@pytest.fixture(scope="module")
def trained_model():
    """Один раз обучаем модель для нескольких тестов."""
    df = make_dataset()
    return train_model(df)


def test_dataset():
    """Датасет создаётся с правильной структурой."""
    df = make_dataset()

    assert df.shape == (300, 6)
    assert set(FEATURES).issubset(df.columns)
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()


def test_dataset_reproducibility():
    """Одинаковый random_state даёт одинаковые данные."""
    df1 = make_dataset(random_state=42)
    df2 = make_dataset(random_state=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_training(trained_model):
    """Модель обучается и возвращает тестовую выборку."""
    model, X_test, y_test = trained_model

    assert len(X_test) == 60
    assert len(y_test) == 60
    assert hasattr(model, "predict")


def test_predictions(trained_model):
    """Модель возвращает корректные классы."""
    model, X_test, _ = trained_model

    predictions = predict_churn(model, X_test)

    assert len(predictions) == len(X_test)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_probabilities(trained_model):
    """Вероятности находятся в диапазоне от 0 до 1."""
    model, X_test, _ = trained_model

    probabilities = predict_churn_probability(model, X_test)

    assert len(probabilities) == len(X_test)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_metrics(trained_model):
    """Все необходимые метрики вычисляются."""
    model, X_test, y_test = trained_model

    results = evaluate_model(model, X_test, y_test)

    expected = { "accuracy", "precision", "recall", "f1", "roc_auc"}

    assert set(results) == expected

    for value in results.values():
        assert 0 <= value <= 1


def test_model_save_load(trained_model, tmp_path):
    """Предсказания сохраняются после загрузки модели."""
    model, X_test, _ = trained_model

    path = tmp_path / "model.joblib"

    save_model(model, path)
    restored_model = load_model(path)

    original_predictions = model.predict(X_test)
    restored_predictions = restored_model.predict(X_test)

    np.testing.assert_array_equal( original_predictions, restored_predictions)
