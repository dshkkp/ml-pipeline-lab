
from pathlib import Path

import joblib


def save_model(model, path):
    """Сохраняет обученную модель в файл"""

    path = Path(path)

    joblib.dump(model, path)


def load_model(path):
    """Загружает обученную модель из файла"""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)
