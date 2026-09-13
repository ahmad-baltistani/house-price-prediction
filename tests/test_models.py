import numpy as np

from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model


def test_train_model():
    X_train = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y_train = np.array([2, 4, 6, 8])

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")


def test_evaluate_model():
    X_train = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y_train = np.array([2, 4, 6, 8])

    model = train_model(X_train, y_train)

    rmse, r2 = evaluate_model(
        model,
        X_train,
        y_train
    )

    assert rmse >= 0
    assert r2 <= 1