import pandas as pd

from src.features.split_data import split_data
from src.features.preprocess import create_preprocessor


def test_split_data():
    df = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100),
        "target": range(100)
    })

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

    assert (X_train["feature1"] == y_train).all()
    assert (X_test["feature1"] == y_test).all()


def test_create_preprocessor():
    import numpy as np
    df = pd.DataFrame({
        "age": [10, 20, None, 40],
        "income": [100, 200, 300, 400],
        "location": ["A", "B", "A", "B"]
    })

    preprocessor = create_preprocessor(df)

    X_processed = preprocessor.fit_transform(df)

    assert X_processed.shape[0] == 4
    assert X_processed.shape[1] == 4

    new_data = pd.DataFrame({
    "age": [25],
    "income": [250],
    "location": ["C"]
    })
    new_processed = preprocessor.transform(new_data)
    assert new_processed.shape[0] == 1
    assert new_processed.shape[1] == 4

    assert not np.isnan(X_processed).any()

