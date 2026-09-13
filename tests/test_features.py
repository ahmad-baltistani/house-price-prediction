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


def test_create_preprocessor():
    df = pd.DataFrame({
        "age": [10, 20, 30, 40],
        "income": [100, 200, 300, 400],
        "location": ["A", "B", "A", "B"]
    })

    preprocessor = create_preprocessor(df)

    X_processed = preprocessor.fit_transform(df)

    assert X_processed.shape[0] == 4
    assert X_processed.shape[1] == 4

