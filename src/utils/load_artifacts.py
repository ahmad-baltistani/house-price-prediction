import joblib
from pathlib import Path


def load_artifact(file_path):
    file_path = Path(file_path)
    return joblib.load(file_path)