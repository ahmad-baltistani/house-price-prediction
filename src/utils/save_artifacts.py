import joblib
from pathlib import Path


def save_artifact(artifact, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, file_path)