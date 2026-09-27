from src.utils.save_artifacts import save_artifact
from src.utils.load_artifacts import load_artifact


def test_save_and_load_artifact(tmp_path):
    test_path = tmp_path / "test_artifact.joblib"

    original_object = {
        "name": "test",
        "score": 100
    }

    save_artifact(original_object, test_path)

    loaded_object = load_artifact(test_path)

    assert test_path.exists()
    assert loaded_object == original_object

    