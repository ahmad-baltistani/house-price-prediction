import pandas as pd

import src.training_pipeline as training_pipeline

def test_training_pipeline(monkeypatch):
    fake_df = pd.DataFrame({
        "feature1": range(20),
        "feature2": range(20, 40),
        "location": ["A", "B"] * 10,
        "median_house_value": [x * 2 for x in range(20)]
    })

    monkeypatch.setattr(
        training_pipeline,
        "load_data",
        lambda path: fake_df
    )

    saved_artifacts = []

    def fake_save_artifact(artifact, path):
        saved_artifacts.append((artifact, path))
        
    monkeypatch.setattr(
        training_pipeline,
        "save_artifact",
        fake_save_artifact
    )

    model, preprocessor, rmse, r2 = training_pipeline.run_training_pipeline()
    
    assert model is not None
    assert preprocessor is not None
    assert isinstance(rmse, float)
    assert isinstance(r2, float)
    assert rmse >= 0
    assert len(saved_artifacts) == 2
    assert saved_artifacts[0][1] == training_pipeline.MODEL_PATH
    assert saved_artifacts[1][1] == training_pipeline.PREPROCESSOR_PATH
    assert saved_artifacts[0][0] is model
    assert saved_artifacts[1][0] is preprocessor

    