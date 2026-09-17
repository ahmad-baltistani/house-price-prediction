from src.config import RAW_DATA_DIR, TARGET_COLUMN
from src.data.load_data import load_data
from src.features.split_data import split_data
from src.features.preprocess import create_preprocessor
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model

def run_training_pipeline():
    df = load_data(RAW_DATA_DIR / "housing.csv")
    
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    preprocessor = create_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = train_model(X_train_processed, y_train)

    rmse, r2 = evaluate_model(model, X_test_processed, y_test) 

    return model, preprocessor, rmse, r2

if __name__ == "__main__": 
    _, _, rmse, r2 = run_training_pipeline()

    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")
    