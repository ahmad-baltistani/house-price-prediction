# House Price Prediction

A machine learning project for predicting house prices using the California Housing dataset.

This project is being developed as an end-to-end ML Engineering project, with a focus on building a clean, modular, testable, and production-ready machine learning workflow.

## Project Structure

```text
house-price-prediction/
│
├── data/
│   ├── raw/
│   │   └── housing.csv
│   └── processed/
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── training_pipeline.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── split_data.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── save_artifacts.py
│       └── load_artifacts.py
│
├── models/
│   ├── model.joblib
│   └── preprocessor.joblib
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_utils.py
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

> The generated model artifacts inside the `models/` directory are ignored by Git.

## Current ML Pipeline

The training pipeline performs the following steps:

1. Load the California Housing dataset.
2. Separate input features and the target variable.
3. Split the data into training and testing sets.
4. Build and fit the preprocessing pipeline.
5. Transform the training and testing data.
6. Train a Linear Regression model.
7. Evaluate the trained model.
8. Save the trained model and fitted preprocessor.

## Data Preprocessing

The preprocessing pipeline handles both numerical and categorical features.

### Numerical Features

- Missing values are filled using median imputation.
- Features are standardized using `StandardScaler`.

### Categorical Features

- Missing values are filled using the most frequent value.
- Categories are encoded using one-hot encoding.
- Unknown categories in new data are handled without breaking the pipeline.

The preprocessor is fitted only on the training data. Test data and new prediction data are transformed using the already fitted preprocessor to avoid data leakage.

## Model

The current model is:

```text
Linear Regression
```

The model is trained using the processed training data.

## Model Results

Current performance on the test set:

```text
RMSE: ~70,059
R²:   ~0.625
```

These metrics are used as the current baseline for the project.

## Model Artifacts

After training, the project saves two artifacts:

```text
models/model.joblib
models/preprocessor.joblib
```

- `model.joblib` contains the trained Linear Regression model.
- `preprocessor.joblib` contains the fitted preprocessing pipeline.

The saved artifacts can be loaded later for prediction without retraining the model.

The generated artifact files are excluded from Git using `.gitignore`.

## Automated Testing

Automated tests are written using `pytest`.

The test suite currently covers:

- Train/test splitting
- Feature and target alignment after splitting
- Numerical missing-value preprocessing
- Categorical preprocessing
- Handling unseen categories
- Model training and prediction
- Model evaluation
- Saving and loading artifacts
- Training pipeline integration
- Verification of model and preprocessor save operations

Unit tests use small controlled datasets so that they remain fast and predictable.

The pipeline integration test uses `monkeypatch` to replace external operations such as loading the real dataset and writing model artifacts while keeping the main ML pipeline components real.

Temporary files are tested using pytest's `tmp_path` fixture so that tests do not modify the project's real model artifacts.

### Run Tests

```bash
python -m pytest -v
```

Current status:

```text
6 tests passed
```

## Run the Training Pipeline

The complete training pipeline can be executed from the project root using:

```bash
python -m src.training_pipeline
```

The pipeline trains and evaluates the model and then saves the trained model and fitted preprocessor.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Pytest
- Git
- GitHub

## Current Development Status

Completed stages:

```text
01 — Git & GitHub
02 — Professional ML Project Structure
03 — Python Modules
04 — Training Pipeline
05 — Model Saving
06 — Testing
```

The project will continue to evolve as more ML engineering concepts are implemented.