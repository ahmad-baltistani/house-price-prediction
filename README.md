# House Price Prediction

A machine learning project for predicting house prices using the California Housing dataset.

## Project Structure

```text
house-price-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
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
│       └── __init__.py
│
├── models/
├── tests/
│   ├── test_features.py
│   └── test_models.py
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## Current Pipeline

- Load California Housing data
- Split features and target
- Train/test split
- Numerical preprocessing: median imputation + standard scaling
- Categorical preprocessing: most-frequent imputation + one-hot encoding
- Train Linear Regression model
- Evaluate model using RMSE and R²

## Model Results

- RMSE: ~70,059
- R²: ~0.625

## Testing

Automated tests are written using pytest.

```bash
pytest
```

Current status: **4 tests passed**