"""
Model training module.

This module performs:
- Data preparation
- Train-test split
- Model training
- Model evaluation
- Model saving
"""

from xml.parsers.expat import model

from pandas.api.types import is_numeric_dtype

import joblib

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.feature_engineering import engineer_features
from src.encoding import encode_features

from src.config import (
    TARGET_COLUMN,
    RANDOM_STATE,
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    RAW_DEFAULTS_PATH,
    XGBOOST_PARAMS
)



def prepare_data():
    """
    Prepare the dataset for model training.

    Returns
    -------
    X : pd.DataFrame

    y : pd.Series
    """

    df = load_data()

    df = preprocess_data(df)

    df = engineer_features(df)

    raw_df = df.copy()

    df = encode_features(df)

    X = df.drop(columns=TARGET_COLUMN)

    y = df[TARGET_COLUMN]

    return X, y, raw_df



def split_data(X, y):
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    X : pd.DataFrame

    y : pd.Series

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    
    
    
def train_model(X_train, y_train):
    """
    Train the final XGBoost model.
    Returns
    -------
    XGBRegressor
        Trained model.
    """

    model = XGBRegressor(**XGBOOST_PARAMS)

    model.fit(X_train, y_train)

    return model



def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = root_mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nModel Performance")
    print("-" * 30)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    
    
    return {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}
    
    
    
def save_feature_names(feature_names):
    """
    Save the feature names used during training.

    Parameters
    ----------
    feature_names : list
        List of feature names.
    """

    FEATURE_NAMES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    print(f"\nFeature names saved to:\n{FEATURE_NAMES_PATH}")




def save_raw_defaults(df):
    """
    Save default values for every feature before encoding.

    Numerical columns -> median
    Categorical columns -> mode
    """

    defaults = {}

    for column in df.columns:

        if column == TARGET_COLUMN:
            continue

        if is_numeric_dtype(df[column]):
            defaults[column] = df[column].median()
        else:
            defaults[column] = df[column].mode()[0]

    RAW_DEFAULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(defaults, RAW_DEFAULTS_PATH)

    print(
        f"\nRaw default values saved to:\n"
        f"{RAW_DEFAULTS_PATH}"
    )
        
    
    

def save_model(model):
    """
    Save the trained model.
    """

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to:\n{MODEL_PATH}")



def main():

    print("Preparing data...")

    X, y, raw_df = prepare_data()
    
    save_feature_names(X.columns.tolist())
    
    save_raw_defaults(raw_df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    print("Training model...")

    model = train_model(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(model)
    
    

if __name__ == "__main__":
    main()