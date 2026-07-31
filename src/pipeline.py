"""
Prediction pipeline module.

This module is responsible for:
- Loading the trained model
- Preparing input data
- Making predictions
"""

import joblib

import pandas as pd

from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.encoding import encode_features

from src.config import (
    MODEL_PATH,
    FEATURE_NAMES_PATH
)


def load_model():
    """
    Load the trained model.
    """

    try:
        return joblib.load(MODEL_PATH)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model before making predictions."
        )



def prepare_input(df):
    """
    Apply all preprocessing steps
    before prediction.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = preprocess_data(
        df,
        training=False
    )

    df = engineer_features(df)

    df = encode_features(df)
    
    feature_names = load_feature_names()

    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    return df



def predict_price(df):
    """
    Predict house prices.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    numpy.ndarray
    """

    model = load_model()

    df = prepare_input(df)

    predictions = model.predict(df)

    return predictions



def load_feature_names():
    """
    Load the feature names used during training.

    Returns
    -------
    list
        List of feature names.
    """

    return joblib.load(FEATURE_NAMES_PATH)



