"""
Prediction pipeline module.

This module is responsible for:
- Loading the trained model
- Preparing input data
- Making predictions
"""

import joblib

import pandas as pd

from src.preprocessing import (
    preprocess_data,
    load_data
)
from src.feature_engineering import engineer_features
from src.encoding import encode_features

from src.config import (
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    RAW_DEFAULTS_PATH,
    TARGET_COLUMN
)

from pandas.api.types import is_numeric_dtype


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




def load_feature_names():
    """
    Load the feature names used during training.

    Returns
    -------
    list
        List of feature names.
    """

    try:
        return joblib.load(FEATURE_NAMES_PATH)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Feature names not found at {FEATURE_NAMES_PATH}. "
            "Train the model before making predictions."
        )




def prepare_input(df, feature_names):
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
    
    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    return df



def predict_price(user_input, model, feature_names):
    """
    Predict house prices.

    Parameters
    ----------
    user_input : dict

    model : XGBRegressor

    feature_names : list

    Returns
    -------
    numpy.ndarray
    """

    df = build_input_dataframe(user_input)

    df = prepare_input(
        df,
        feature_names
    )

    predictions = model.predict(df)

    return predictions



def load_raw_defaults():
    """
    Load the raw default feature values.

    Returns
    -------
    dict
    """

    return joblib.load(RAW_DEFAULTS_PATH)



def build_input_dataframe(user_input):

    defaults = load_raw_defaults()

    complete_data = defaults.copy()

    for key, value in user_input.items():

        if value != "":

            complete_data[key] = value

    df = pd.DataFrame([complete_data])

    sample = load_data().drop(columns=[TARGET_COLUMN])

    for column in sample.columns:

        if is_numeric_dtype(sample[column]):

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df