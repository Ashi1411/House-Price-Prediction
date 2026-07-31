"""
Data preprocessing module.

This module contains functions for:
- Loading the dataset
- Handling missing values
- Removing training outliers
"""

import pandas as pd

from src.config import (
    DATA_PATH,
    GARAGE_CATEGORICAL,
    GARAGE_NUMERICAL,
    BASEMENT_CATEGORICAL,
    BASEMENT_NUMERICAL,
    MASONRY_CATEGORICAL,
    MASONRY_NUMERICAL,
    FIREPLACE_FEATURE,
    POOL_FEATURE,
    LOT_FRONTAGE,
    ELECTRICAL,
    NEIGHBORHOOD,
    FENCE_FEATURE,
    SPECIAL_NEIGHBORHOODS,
    NONE_CATEGORY,
    ZERO_VALUE
)


def load_data():
    """
    Load the Ames Housing dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """

    return pd.read_csv(DATA_PATH)


def fill_missing_values(df):
    """
    Fill missing values using the same strategy
    used during model training.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # ======================================
    # Garage Features
    # ======================================

    for col in GARAGE_CATEGORICAL:
        df[col] = df[col].fillna(NONE_CATEGORY)

    for col in GARAGE_NUMERICAL:
        df[col] = df[col].fillna(ZERO_VALUE)
        
    
    # ======================================
    # Basement Features
    # ======================================

    for col in BASEMENT_CATEGORICAL:
        df[col] = df[col].fillna(NONE_CATEGORY)

    for col in BASEMENT_NUMERICAL:
        df[col] = df[col].fillna(ZERO_VALUE)
    
    
    # ======================================
    # Masonry Features
    # ======================================

    for col in MASONRY_CATEGORICAL:
        df[col] = df[col].fillna(NONE_CATEGORY)

    for col in MASONRY_NUMERICAL:
        df[col] = df[col].fillna(ZERO_VALUE)
        
    
    # ======================================
    # Fireplace
    # ======================================

    df[FIREPLACE_FEATURE] = df[FIREPLACE_FEATURE].fillna(NONE_CATEGORY)
    
    
    # ======================================
    # Pool
    # ======================================

    df[POOL_FEATURE] = df[POOL_FEATURE].fillna(NONE_CATEGORY)
    
    
    # ======================================
    # Lot Frontage
    # ======================================

    df[LOT_FRONTAGE] = (
        df.groupby(NEIGHBORHOOD)[LOT_FRONTAGE]
          .transform(lambda x: x.fillna(x.median()))
    )
    
    
    overall_median = df[LOT_FRONTAGE].median()

    df.loc[
        df[NEIGHBORHOOD].isin(SPECIAL_NEIGHBORHOODS),
        LOT_FRONTAGE
    ] = df.loc[
        df[NEIGHBORHOOD].isin(SPECIAL_NEIGHBORHOODS),
        LOT_FRONTAGE
    ].fillna(overall_median)
    
    
    # ======================================
    # Electrical
    # ======================================

    df[ELECTRICAL] = df[ELECTRICAL].fillna(
        df[ELECTRICAL].mode()[0]
    )
    
    
    # ======================================
    # Fence
    # ======================================

    df[FENCE_FEATURE] = df[FENCE_FEATURE].fillna(NONE_CATEGORY)
    
    
    return df


## here we are removing those 3 outliers only that we decided during EDA. We are not removing any other outliers because they represent luxury and legitimate houses.
def remove_outliers(df):
    """
    Remove known outliers from the training data.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    return df[
        ~(
            (df["Gr Liv Area"] > 4000) &
            (df["SalePrice"] < 300000)
        )
    ].copy()
    


def preprocess_data(df, training=True):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame

    training : bool
        Whether preprocessing is being
        performed for model training.

    Returns
    -------
    pd.DataFrame
    """

    df = fill_missing_values(df)

    if training:
        df = remove_outliers(df)

    return df


