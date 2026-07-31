"""
Encoding module.

This module contains functions for encoding
categorical features.
"""

import pandas as pd

from src.config import (
    ORDINAL_MAPPINGS,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES
)



def ordinal_encode(df):
    """
    Apply ordinal encoding to ordinal features.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    for feature in ORDINAL_FEATURES:
        
        if feature not in ORDINAL_MAPPINGS:
            raise ValueError(
                f"No ordinal mapping defined for '{feature}'."
            )
            
        
        df[feature] = df[feature].map(
            ORDINAL_MAPPINGS[feature]
        )

    return df



def one_hot_encode(df):
    """
    Apply one-hot encoding to nominal features.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    df = pd.get_dummies(
        df,
        columns=NOMINAL_FEATURES,
        drop_first=False,
        dtype=int
    )

    return df



def encode_features(df):
    """
    Apply all encoding steps.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = ordinal_encode(df)

    df = one_hot_encode(df)

    return df

