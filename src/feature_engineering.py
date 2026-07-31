"""
Feature engineering module.

This module contains functions for creating
new features from the existing dataset.
"""

from src.config import (
    YEAR_BUILT,
    YEAR_REMODELED,
    YEAR_SOLD,
    GR_LIV_AREA,
    TOTAL_BSMT_SF,
    GARAGE_AREA,
    FULL_BATH,
    HALF_BATH,
    BSMT_FULL_BATH,
    BSMT_HALF_BATH,
    OPEN_PORCH,
    ENCLOSED_PORCH,
    THREE_SEASON_PORCH,
    SCREEN_PORCH,
    WOOD_DECK
)


def create_house_age(df):
    """
    Create House Age feature.
    """

    df = df.copy()

    df["House Age"] = (
        df[YEAR_SOLD] - df[YEAR_BUILT]
    )

    return df



def create_remodel_age(df):
    """
    Create Remodel Age feature.
    """

    df = df.copy()

    df["Remodel Age"] = (
        df[YEAR_SOLD] - df[YEAR_REMODELED]
    )

    return df



def create_total_bathrooms(df):
    """
    Create Total Bathrooms feature.
    """

    df = df.copy()

    df["Total Bathrooms"] = (
        df[FULL_BATH]
        + 0.5 * df[HALF_BATH]
        + df[BSMT_FULL_BATH]
        + 0.5 * df[BSMT_HALF_BATH]
    )

    return df



def create_total_living_area(df):
    """
    Create Total Living Area feature.
    """

    df = df.copy()

    df["Total Living Area"] = (
        df[GR_LIV_AREA]
        + df[TOTAL_BSMT_SF]
    )

    return df



def create_total_porch_area(df):
    """
    Create Total Porch Area feature.
    """

    df = df.copy()

    df["Total Porch Area"] = (
        df[OPEN_PORCH]
        + df[ENCLOSED_PORCH]
        + df[THREE_SEASON_PORCH]
        + df[SCREEN_PORCH]
        + df[WOOD_DECK]
    )

    return df


def create_total_house_area(df):
    """
    Create Total House Area feature.
    """

    df = df.copy()

    df["Total House Area"] = (
        df["Total Living Area"]
        + df[GARAGE_AREA]
    )

    return df



def engineer_features(df):
    """
    Apply all feature engineering steps.
    """

    df = create_house_age(df)
    df = create_remodel_age(df)
    df = create_total_bathrooms(df)
    df = create_total_living_area(df)
    df = create_total_porch_area(df)
    df = create_total_house_area(df)

    return df



