"""
Prediction module.

This module provides a simple interface for
predicting house prices using the trained model.
"""

import pandas as pd

from src.pipeline import predict_price


def predict(input_data, model, feature_names):
    """
    Predict the price of a house.

    Parameters
    ----------
    input_data : dict
        Dictionary containing the house features.

    Returns
    -------
    float
        Predicted house price.
    """

    df = pd.DataFrame([input_data])

    prediction = predict_price(
        df,
        model,
        feature_names
    )
    
    return float(prediction[0])


## ! for testing
if __name__ == "__main__":

    sample_house = {
        # We'll fill this later
    }

    print(predict(sample_house))