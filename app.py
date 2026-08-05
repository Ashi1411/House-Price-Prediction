"""
Flask application for House Price Prediction.
"""

import os

from flask import (
    Flask,
    render_template,
    request
)

import traceback

from src.pipeline import (
    load_model,
    load_feature_names,
    predict_price
)

from src.form_options import get_form_options



try:
    model = load_model()
    feature_names = load_feature_names()
    form_options = get_form_options()
except Exception as e:
    print(f"Startup Error: {e}")
    raise


@app.route("/")
def home():
    """
    Render the home page.
    """
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():

    try:

        user_input = request.form.to_dict()

        prediction = predict_price(
            user_input,
            model,
            feature_names
        )

        predicted_price = round(
            float(prediction[0]),
            2
        )

        return render_template(
            "predict.html",
            prediction=f"{predicted_price:,.2f}",
            options=form_options,
            form_data=user_input
        )

    except Exception as e:

        traceback.print_exc()

        return render_template(
            "predict.html",
            options=form_options,
            form_data=user_input,
            error=str(e)
        )
        



@app.route("/predict", methods=["GET"])
def predict_page():

    return render_template(
        "predict.html",
        options=form_options,
        form_data={}
    )
    


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )