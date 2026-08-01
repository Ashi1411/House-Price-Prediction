"""
Generate dropdown options for the prediction form.
"""

from src.preprocessing import load_data


def get_form_options():
    """
    Returns all categorical dropdown values.

    Returns
    -------
    dict
    """

    df = load_data()

    categorical_columns = [

        "Neighborhood",
        "MS Zoning",
        "Street",
        "Lot Shape",
        "Utilities",
        "Land Slope",
        "Lot Config",

        "House Style",
        "Bldg Type",

        "Roof Style",
        "Roof Matl",

        "Exterior 1st",
        "Exterior 2nd",

        "Foundation",

        "Heating",
        "Heating QC",

        "Central Air",

        "Electrical",

        "Functional",

        "Garage Type",
        "Garage Finish",
        "Garage Qual",
        "Garage Cond",

        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",

        "Kitchen Qual",

        "Pool QC",

        "Fence",

        "Fireplace Qu",

        "Sale Type",
        "Sale Condition",

        "Misc Feature",

        "Exter Qual",
        "Exter Cond"

    ]

    options = {}

    for column in categorical_columns:

        values = sorted(
            df[column]
            .fillna("None")
            .astype(str)
            .str.strip()
            .unique()
        )

        options[column] = values

    return options