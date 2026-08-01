"""
Configuration file for the House Price Prediction project.

This file contains all project-wide constants such as:
- Dataset paths
- Model paths
- Random seed
- Feature lists
- Encoding mappings
"""

# ==========================================================
# Dataset Configuration
# ==========================================================

from pathlib import Path

# ==========================
# Base Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ==========================
# Dataset
# ==========================

DATA_PATH = DATA_DIR / "AmesHousing.csv"
TARGET_COLUMN = "SalePrice"
RANDOM_STATE = 42


# ==========================================================
# Model Configuration
# ==========================================================

MODEL_FILENAME = "house_price_pipeline.pkl"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME


# ==========================================================
# Saved Artifacts
# ==========================================================

FEATURE_NAMES_FILENAME = "feature_names.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / FEATURE_NAMES_FILENAME 

RAW_DEFAULTS_FILENAME = "raw_defaults.pkl"
RAW_DEFAULTS_PATH = MODEL_DIR / RAW_DEFAULTS_FILENAME


# ==========================================================
# Missing Value Handling
# ==========================================================

GARAGE_CATEGORICAL = [
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond"
]

GARAGE_NUMERICAL = [
    "Garage Yr Blt",
    "Garage Cars",
    "Garage Area"
]

BASEMENT_CATEGORICAL = [
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2"
]

BASEMENT_NUMERICAL = [
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath"
]

MASONRY_CATEGORICAL = [
    "Mas Vnr Type"
]

MASONRY_NUMERICAL = [
    "Mas Vnr Area"
]

FIREPLACE_FEATURE = "Fireplace Qu"

POOL_FEATURE = "Pool QC"

LOT_FRONTAGE = "Lot Frontage"

ELECTRICAL = "Electrical"

NEIGHBORHOOD = "Neighborhood"

FENCE_FEATURE = "Fence"


# ==========================================================
# Ordinal Encoding Mappings
# ==========================================================

QUALITY_MAPPING = {
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5
}

FINISH_MAPPING = {
    "None": 0,
    "Unf": 1,
    "RFn": 2,
    "Fin": 3
}

EXPOSURE_MAPPING = {
    "None": 0,
    "No": 1,
    "Mn": 2,
    "Av": 3,
    "Gd": 4
}

BASEMENT_FINISH_MAPPING = {
    "None": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6
}



PAVED_DRIVE_MAPPING = {
    "N": 0,
    "P": 1,
    "Y": 2
}

LOT_SHAPE_MAPPING = {
    "IR3": 0,
    "IR2": 1,
    "IR1": 2,
    "Reg": 3
}

UTILITIES_MAPPING = {
    "ELO": 0,
    "NoSeWa": 1,
    "NoSewr": 2,
    "AllPub": 3
}

LAND_SLOPE_MAPPING = {
    "Sev": 0,
    "Mod": 1,
    "Gtl": 2
}


ORDINAL_MAPPINGS = {

    # Exterior
    "Exter Qual": QUALITY_MAPPING,
    "Exter Cond": QUALITY_MAPPING,

    # Basement
    "Bsmt Qual": QUALITY_MAPPING,
    "Bsmt Cond": QUALITY_MAPPING,
    "Bsmt Exposure": EXPOSURE_MAPPING,
    "BsmtFin Type 1": BASEMENT_FINISH_MAPPING,
    "BsmtFin Type 2": BASEMENT_FINISH_MAPPING,

    # Heating
    "Heating QC": QUALITY_MAPPING,

    # Kitchen
    "Kitchen Qual": QUALITY_MAPPING,

    # Fireplace
    "Fireplace Qu": QUALITY_MAPPING,

    # Garage
    "Garage Finish": FINISH_MAPPING,
    "Garage Qual": QUALITY_MAPPING,
    "Garage Cond": QUALITY_MAPPING,

    # Pool
    "Pool QC": QUALITY_MAPPING,

    # Driveway
    "Paved Drive": PAVED_DRIVE_MAPPING,
    
    "Lot Shape": LOT_SHAPE_MAPPING,
    "Utilities": UTILITIES_MAPPING,
    "Land Slope": LAND_SLOPE_MAPPING,
}


# ==========================================================
# Feature Lists
# ==========================================================

ORDINAL_FEATURES = [
    "Exter Qual",
    "Exter Cond",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Heating QC",
    "Kitchen Qual",
    "Fireplace Qu",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Pool QC",
    "Paved Drive",
    "Lot Shape",
    "Utilities",
    "Land Slope",
]


NOMINAL_FEATURES = [
    "MS SubClass",
    "MS Zoning",
    "Street",
    "Alley",

    "Land Contour",
    "Lot Config",

    "Neighborhood",

    "Condition 1",
    "Condition 2",

    "Bldg Type",
    "House Style",

    "Roof Style",
    "Roof Matl",

    "Exterior 1st",
    "Exterior 2nd",

    "Mas Vnr Type",

    "Foundation",

    "Heating",

    "Central Air",

    "Electrical",

    "Functional",

    "Garage Type",

    "Fence",

    "Misc Feature",

    "Sale Type",
    "Sale Condition"
]


# ==========================================================
# Feature Engineered Features
# ==========================================================

ENGINEERED_FEATURES = [
    "House Age",
    "Remodel Age",
    "Total Bathrooms",
    "Total Living Area",
    "Total Porch Area",
    "Total House Area"
]


# ==========================================================
# Feature Names Used in Feature Engineering
# ==========================================================

YEAR_BUILT = "Year Built"
YEAR_REMODELED = "Year Remod/Add"
YEAR_SOLD = "Yr Sold"

GR_LIV_AREA = "Gr Liv Area"
TOTAL_BSMT_SF = "Total Bsmt SF"
GARAGE_AREA = "Garage Area"

FULL_BATH = "Full Bath"
HALF_BATH = "Half Bath"
BSMT_FULL_BATH = "Bsmt Full Bath"
BSMT_HALF_BATH = "Bsmt Half Bath"

OPEN_PORCH = "Open Porch SF"
ENCLOSED_PORCH = "Enclosed Porch"
THREE_SEASON_PORCH = "3Ssn Porch"
SCREEN_PORCH = "Screen Porch"
WOOD_DECK = "Wood Deck SF"


# Special Neighbours
SPECIAL_NEIGHBORHOODS = [
    "GrnHill",
    "Landmrk"
]


# Missing value constants
NONE_CATEGORY = "None"

ZERO_VALUE = 0


MISSING_VALUE_CONFIG = {
    "garage": {
        "categorical": GARAGE_CATEGORICAL,
        "numerical": GARAGE_NUMERICAL
    },

    "basement": {
        "categorical": BASEMENT_CATEGORICAL,
        "numerical": BASEMENT_NUMERICAL
    },

    "masonry": {
        "categorical": MASONRY_CATEGORICAL,
        "numerical": MASONRY_NUMERICAL
    }
}


# ==========================================================
# XGBoost Configuration
# ==========================================================

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE
}






