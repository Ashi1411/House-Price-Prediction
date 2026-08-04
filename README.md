# 🏠 House Price Prediction Web Application using Flask & XGBoost

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple?logo=bootstrap)
![License](https://img.shields.io/badge/License-MIT-orange)

A production-ready **House Price Prediction Web Application** built using **Flask** and **XGBoost Regression**. The application estimates the market value of residential properties based on user-provided house characteristics using the **Ames Housing Dataset**.

The project follows a complete end-to-end Machine Learning workflow, including data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive web interface.

---

## 📌 Overview

Predicting house prices is a classic regression problem where multiple factors such as location, house size, construction quality, and amenities influence the final selling price.

This project demonstrates how a Machine Learning model can be trained on historical housing data and deployed as a real-time web application that allows users to estimate house prices by simply filling out a form.

The application automatically handles missing features using default values derived from the training dataset, performs all preprocessing and feature engineering steps internally, and generates predictions using a trained **XGBoost Regression** model.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Data Preprocessing](#-data-preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Model Training](#-model-training)
- [Prediction Pipeline](#-prediction-pipeline)
- [Software Architecture](#-software-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Application Screenshots](#-application-screenshots)
- [Model Performance](#-model-performance)
- [Getting Started](#-getting-started)
- [Using the Application](#-using-the-application)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Author](#-author)
---


# 📊 Dataset

This project uses the **Ames Housing Dataset**, a widely used benchmark dataset for regression problems in Machine Learning.

### Dataset Characteristics

- 📍 2,930 residential properties
- 🏠 80+ explanatory features
- 💰 Target Variable: `SalePrice`
- 📈 Regression Problem

The dataset contains a diverse range of property characteristics, including:

- Location
- Lot dimensions
- Construction quality
- House style
- Basement details
- Garage information
- Interior features
- Exterior features
- Sale information

The objective is to predict the final selling price (`SalePrice`) of a house using these features.

---

# 🧠 Machine Learning Pipeline

The project follows a complete end-to-end Machine Learning workflow.

```text
                Raw Dataset
                     │
                     ▼
             Data Preprocessing
                     │
                     ▼
         Missing Value Handling
                     │
                     ▼
          Feature Engineering
                     │
                     ▼
         Categorical Encoding
                     │
                     ▼
          Train-Test Split
                     │
                     ▼
      XGBoost Regression Model
                     │
                     ▼
          Model Evaluation
                     │
                     ▼
      Model Serialization (.pkl)
                     │
                     ▼
        Flask Prediction Pipeline
                     │
                     ▼
          House Price Prediction
```
---

# ⚙️ Data Preprocessing

Before training the model, the dataset undergoes several preprocessing steps:

- Handling missing values
- Removing unnecessary features
- Standardizing categorical values
- Correcting inconsistent entries
- Preparing data for feature engineering

The same preprocessing pipeline is automatically applied during prediction, ensuring consistency between training and inference.

---

# 🛠️ Feature Engineering

Several additional features are engineered to improve model performance.

Examples include:

- House Age
- Remodeled House Age
- Total Bathrooms
- Total Porch Area
- Total Living Area
- Total Outdoor Area

Feature engineering enables the model to capture more meaningful relationships within the data.

---

# 🤖 Model Training

The prediction model is built using **XGBoost Regression**, an efficient gradient boosting algorithm known for its strong performance on structured tabular datasets.

Training workflow:

- Data preprocessing
- Feature engineering
- Categorical encoding
- Train-Test Split
- Model fitting
- Performance evaluation
- Model serialization using Joblib

The trained model is stored as:

models/
    house_price_pipeline.pkl

---

# ❓ Why XGBoost?

XGBoost was selected because it offers:

- High accuracy on structured/tabular datasets
- Built-in regularization to reduce overfitting
- Efficient handling of missing values
- Excellent performance compared to many traditional regression algorithms
- Fast training and prediction

These characteristics make XGBoost one of the most widely used algorithms for tabular Machine Learning problems.
---

# 🔄 Prediction Pipeline

During inference, user input passes through the exact same pipeline used during training.

```text
User Input
     │
     ▼
Raw Default Values
     │
     ▼
Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Encoding
     │
     ▼
Feature Alignment
     │
     ▼
XGBoost Model
     │
     ▼
Predicted Price
```

This guarantees that training and prediction use identical feature transformations.

---

# 🏗️ Software Architecture

```text
                User

                  │

                  ▼

            Flask Frontend

                  │

                  ▼

              Flask Routes

                  │

                  ▼

          Prediction Pipeline

       ┌──────────┼──────────┐

       ▼          ▼          ▼

Preprocessing Feature Eng. Encoding

       └──────────┼──────────┘

                  ▼

         XGBoost Regression

                  ▼

          Predicted Price
```

---

# 📂 Project Structure

```text
House-Price-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── house_price_pipeline.pkl
│   ├── feature_names.pkl
│   └── raw_defaults.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── encoding.py
│   ├── pipeline.py
│   ├── train.py
│   └── form_options.py
│
├── static/
│   ├── css/
│   ├── images/
│   └── js/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── macros.html
│
└── screenshots/
```

---



## 🚀 Key Features

- 🏡 Predict house prices in real time through a web interface
- 🤖 Powered by an optimized XGBoost Regression model
- 📊 Trained on the Ames Housing Dataset (2900+ houses)
- ⚙️ Automatic preprocessing and feature engineering
- 🔄 Handles missing values intelligently using training defaults
- 🎯 Dynamic form generation with dropdown options
- 📱 Responsive user interface built with Bootstrap
- 💾 Serialized prediction pipeline using Joblib
- 🧩 Modular and scalable project architecture
- 🌐 Ready for cloud deployment using Flask

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Backend | Flask |
| Frontend | HTML, CSS, Bootstrap, JavaScript |
| Model Serialization | Joblib |


---

# 📸 Application Screenshots

## 🏠 Home Page

> *(Add a screenshot after deployment)*

![Home Page](screenshots/home.png)

---

## 📝 Prediction Form

> *(Add a screenshot after deployment)*

![Prediction Form](screenshots/prediction-form.png)

---

## 💰 Prediction Result

> *(Add a screenshot after deployment)*

![Prediction Result](screenshots/prediction-result.png)

---



# 📈 Model Performance

The XGBoost Regression model was evaluated on the test dataset using standard regression metrics.

| Metric | Score |
|---------|-------|
| Mean Absolute Error (MAE) | **12,771.76** |
| Root Mean Squared Error (RMSE) | **20,140.86** |
| R² Score | **0.9497** |

### Interpretation

- **R² Score of 0.9497** indicates that the model explains approximately **95% of the variance** in house prices.
- A relatively low **MAE** demonstrates that the predicted prices are generally close to the actual selling prices.
- The model provides accurate and reliable predictions for residential properties in the Ames Housing Dataset.

---

# 🚀 Getting Started

Follow these steps to run the project locally.

## 1. Clone the Repository

```bash
git clone https://github.com/<your-github-username>/House-Price-Prediction.git

cd House-Price-Prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Train the Model

```bash
python -m src.train
```

This command will:

- Preprocess the dataset
- Perform feature engineering
- Encode categorical variables
- Train the XGBoost Regression model
- Save the trained pipeline
- Save feature names
- Save default feature values

---

## 5. Launch the Web Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

# 💻 Using the Application

1. Open the Home Page.
2. Click **Predict House Price**.
3. Fill in the required house details.
4. Submit the prediction form.
5. View the estimated market value generated by the Machine Learning model.

The application automatically performs all preprocessing and feature engineering steps before making predictions.

---

# 🌐 Deployment

The application is designed for deployment on cloud platforms such as:

- Render
- Railway
- PythonAnywhere

> **Live Demo:** *(Add deployment URL here after deployment.)*

---

# 🔮 Future Improvements

The project can be further enhanced with the following features:

- 📈 SHAP-based feature importance visualization
- 📊 Prediction confidence estimation
- 🐳 Docker containerization
- ☁️ Cloud deployment using AWS or Azure
- 🔐 User authentication
- 📁 Store prediction history using PostgreSQL
- 📡 REST API using Flask Blueprint architecture
- 🤖 Model retraining pipeline
- 📱 Progressive Web App (PWA) support

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you'd like to contribute:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

# 👨‍💻 Author

**Ashi Jain**

B.Tech Electrical Engineering  
Netaji Subhas University of Technology (NSUT)

GitHub: **[@Ashi1411](https://github.com/Ashi1411)**

LinkedIn: **[Ashi Jain](https://www.linkedin.com/in/ashi-jain-787798282/)**

---

# ⭐ If you found this project helpful...

If you found this project useful or interesting, consider giving it a ⭐ on GitHub. It helps others discover the project and motivates future improvements.

---