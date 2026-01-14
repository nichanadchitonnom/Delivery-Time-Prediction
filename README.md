# Delivery Time Prediction 
## About The Project
The primary objective of this project is to develop a predictive model that can forecast the food delivery time. 
This model aims to predict food delivery time to support better operational decisions in food delivery platforms.


## Problem Statement

Food delivery platforms face significant challenges in accurately estimating delivery time due to various dynamic factors.
Inaccurate delivery time predictions can lead to customer dissatisfaction, reduced trust in the platform, and inefficient operational planning.

The objective of this project is to build an end-to-end machine learning system that predicts food delivery time using historical delivery data. 
this system aims to provide more reliable delivery time estimates and support better decision-making for food delivery operations.

**Models Trained**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor 
* LightGBM Regressor

## Attributes
- Order_ID
- Distance_km
- Weather: Clear, Rainy, Snowy, Foggy, and Windy.
- Traffic_Level: Low, Medium, or High.
- Time_of_Day: Morning, Afternoon, Evening, or Night.
- Vehicle_Type: Bike, Scooter, and Car.
- Preparation_Time_min
- Courier_Experience_yrs
- Delivery_Time_min (target variable).

## Built With
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- shap
- Flask


## How to Run the Project

### 1️. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️. Train Model

```bash
python -m src.pipeline.Training_pipeline
```

### 4️. Run Web App

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```
## Key Takeaways
- Developed a complete end-to-end machine learning pipeline, from data ingestion and preprocessing to model training and deployment.
- Learned how feature engineering (e.g., Peak_Score and Is_Peak_Hour) can significantly impact model performance.
- Gained experience in model selection and hyperparameter tuning, understanding why experimental results may differ from production pipelines.
- Deployed a trained model using Flask, enabling real-time delivery time prediction via a web interface.

