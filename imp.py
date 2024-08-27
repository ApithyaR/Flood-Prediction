import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from sklearn.model_selection import train_test_split
import joblib
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor

if not os.path.exists('models'):
    os.makedirs('models')

# Load dataset
#df = pd.read_csv(r"C:\Apithya\ICBT TOP-UP\TOPUP- 2nd Semester\CI\Prediction_submission5 (1).csv")  
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_id = test[["id"]]
train = train.set_index("id")
test = test.set_index("id")
X = train.drop(columns=['FloodProbability'])
y = train['FloodProbability']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
cb_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1,verbose=False) 
cb_model.fit(X_train_scaled, y_train)


# Save the catboost Regression model
with open('models/cb_submission.pkl', 'wb') as file:
    pickle.dump(cb_model, file)
    #joblib.dump(cb_model, file)

print("Model trained and saved successfully.")





