# Flood-Prediction
Regression with a Flood Prediction Dataset-Kaggle

# Overview
The Flood Prediction Application is a web-based platform designed to predict the probability of flooding based on environmental data using the CatBoost machine learning 
model. The application is built with Flask and offers users the ability to input new data and receive flood predictions.

# Features
*Data Preprocessing: Cleans and prepares data for accurate predictions.
*Flood Prediction: Utilizes the CatBoost Regressor model to predict flood probabilities.
*Data Visualization: Provides visual representations of data distribution and model predictions.

# Installation
Python 3.7 or higher
Virtual environment (recommended)

# STEPS
    git clone https://github.com/ApithyaR/flood-prediction.git
    cd flood-prediction-app
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -m flask run

Open your browser and navigate to http://localhost:5000/.

Model and Techniques
    CatBoost Regressor: The primary model used for predicting flood probabilities. CatBoost is known for its performance with categorical features and is well-suited for this task.

Data
The dataset used for this application is sourced from the Kaggle competition "Playground Series - Season 4, Episode 5", which includes various environmental features to 
predict flood probabilities.

Contributing
If you'd like to contribute, please fork the repository and submit a pull request with your improvements.

License
This project is licensed under the MIT License.

Contact
For any questions or issues, feel free to contact Apithya at abithya2018@gmail.com

