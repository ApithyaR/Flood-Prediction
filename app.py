import pandas as pd
from flask import Flask, request, render_template
import pickle
import os
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        # Load the trained CatBoost model
        model_path = 'models/cb_submission.pkl'  
        try:
            with open(model_path, 'rb') as file:
                cb_model = pickle.load(file)
            print(f"Model type: {type(cb_model)}")
            if not isinstance(cb_model, CatBoostRegressor):
                raise TypeError("Loaded object is not of type CatBoostRegressor")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            cb_model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            cb_model = None

        if cb_model is None:
            return render_template("index.html", prediction_text="Model is not loaded.")
        
        # Extract feature values from the form
        features = [
            float(request.form['MonsoonIntensity']),
            float(request.form['TopographyDrainage']),
            float(request.form['RiverManagement']),
            float(request.form['Deforestation']),
            float(request.form['Urbanization']),
            float(request.form['ClimateChange']),
            float(request.form['DamsQuality']),
            float(request.form['Siltation']),
            float(request.form['AgriculturalPractices']),
            float(request.form['Encroachm']),
            float(request.form['IneffectiveDisasterPreparedness']),
            float(request.form['DrainageSystems']),
            float(request.form['CoastalVulnerability']),
            float(request.form['Landslides']),
            float(request.form['Watersheds']),
            float(request.form['DeterioratingInfrastructure']),
            float(request.form['PopulationScore']),
            float(request.form['WetlandLoss']),
            float(request.form['InadequatePlanning']),
            float(request.form['PoliticalFactors'])
        ]
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features], columns=[
            'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
            'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
            'AgriculturalPractices', 'Encroachm', 'IneffectiveDisasterPreparedness', 
            'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
            'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss', 
            'InadequatePlanning', 'PoliticalFactors'
        ])
        
        # Predict using the loaded model
        try:
            prediction = cb_model.predict(feature_df)
            print(prediction)
            return render_template("index.html", prediction_text=prediction[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template("index.html", prediction_text=f"Error during prediction: {str(e)}")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
