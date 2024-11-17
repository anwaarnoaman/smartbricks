# train.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from joblib import dump
from preprocess import *
from feature_selection import *
from base_models import *
from meta_learner import *
from evaluation import *
import config
 


def load_data(data_path):
    return pd.read_csv(data_path)

def save_data(data, output_path):
    
    data.to_csv(output_path, index=False)

def train_and_save_models():
    # Load and preprocess the data
    raw_data = load_data(config.DATA_PATH).head(10000) 
    processed_data = preprocess_data(raw_data)
    save_data(processed_data, config.PROCESSED_DATA_PATH)
   
    # Feature Selection
    rent_features = feature_selection(processed_data)
    print(rent_features)

    base_models = train_base_models(processed_data[rent_features], processed_data[config.TARGET_COLUMN])
    print("Base model trained")
    for name, model in base_models.items():
        path = config.MODEL_SAVE_PATHS[name.lower().replace(" ", "_")]
        joblib.dump(model, path)
        print(f"Saved {name} model to {path}")

    print(base_models)

    meta_learner = train_meta_learner(np.array([model.predict(processed_data[rent_features]) for model in base_models.values()]).T, processed_data[config.TARGET_COLUMN])

    # Prepare base model predictions for the meta-learner
    base_model_predictions = np.array([
        model.predict(processed_data[rent_features]) for model in base_models.values()
    ]).T
    print("Base predictions shape:", base_model_predictions.shape)  # Debugging line
    # Make predictions using the meta-learner
    meta_predictions = meta_learner.predict(base_model_predictions)

    print("Rent Model Evaluation:", evaluate_model(processed_data[config.TARGET_COLUMN], meta_predictions))
 

    #Save meta-learner 
    joblib.dump(meta_learner,config.META_MODEL_PATH)


    print("Training completed and models saved.")

if __name__ == "__main__":
    train_and_save_models()
