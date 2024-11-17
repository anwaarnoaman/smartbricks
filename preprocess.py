# preprocessor.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def handle_missing_data(df):
    # Fill categorical missing values with "Unknown"
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # Fill numerical missing values with median
    numerical_cols = df.select_dtypes(include=["number"]).columns
    imputer = SimpleImputer(strategy="median")
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    return df

def feature_engineering(df_rents):
    # Add contract duration in days
    df_rents["contract_duration"] = (
        pd.to_datetime(df_rents["contract_end_date"]) - pd.to_datetime(df_rents["contract_start_date"])
    ).dt.days
    return df_rents

def encode_categorical_with_frequency(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        freq_map = df[col].value_counts().to_dict()
        df[col] = df[col].map(freq_map)
    return df

def scale_numerical(df):
    numerical_cols = df.select_dtypes(include=["number"]).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def preprocess_data(data):
    filter=[
            'contract_start_date',
            'contract_end_date', 'version_text',
            'contract_amount', 'annual_amount', 'is_freehold', 
            'property_size_sqm', 
            'property_type_en',  'property_subtype_en',
            'property_usage_en', 
            'total_properties', 'rooms', 'parking',
            'project_name_en',   'area_en',  
            'nearest_landmark_en',   'nearest_metro_en',
            'nearest_mall_en', 
            'master_project_en', 
            ]
    data=data[
             filter 
            ]
    # Handle missing data
    data = handle_missing_data(data)
    # Perform feature engineering
    data = feature_engineering(data)
    # Encode categorical variables
    data = encode_categorical_with_frequency(data)
    # Scale numerical features
    data = scale_numerical(data)
    return data

def load_data(data_path):
    return pd.read_csv(data_path)

def save_data(data, output_path):
    
    data.to_csv(output_path, index=False)
