# config.py

# File paths
DATA_PATH = "data/snp_dld_2024_rents.csv"
PROCESSED_DATA_PATH = "data/processed_rents.csv"
NEW_DATA_PATH = "../data/new_rents.csv"

# Model search spaces for BayesSearchCV
HYPERPARAMETER_SEARCH_SPACES = {
    "XGBoost": {
        "max_depth": (3, 10),
        "n_estimators": (50, 300),
    },
    "Random Forest": {
        "max_depth": (3, 10),
        "n_estimators": (50, 300),
    },
    "SVR": {
        "C": (0.1, 10.0),
        "gamma": (0.01, 1.0),
    },
}

# General settings for BayesSearchCV
BAYES_SEARCH_SETTINGS = {
    "n_iter": 20,  # Number of iterations for the search
    "cv": 3,       # Number of cross-validation folds
}

# Model save paths
MODEL_SAVE_PATHS = {
    "xgboost": "models/XGBoost.pkl",
    "random_forest": "models/Random_Forest.pkl",
    "svr": "models/SVR.pkl",
}



# General settings
RANDOM_STATE = 42
TARGET_COLUMN = "annual_amount"



# Meta-learner configuration
META_LEARNER_CONFIG = {
    "input_dim": None,         # Placeholder, dynamically set during training
    "hidden_layers": [64, 32], # Number of neurons in hidden layers
    "dropout_rate": 0.2,       # Dropout rate
    "output_dim": 1,           # Output layer neurons
    "activation": {
        "hidden": "relu",      # Activation for hidden layers
        "output": "linear"     # Activation for output layer
    }
}

# Training settings
META_LEARNER_TRAINING = {
    "epochs": 50,
    "batch_size": 32,
    "test_size": 0.2,
    "random_state": 42
}
 
META_MODEL_PATH=   "models/meta_model.pkl"