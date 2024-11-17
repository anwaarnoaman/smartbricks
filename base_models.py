import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
import config

def train_base_models(X, y):
    models = {
        "XGBoost": xgb.XGBRegressor(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR()
    }
    
    tuned_models = {}
    search_space = config.HYPERPARAMETER_SEARCH_SPACES
    for name, model in models.items():
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space[name],
            n_iter=config.BAYES_SEARCH_SETTINGS["n_iter"],
            cv=config.BAYES_SEARCH_SETTINGS["cv"] 
        )
        bayes_search.fit(X, y)
        tuned_models[name] = bayes_search.best_estimator_
        print(f"Best parameters for {name}: {bayes_search.best_params_}")
        print(f"Best score for {name}: {bayes_search.best_score_}")
    

    return tuned_models
