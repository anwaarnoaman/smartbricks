

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression
import config

def feature_selection(data):
    print("Feature selection started")

    # Split into features and target
    X = data.drop(columns=[config.TARGET_COLUMN])
    y = data[config.TARGET_COLUMN]
    
    # Correlation Analysis
    corr_matrix = data.corr()
    correlated_features = corr_matrix[config.TARGET_COLUMN][abs(corr_matrix[config.TARGET_COLUMN]) > 0.7].index.tolist()
    print("Correlation Analysis")
    # Univariate feature selection
    univariate_selector = SelectKBest(score_func=mutual_info_regression, k=10)
    univariate_selector.fit(X, y)
    print("Univariate feature selection")
    # Feature importance with tree-based models
    model = RandomForestRegressor()
    model.fit(X, y)
    print("Feature importance with tree-based models")
    # Recursive Feature Elimination (RFE)
    rfe_selector = RFE(estimator=model, n_features_to_select=10)
    rfe_selector.fit(X, y)
    print(" Recursive Feature Elimination (RFE)")
    # Combine selected features
    selected_features = set(correlated_features + list(X.columns[univariate_selector.get_support()]) + list(X.columns[rfe_selector.get_support()]))
    return list(selected_features)
