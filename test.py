import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from preprocess import preprocess_data, load_data
from meta_learner import train_meta_learner
import config

class TestPipelineComponents(unittest.TestCase):

    # Test configuration loading
    def test_config_meta_learner(self):
        self.assertIn("hidden_layers", config.META_LEARNER_CONFIG)
        self.assertIn("epochs", config.META_LEARNER_TRAINING)
        self.assertTrue(isinstance(config.META_LEARNER_CONFIG["hidden_layers"], list))
        self.assertTrue(isinstance(config.META_LEARNER_TRAINING["epochs"], int))

    # Test preprocessing
    def test_preprocessing(self):
        # Simulated raw data
        raw_data = pd.DataFrame({
            "feature1": [1, 2, None],
            "feature2": ["cat", "dog", None],
            "contract_start_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "contract_end_date": ["2023-01-10", "2023-01-12", "2023-01-15"]
        })
        
        processed_data = preprocess_data(raw_data)
        self.assertIn("contract_duration", processed_data.columns)
        self.assertFalse(processed_data.isnull().any().any())  # No missing values
        self.assertTrue(processed_data["feature2"].dtype in [np.float64, np.int64])  # Encoded column

    # Test data loading
    def test_load_data(self):
        data_path = "data/snp_dld_2024_rents.csv"
        try:
            df = load_data(data_path)
            self.assertIsInstance(df, pd.DataFrame)
        except FileNotFoundError:
            self.fail(f"Data file not found at {data_path}")

    # Test train_meta_learner
    def test_meta_learner_training(self):
        # Simulated base model predictions and target values
        base_model_predictions = np.random.rand(100, 3)  # 100 samples, 3 models
        y = np.random.rand(100)

        # Train the meta-learner
        model = train_meta_learner(base_model_predictions, y)

        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), len(config.META_LEARNER_CONFIG["hidden_layers"]) + 2)  # Hidden + Output

    # Test data split
    def test_data_split(self):
        X = np.random.rand(100, 10)  # 100 samples, 10 features
        y = np.random.rand(100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.META_LEARNER_TRAINING["test_size"], 
            random_state=config.META_LEARNER_TRAINING["random_state"]
        )

        self.assertEqual(len(X_train), int((1 - config.META_LEARNER_TRAINING["test_size"]) * len(X)))
        self.assertEqual(len(X_test), int(config.META_LEARNER_TRAINING["test_size"] * len(X)))

if __name__ == "__main__":
    unittest.main()
