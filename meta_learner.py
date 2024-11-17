from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import config


def train_meta_learner(base_model_predictions, y):
    # X_train, X_test, y_train, y_test = train_test_split(base_model_predictions, y, test_size=0.2, random_state=42)

    # model = Sequential([
    #     Dense(64, activation="relu", input_dim=X_train.shape[1]),
    #     Dropout(0.2),
    #     Dense(32, activation="relu"),
    #     Dense(1, activation="linear")
    # ])
    
    # model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    # model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    # return model


 # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        base_model_predictions, y, 
        test_size=config.META_LEARNER_TRAINING["test_size"], 
        random_state=config.META_LEARNER_TRAINING["random_state"]
    )

    # Dynamically set input dimension
    config.META_LEARNER_CONFIG["input_dim"] = X_train.shape[1]

    # Build the meta-learner model
    model = Sequential()
    model.add(Dense(
        config.META_LEARNER_CONFIG["hidden_layers"][0], 
        activation=config.META_LEARNER_CONFIG["activation"]["hidden"], 
        input_dim=config.META_LEARNER_CONFIG["input_dim"]
    ))
    model.add(Dropout(config.META_LEARNER_CONFIG["dropout_rate"]))
    model.add(Dense(
        config.META_LEARNER_CONFIG["hidden_layers"][1], 
        activation=config.META_LEARNER_CONFIG["activation"]["hidden"]
    ))
    model.add(Dense(
        config.META_LEARNER_CONFIG["output_dim"], 
        activation=config.META_LEARNER_CONFIG["activation"]["output"]
    ))

    # Compile the model
    model.compile(
        optimizer="adam", 
        loss="mse", 
        metrics=["mae"]
    )

    # Train the model
    model.fit(
        X_train, y_train, 
        epochs=config.META_LEARNER_TRAINING["epochs"], 
        batch_size=config.META_LEARNER_TRAINING["batch_size"], 
        validation_data=(X_test, y_test)
    )

    return model