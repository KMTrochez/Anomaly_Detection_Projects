import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib


def train_anomaly_model(data):
    # Instantiate the Isolation Forest model
    model = IsolationForest(contamination=0.05, random_state=42)  # You can adjust contamination as needed

    # Fit the model to the preprocessed data
    model.fit(data)

    return model


def save_model(model, model_filename):
    # Save the trained model to a file using joblib
    joblib.dump(model, model_filename)


if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_data = pd.read_csv("data/preprocessed_data.csv")

    # Extract the features for training
    features = preprocessed_data.drop(columns=["anomaly_column_name"])  # Replace with actual column name

    # Train the anomaly detection model
    trained_model = train_anomaly_model(features)

    # Save the trained model to a file
    model_filename = "trained_anomaly_model.joblib"
    save_model(trained_model, model_filename)

    print("Anomaly detection model trained and saved.")
