import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def select_best_model(data):
    features = data.drop(columns=["anomaly_column_name"])  # Replace with actual column name

    # Split the data into features and target
    X = features

    # Instantiate and train different models
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    isolation_forest.fit(X)

    return isolation_forest


if __name__ == "__main__":
    data_file_path = "data/preprocessed_data.csv"
    preprocessed_data = load_data(data_file_path)

    best_model = select_best_model(preprocessed_data)

    print("Best model selected:", best_model)
