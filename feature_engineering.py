import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def create_additional_features(data):
    # Example: Create a new feature by combining existing ones
    data["total_sales"] = data["sales_quantity"] * data["unit_price"]

    # Add more feature engineering steps as needed

    return data


if __name__ == "__main__":
    data_file_path = "data/preprocessed_data.csv"
    preprocessed_data = load_data(data_file_path)

    engineered_data = create_additional_features(preprocessed_data)

    # Save the engineered data to a new CSV file
    engineered_data.to_csv("data/engineered_data.csv", index=False)

    print("Feature engineering completed and data saved.")
