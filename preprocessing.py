import pandas as pd


def load_data(file_path):
    # Load the raw data from the CSV file
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    # Handle missing values by either removing or imputing them
    data.dropna(inplace=True)  # Remove rows with missing values

    # Perform other data cleaning steps if necessary

    return data


def normalize_features(data):
    # Normalize the features to have zero mean and unit variance
    normalized_data = (data - data.mean()) / data.std()
    return normalized_data


def preprocess_main(file_path):
    # Load the raw data
    raw_data = load_data(file_path)

    # Clean the data
    cleaned_data = clean_data(raw_data)

    # Normalize features
    normalized_data = normalize_features(cleaned_data)

    return normalized_data


if __name__ == "__main__":
    data_file_path = "data/raw_data.csv"  # Path to your raw data file
    preprocessed_data = preprocess_main(data_file_path)

    # Save the preprocessed data to a new CSV file
    preprocessed_data.to_csv("data/preprocessed_data.csv", index=False)
