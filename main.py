import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)

def load_data(filepath):
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise

def preprocess_data(df):
    """Clean and preprocess the dataframe."""
    # Binary encoding
    binary_columns = [
        'Smoking', 'Recent Major Life Event', 'Dizziness',
        'Medication', 'Family History of Anxiety'
    ]
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Encode gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2, 'Other': 3})

    # Encode occupation
    occupation_map = {occ: idx for idx, occ in enumerate(df['Occupation'].unique())}
    df['Occupation'] = df['Occupation'].map(occupation_map)

    logging.info("Data preprocessing complete.")
    return df

def train_model(X, y):
    """Train a regression model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X_scaled, y)

    logging.info("Model training complete.")
    return model, scaler
def main():
    data_path = "enhanced_anxiety_dataset.csv"
    df = load_data(data_path)
    df = preprocess_data(df)

    y = df.pop('Anxiety Level (1-10)')
    X = df

    model, scaler = train_model(X, y)

if __name__ == "__main__":
    main()
