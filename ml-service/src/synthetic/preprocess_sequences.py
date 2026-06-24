import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def main():
    # 1. Load data/sample_sales.csv
    data_path = 'data/sample_sales.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # 2. Parse date as datetime
    # The date is used to ensure chronological ordering, but is not directly a model input.
    df['date'] = pd.to_datetime(df['date'])

    # 3. Sort rows by date
    # Essential for time-series forecasting to prevent look-ahead bias and keep sequences intact.
    df = df.sort_values(by='date').reset_index(drop=True)

    # 4. Select feature columns
    # These features are inputs to the LSTM model. 
    # Product is ignored because we only have one product, so it doesn't add any variance or predictive power.
    feature_cols = ['units_sold', 'price', 'stock', 'promotion', 'day_of_week', 'month']
    
    # 5. Set target column
    # The target is what we want the LSTM to predict.
    target_col = 'units_sold'

    print(f"Total rows: {len(df)}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target column: {target_col}")

    # 7. Normalize feature values using MinMaxScaler
    # Neural networks perform better when inputs are on a similar scale (e.g., 0 to 1).
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(df[feature_cols])

    # 8. Normalize target values using a separate MinMaxScaler
    # We use a separate scaler for the target so we can easily inverse-transform the predictions later.
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(df[[target_col]])

    # 9. Create sliding windows
    # lookback_days = 30
    # forecast_days = 7
    # Example:
    # X[0] = Day 1 to Day 30 feature values
    # y[0] = Day 31 to Day 37 target (units_sold) values
    lookback_days = 30
    forecast_days = 7

    X, y = [], []
    # We iterate until the index where we can still grab a full forecast window
    for i in range(len(df) - lookback_days - forecast_days + 1):
        X.append(scaled_features[i : i + lookback_days])
        y.append(scaled_target[i + lookback_days : i + lookback_days + forecast_days].flatten())

    X = np.array(X)
    y = np.array(y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # 10. Split data chronologically: 80% train, 20% test
    # Important: Do not randomly shuffle time-series data before splitting, otherwise the model
    # would "cheat" by learning from future events to predict past ones.
    split_index = int(len(X) * 0.8)
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"train shape: {X_train.shape}")
    print(f"test shape: {X_test.shape}")
    print(f"first input window shape: {X_train[0].shape}")
    print(f"first target window shape: {y_train[0].shape}")

    print("\n--- Explanation of one training sample ---")
    print(f"One input sample (X[0]) contains {lookback_days} days of historical data across {len(feature_cols)} features.")
    print(f"Its corresponding target (y[0]) contains the next {forecast_days} days of '{target_col}' values to predict.")

    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 11. Save processed arrays to data/processed/synthetic_sequences.npz
    np.savez('data/processed/synthetic_sequences.npz', 
             X_train=X_train, X_test=X_test, 
             y_train=y_train, y_test=y_test)
    print("\nSaved sequences to data/processed/synthetic_sequences.npz")

    # 12. Save scalers to models/
    joblib.dump(feature_scaler, 'models/feature_scaler.pkl')
    joblib.dump(target_scaler, 'models/target_scaler.pkl')
    print("Saved scalers to models/feature_scaler.pkl and models/target_scaler.pkl")

if __name__ == "__main__":
    main()
