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
    df['date'] = pd.to_datetime(df['date'])

    # 3. Sort rows by date
    df = df.sort_values(by='date').reset_index(drop=True)

    # 4. Define window sizes
    lookback_days = 30
    forecast_days = 7

    # 5. Define feature columns
    past_feature_columns = [
        "units_sold",
        "price",
        "stock",
        "promotion",
        "day_of_week",
        "month"
    ]

    # Future known covariates
    # Important: Do not include future units_sold because that is the value we are trying to predict.
    # Do not include future stock for now because in real business future stock may not always be known.
    future_feature_columns = [
        "price",
        "promotion",
        "day_of_week",
        "month"
    ]

    target_column = "units_sold"

    print(f"Total rows: {len(df)}")
    print(f"Past feature columns: {past_feature_columns}")
    print(f"Future feature columns: {future_feature_columns}")
    print(f"Target column: {target_column}")

    # 6. Normalize using separate scalers
    past_feature_scaler = MinMaxScaler()
    scaled_past_features = past_feature_scaler.fit_transform(df[past_feature_columns])

    future_feature_scaler = MinMaxScaler()
    scaled_future_features = future_feature_scaler.fit_transform(df[future_feature_columns])

    future_target_scaler = MinMaxScaler()
    scaled_target = future_target_scaler.fit_transform(df[[target_column]])

    # 7. Create sliding windows
    X_past, X_future, y = [], [], []
    
    # We iterate until the index where we can still grab a full forecast window
    for i in range(len(df) - lookback_days - forecast_days + 1):
        # Previous 30 days of past features
        X_past.append(scaled_past_features[i : i + lookback_days])
        # Next 7 days of future known features
        X_future.append(scaled_future_features[i + lookback_days : i + lookback_days + forecast_days])
        # Next 7 days of units_sold (target)
        y.append(scaled_target[i + lookback_days : i + lookback_days + forecast_days].flatten())

    X_past = np.array(X_past)
    X_future = np.array(X_future)
    y = np.array(y)

    print(f"\nX_past shape: {X_past.shape}")
    print(f"X_future shape: {X_future.shape}")
    print(f"y shape: {y.shape}")

    # 8. Split chronologically: 80% train, 20% test
    # Do not randomly shuffle before splitting.
    split_index = int(len(X_past) * 0.8)
    
    X_past_train, X_past_test = X_past[:split_index], X_past[split_index:]
    X_future_train, X_future_test = X_future[:split_index], X_future[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print("\n--- Train/Test Split ---")
    print(f"train shapes: X_past={X_past_train.shape}, X_future={X_future_train.shape}, y={y_train.shape}")
    print(f"test shapes: X_past={X_past_test.shape}, X_future={X_future_test.shape}, y={y_test.shape}")
    print(f"first X_past sample shape: {X_past_train[0].shape}")
    print(f"first X_future sample shape: {X_future_train[0].shape}")
    print(f"first y sample shape: {y_train[0].shape}")

    # 12. Print a clear explanation
    print("\n--- ML Sequence Explanation ---")
    print("X_past teaches the model what happened before.")
    print("X_future tells the model known future business conditions.")
    print("y is what the model must predict.")

    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 9. Save processed arrays
    np.savez('data/processed/synthetic_future_sequences.npz', 
             X_past_train=X_past_train, X_past_test=X_past_test, 
             X_future_train=X_future_train, X_future_test=X_future_test,
             y_train=y_train, y_test=y_test)
    print("\nSaved sequences to data/processed/synthetic_future_sequences.npz")

    # 10. Save scalers
    joblib.dump(past_feature_scaler, 'models/past_feature_scaler.pkl')
    joblib.dump(future_feature_scaler, 'models/future_feature_scaler.pkl')
    joblib.dump(future_target_scaler, 'models/future_target_scaler.pkl')
    print("Saved scalers to models/past_feature_scaler.pkl, models/future_feature_scaler.pkl, and models/future_target_scaler.pkl")

if __name__ == "__main__":
    main()
