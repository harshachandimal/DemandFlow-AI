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
    
    # 11. Print total rows before feature engineering
    print(f"Total rows before feature engineering: {len(df)}")

    # Feature Engineering
    
    # 1. Create is_weekend
    # if day_of_week is 5 (Saturday) or 6 (Sunday), is_weekend = 1, otherwise 0
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)

    # 2. Create rolling_7_day_average
    # rolling mean of units_sold over the previous 7 days.
    # Important: We shift by 1 before rolling so that today's average 
    # only includes up to yesterday, preventing future data leakage.
    df['rolling_7_day_average'] = df['units_sold'].shift(1).rolling(window=7).mean()

    # 3. Create same_day_last_week_sales
    # units_sold shifted by 7 days. 
    # Today's same_day_last_week_sales is the sales from exactly 7 days ago.
    df['same_day_last_week_sales'] = df['units_sold'].shift(7)

    # 4. Drop rows with missing lag/rolling values created at the beginning.
    df = df.dropna().reset_index(drop=True)
    
    # 11. Print total rows after dropping
    print(f"Total rows after dropping missing lag rows: {len(df)}")

    # Define window sizes
    lookback_days = 30
    forecast_days = 7

    past_feature_columns = [
        "units_sold",
        "price",
        "stock",
        "promotion",
        "day_of_week",
        "month",
        "is_weekend",
        "rolling_7_day_average"
    ]

    # Future known/lag features
    # Do not include future units_sold directly.
    # same_day_last_week_sales is allowed because it comes from 7 days before the future day,
    # which is already historical information.
    future_feature_columns = [
        "price",
        "promotion",
        "day_of_week",
        "month",
        "is_weekend",
        "same_day_last_week_sales",
        "rolling_7_day_average"
    ]

    target_column = "units_sold"

    print(f"Past feature columns: {past_feature_columns}")
    print(f"Future feature columns: {future_feature_columns}")
    print(f"Target column: {target_column}")

    # 5. Normalize using separate scalers
    past_feature_scaler = MinMaxScaler()
    scaled_past_features = past_feature_scaler.fit_transform(df[past_feature_columns])

    future_feature_scaler = MinMaxScaler()
    scaled_future_features = future_feature_scaler.fit_transform(df[future_feature_columns])

    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(df[[target_column]])

    # 6. Create sliding sequences
    X_past, X_future, y = [], [], []
    
    for i in range(len(df) - lookback_days - forecast_days + 1):
        # Previous 30 days of past features
        X_past.append(scaled_past_features[i : i + lookback_days])
        # Next 7 days of future known/lag features
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
    split_index = int(len(X_past) * 0.8)
    
    X_past_train, X_past_test = X_past[:split_index], X_past[split_index:]
    X_future_train, X_future_test = X_future[:split_index], X_future[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print("\n--- Train/Test Split ---")
    print(f"train shapes: X_past={X_past_train.shape}, X_future={X_future_train.shape}, y={y_train.shape}")
    print(f"test shapes: X_past={X_past_test.shape}, X_future={X_future_test.shape}, y={y_test.shape}")

    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 9. Save processed arrays
    np.savez('data/processed/synthetic_future_sequences_v3.npz', 
             X_past_train=X_past_train, X_past_test=X_past_test, 
             X_future_train=X_future_train, X_future_test=X_future_test,
             y_train=y_train, y_test=y_test)
    print("\nSaved sequences to data/processed/synthetic_future_sequences_v3.npz")

    # 10. Save scalers
    joblib.dump(past_feature_scaler, 'models/v3_past_feature_scaler.pkl')
    joblib.dump(future_feature_scaler, 'models/v3_future_feature_scaler.pkl')
    joblib.dump(target_scaler, 'models/v3_target_scaler.pkl')
    print("Saved scalers to models/v3_past_feature_scaler.pkl, models/v3_future_feature_scaler.pkl, and models/v3_target_scaler.pkl")

if __name__ == "__main__":
    main()
