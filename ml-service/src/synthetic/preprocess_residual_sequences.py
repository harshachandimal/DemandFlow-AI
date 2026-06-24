import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

    # 3. Sort by date
    df = df.sort_values(by='date').reset_index(drop=True)
    
    rows_before = len(df)
    print(f"Total rows before feature engineering: {rows_before}")

    # 4. Create engineered features
    # is_weekend
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)

    # rolling_7_day_average using only past values (shifted by 1 to prevent data leakage)
    df['rolling_7_day_average'] = df['units_sold'].shift(1).rolling(window=7).mean()

    # same_day_last_week_sales (units_sold shifted by 7 days)
    df['same_day_last_week_sales'] = df['units_sold'].shift(7)

    # 9. Create residual column: target residual = units_sold - same_day_last_week_sales
    df['residual'] = df['units_sold'] - df['same_day_last_week_sales']

    # 5. Drop rows with missing lag/rolling/residual values
    df = df.dropna().reset_index(drop=True)
    rows_after = len(df)
    print(f"Total rows after dropping missing values: {rows_after}")

    # 6. Define sequence dimensions
    lookback_days = 30
    forecast_days = 7

    # 7. Define past features
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

    # 8. Define future features
    future_feature_columns = [
        "price",
        "promotion",
        "day_of_week",
        "month",
        "is_weekend",
        "same_day_last_week_sales",
        "rolling_7_day_average"
    ]

    print(f"\nPast features ({len(past_feature_columns)}): {past_feature_columns}")
    print(f"Future features ({len(future_feature_columns)}): {future_feature_columns}")
    print("Target: residual (units_sold - same_day_last_week_sales)")

    # 11. Normalize
    # past features with MinMaxScaler
    past_feature_scaler = MinMaxScaler()
    scaled_past_features = past_feature_scaler.fit_transform(df[past_feature_columns])

    # future features with MinMaxScaler
    future_feature_scaler = MinMaxScaler()
    scaled_future_features = future_feature_scaler.fit_transform(df[future_feature_columns])

    # target residual with StandardScaler (since residuals can be negative and represent deviations)
    residual_target_scaler = StandardScaler()
    scaled_residual_target = residual_target_scaler.fit_transform(df[['residual']])

    # 10. Create sliding sequences
    X_past, X_future, y_residual, baseline_future, y_actual = [], [], [], [], []

    for i in range(len(df) - lookback_days - forecast_days + 1):
        # previous 30 days of past features
        X_past.append(scaled_past_features[i : i + lookback_days])
        
        # next 7 days of future known features
        X_future.append(scaled_future_features[i + lookback_days : i + lookback_days + forecast_days])
        
        # next 7 days residual values (scaled target)
        y_residual.append(scaled_residual_target[i + lookback_days : i + lookback_days + forecast_days].flatten())
        
        # next 7 days same_day_last_week_sales (unscaled baseline values)
        baseline_future.append(df['same_day_last_week_sales'].values[i + lookback_days : i + lookback_days + forecast_days])
        
        # next 7 days actual units_sold (unscaled actual values)
        y_actual.append(df['units_sold'].values[i + lookback_days : i + lookback_days + forecast_days])

    X_past = np.array(X_past)
    X_future = np.array(X_future)
    y_residual = np.array(y_residual)
    baseline_future = np.array(baseline_future)
    y_actual = np.array(y_actual)

    # 15. Print sequence shapes
    print(f"\n--- Processed Array Shapes ---")
    print(f"X_past shape       : {X_past.shape}")
    print(f"X_future shape     : {X_future.shape}")
    print(f"y_residual shape   : {y_residual.shape}")
    print(f"baseline shape     : {baseline_future.shape}")
    print(f"y_actual shape     : {y_actual.shape}")

    # 12. Split chronologically: 80% train, 20% test
    split_index = int(len(X_past) * 0.8)

    X_past_train, X_past_test = X_past[:split_index], X_past[split_index:]
    X_future_train, X_future_test = X_future[:split_index], X_future[split_index:]
    y_residual_train, y_residual_test = y_residual[:split_index], y_residual[split_index:]
    baseline_train, baseline_test = baseline_future[:split_index], baseline_future[split_index:]
    y_actual_train, y_actual_test = y_actual[:split_index], y_actual[split_index:]

    # 15. Print train/test split shapes
    print("\n--- Train/Test Split Shapes ---")
    print(f"Train: X_past={X_past_train.shape}, X_future={X_future_train.shape}, y_residual={y_residual_train.shape}, baseline={baseline_train.shape}, y_actual={y_actual_train.shape}")
    print(f"Test : X_past={X_past_test.shape}, X_future={X_future_test.shape}, y_residual={y_residual_test.shape}, baseline={baseline_test.shape}, y_actual={y_actual_test.shape}")

    # 16. Explanation
    print("\n--- ML Design Explanation ---")
    print("1. Weekly baseline (same_day_last_week_sales) gives the starting prediction.")
    print("   Since retail sales follow strong weekly cycles, last week's sales on the same weekday")
    print("   is a highly accurate heuristic reference.")
    print("2. The residual is the correction: residual = actual_sales - weekly_baseline.")
    print("   This represents local fluctuations due to prices, promotions, or random noise.")
    print("3. The neural network learns to predict the CORRECTION (residual), not the full sales value.")
    print("   By letting the network model only the residual deviation, we dramatically lower the complexity")
    print("   of the function it has to approximate, leading to better generalization and accuracy.")

    # 13. Save processed arrays
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    np.savez('data/processed/synthetic_residual_sequences.npz',
             X_past_train=X_past_train, X_past_test=X_past_test,
             X_future_train=X_future_train, X_future_test=X_future_test,
             y_residual_train=y_residual_train, y_residual_test=y_residual_test,
             baseline_train=baseline_train, baseline_test=baseline_test,
             y_actual_train=y_actual_train, y_actual_test=y_actual_test)
    print("\nSaved sequences to data/processed/synthetic_residual_sequences.npz")

    # 14. Save scalers
    joblib.dump(past_feature_scaler, 'models/residual_past_feature_scaler.pkl')
    joblib.dump(future_feature_scaler, 'models/residual_future_feature_scaler.pkl')
    joblib.dump(residual_target_scaler, 'models/residual_target_scaler.pkl')
    print("Saved scalers to models/residual_past_feature_scaler.pkl, models/residual_future_feature_scaler.pkl, and models/residual_target_scaler.pkl")

if __name__ == "__main__":
    main()
