import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib

def main():
    # 1. Load the processed CSV
    input_csv = "data/real/processed/rossmann_store_1_processed.csv"
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}")
        return

    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # 2. Parse Date as datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # 3. Sort by Date
    df = df.sort_values('Date').reset_index(drop=True)

    # 4. Print stats
    print("\n--- Dataset Stats ---")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Sales - Mean: {df['Sales'].mean():.2f}, Min: {df['Sales'].min()}, Max: {df['Sales'].max()}")

    # 5. Create feature engineering columns
    # A. lag_7_sales: Sales from the same day last week
    # Meaning: Sales from the same day last week (historical information)
    df['lag_7_sales'] = df['Sales'].shift(7)

    # B. rolling_7_sales: Rolling average of Sales over previous 7 days
    # Meaning: smooths out daily variations to provide a recent trend.
    # Important: Use shift(1) before rolling so future values are not leaked.
    df['rolling_7_sales'] = df['Sales'].shift(1).rolling(window=7).mean()

    # C. is_promo_day: copy Promo as numeric 0/1 if needed
    if 'Promo' in df.columns:
        df['is_promo_day'] = df['Promo'].astype(int)

    # 6. Drop rows with missing lag/rolling values
    df = df.dropna(subset=['lag_7_sales', 'rolling_7_sales']).reset_index(drop=True)
    print(f"\nRows after dropping missing lag/rolling values: {len(df)}")

    # 7. Define lookback and forecast
    lookback_days = 30
    forecast_days = 7

    # 8. Define past features
    past_feature_cols = [
        'Sales', 'Promo', 'SchoolHoliday', 'DayOfWeek', 'Month', 'Day', 
        'IsWeekend', 'lag_7_sales', 'rolling_7_sales'
    ]

    # 9. Define future known features
    future_feature_cols = [
        'Promo', 'SchoolHoliday', 'DayOfWeek', 'Month', 'Day', 
        'IsWeekend', 'lag_7_sales', 'rolling_7_sales'
    ]

    # 10. Target
    target_col = 'Sales'

    # 11. Remove any missing features and print a warning
    for col in past_feature_cols + future_feature_cols:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' is missing from the dataset. Removing from feature lists.")
            if col in past_feature_cols:
                past_feature_cols.remove(col)
            if col in future_feature_cols:
                future_feature_cols.remove(col)

    # Ensure unique cols to avoid modifying list during iteration incorrectly above, but simple remove is ok since we don't have duplicates in the same list.
    
    # Clean duplicates in future_feature_cols removal if they were missing (fixed by copying lists or just checking existing)
    past_feature_cols = [c for c in past_feature_cols if c in df.columns]
    future_feature_cols = [c for c in future_feature_cols if c in df.columns]

    # 12. Normalize
    print("\nNormalizing features...")
    past_scaler = MinMaxScaler()
    future_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Create scaled versions of the dataframe columns to build sequences
    df_past_scaled = pd.DataFrame(past_scaler.fit_transform(df[past_feature_cols]), columns=past_feature_cols)
    df_future_scaled = pd.DataFrame(future_scaler.fit_transform(df[future_feature_cols]), columns=future_feature_cols)
    
    # Scale target
    target_data = df[[target_col]].values
    target_scaled = target_scaler.fit_transform(target_data)
    
    # 13. Create sequences
    print("\nCreating sequences...")
    X_past, X_future, y = [], [], []
    
    # Iterate through the dataframe to build sequences
    for i in range(len(df) - lookback_days - forecast_days + 1):
        # Past 30 days
        X_past.append(df_past_scaled.iloc[i : i + lookback_days].values)
        
        # Future 7 days (known features only)
        X_future.append(df_future_scaled.iloc[i + lookback_days : i + lookback_days + forecast_days].values)
        
        # Target 7 days (Sales)
        y.append(target_scaled[i + lookback_days : i + lookback_days + forecast_days].flatten())

    X_past = np.array(X_past)
    X_future = np.array(X_future)
    y = np.array(y)

    # 14. Split chronologically (80% train, 20% test)
    # Do not randomly shuffle before splitting
    split_idx = int(len(X_past) * 0.8)
    
    X_past_train = X_past[:split_idx]
    X_future_train = X_future[:split_idx]
    y_train = y[:split_idx]
    
    X_past_test = X_past[split_idx:]
    X_future_test = X_future[split_idx:]
    y_test = y[split_idx:]

    # 15. Save arrays
    os.makedirs('data/real/processed', exist_ok=True)
    out_npz = 'data/real/processed/rossmann_store_1_sequences.npz'
    np.savez(
        out_npz,
        X_past_train=X_past_train,
        X_future_train=X_future_train,
        y_train=y_train,
        X_past_test=X_past_test,
        X_future_test=X_future_test,
        y_test=y_test
    )
    print(f"\nSaved sequences to {out_npz}")

    # 16. Save scalers
    os.makedirs('models/real', exist_ok=True)
    joblib.dump(past_scaler, 'models/real/rossmann_past_feature_scaler.pkl')
    joblib.dump(future_scaler, 'models/real/rossmann_future_feature_scaler.pkl')
    joblib.dump(target_scaler, 'models/real/rossmann_target_scaler.pkl')
    print("Saved scalers to models/real/")

    # 17. Calculate baseline metrics on the test set
    print("\nCalculating Baseline Metrics on Test Set...")
    
    # Unscale y_test for meaningful metrics
    y_test_unscaled = target_scaler.inverse_transform(y_test)
    
    # We need to unscale the past sales for the baselines
    sales_idx = past_feature_cols.index('Sales')
    X_past_sales_scaled = X_past_test[:, :, sales_idx]
    
    X_past_sales_unscaled = np.zeros_like(X_past_sales_scaled)
    for i in range(X_past_sales_scaled.shape[1]):
        X_past_sales_unscaled[:, i] = target_scaler.inverse_transform(X_past_sales_scaled[:, i].reshape(-1, 1)).flatten()

    # A. Last Value Baseline: repeat the last known Sales value for 7 days
    last_value = X_past_sales_unscaled[:, -1] # Shape: (samples,)
    last_value_pred = np.tile(last_value.reshape(-1, 1), (1, forecast_days)) # Shape: (samples, 7)
    
    # B. Weekly Seasonal Baseline: repeat the last 7 Sales values
    weekly_seasonal_pred = X_past_sales_unscaled[:, -7:] # Shape: (samples, 7)

    def calc_metrics(y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        # MAPE
        mask = y_true_flat != 0
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
        return mae, rmse, mape

    lv_mae, lv_rmse, lv_mape = calc_metrics(y_test_unscaled, last_value_pred)
    ws_mae, ws_rmse, ws_mape = calc_metrics(y_test_unscaled, weekly_seasonal_pred)

    # 18. Print shapes and baseline metrics
    print("\n--- Array Shapes ---")
    print(f"X_past shape: {X_past.shape}")
    print(f"X_future shape: {X_future.shape}")
    print(f"y shape: {y.shape}")
    print(f"Train shapes: X_past: {X_past_train.shape}, X_future: {X_future_train.shape}, y: {y_train.shape}")
    print(f"Test shapes: X_past: {X_past_test.shape}, X_future: {X_future_test.shape}, y: {y_test.shape}")

    baseline_output = (
        "--- Baseline Metrics (Test Set) ---\n"
        "1. Last Value Baseline:\n"
        f"   MAE:  {lv_mae:.2f}\n"
        f"   RMSE: {lv_rmse:.2f}\n"
        f"   MAPE: {lv_mape:.2f}%\n"
        "\n"
        "2. Weekly Seasonal Baseline:\n"
        f"   MAE:  {ws_mae:.2f}\n"
        f"   RMSE: {ws_rmse:.2f}\n"
        f"   MAPE: {ws_mape:.2f}%\n"
    )
    print("\n" + baseline_output)

    # 19. Save baseline metrics
    os.makedirs('reports/real', exist_ok=True)
    report_path = 'reports/real/rossmann_store_1_baseline_metrics.txt'
    with open(report_path, 'w') as f:
        f.write(baseline_output)
    print(f"Saved baseline metrics to {report_path}")

if __name__ == "__main__":
    main()
