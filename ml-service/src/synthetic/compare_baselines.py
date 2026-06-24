import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# 4. Recreate the same DemandLSTM architecture
class DemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, forecast_days=7):
        super(DemandLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.linear1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, forecast_days)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        last_out = self.linear1(last_out)
        last_out = self.relu(last_out)
        return self.linear2(last_out)

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
    mape = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100
    return mae, rmse, mape

def main():
    # 1. Load processed arrays
    data_path = 'data/processed/synthetic_sequences.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("Loading test sequences...")
    with np.load(data_path) as data:
        X_test_np = data['X_test']
        y_test_np = data['y_test']

    # 2. Load target scaler
    scaler_path = 'models/target_scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found.")
        return
    target_scaler = joblib.load(scaler_path)

    # 3. Load the saved model
    model_path = 'models/lstm_demand_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    input_size = 6
    forecast_days = 7
    model = DemandLSTM(input_size=input_size, forecast_days=forecast_days)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 5. Generate predictions for X_test using the LSTM
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    with torch.no_grad():
        lstm_pred_scaled = model(X_test_tensor).numpy()

    # 6. Create Baseline 1: Last-value baseline
    # For each test sample, take the last known units_sold value (feature index 0)
    # from the 30-day input window and repeat it 7 times
    # X_test_np shape is (samples, 30, 6)
    last_value_scaled = X_test_np[:, -1, 0] # Shape (samples,)
    last_value_pred_scaled = np.tile(last_value_scaled.reshape(-1, 1), (1, forecast_days)) # Shape (samples, 7)

    # 7. Create Baseline 2: Weekly seasonal baseline
    # For each test sample, take the last 7 days of units_sold from the input window
    weekly_seasonal_pred_scaled = X_test_np[:, -7:, 0] # Shape (samples, 7)

    # 8. Un-scale all predictions and y_test to real units_sold
    # The scaler expects a 2D array of a single feature, so we reshape(-1, 1), transform, and reshape back
    y_test_real = target_scaler.inverse_transform(y_test_np.reshape(-1, 1)).reshape(-1, forecast_days)
    lstm_pred_real = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).reshape(-1, forecast_days)
    last_value_pred_real = target_scaler.inverse_transform(last_value_pred_scaled.reshape(-1, 1)).reshape(-1, forecast_days)
    weekly_seasonal_pred_real = target_scaler.inverse_transform(weekly_seasonal_pred_scaled.reshape(-1, 1)).reshape(-1, forecast_days)

    # 9. Calculate metrics
    metrics = {}
    metrics['LSTM'] = calculate_metrics(y_test_real, lstm_pred_real)
    metrics['Last Value Baseline'] = calculate_metrics(y_test_real, last_value_pred_real)
    metrics['Weekly Seasonal Baseline'] = calculate_metrics(y_test_real, weekly_seasonal_pred_real)

    # 10. Print comparison table
    print("\n--- Baseline Comparison ---")
    print(f"{'Method':<30} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 65)
    for method, (mae, rmse, mape) in metrics.items():
        print(f"{method:<30} {mae:<10.2f} {rmse:<10.2f} {mape:<10.2f}%")

    # 11. Save the comparison to reports/baseline_comparison.csv
    os.makedirs('reports', exist_ok=True)
    df_metrics = pd.DataFrame.from_dict(
        metrics, 
        orient='index', 
        columns=['MAE', 'RMSE', 'MAPE']
    )
    df_metrics.index.name = 'Method'
    df_metrics.reset_index(inplace=True)
    csv_path = 'reports/baseline_comparison.csv'
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nComparison saved to {csv_path}")

    # 12. Create a bar chart
    plt.figure(figsize=(10, 6))
    methods = df_metrics['Method']
    mapes = df_metrics['MAPE']
    
    bars = plt.bar(methods, mapes, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('MAPE Comparison: LSTM vs Baselines')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    chart_path = 'reports/baseline_comparison_mape.png'
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    main()
