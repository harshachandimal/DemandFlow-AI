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
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.linear1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, forecast_days)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        last_out = self.linear1(last_out)
        last_out = self.relu(last_out)
        predictions = self.linear2(last_out)
        return predictions

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

    # Initialize architecture
    input_size = 6
    forecast_days = 7
    
    model = DemandLSTM(
        input_size=input_size, 
        hidden_size=64, 
        num_layers=2, 
        dropout=0.2, 
        forecast_days=forecast_days
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode

    # 5. Predict on the full X_test set
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).numpy()

    # 8. Print shapes
    print("\n--- Evaluation Data Shapes ---")
    print(f"X_test shape: {X_test_np.shape}")
    print(f"y_test shape: {y_test_np.shape}")
    print(f"predictions shape: {predictions_scaled.shape}")

    # 6. Inverse transform
    # The scaler was fitted on 1D arrays, so we flatten, transform, and reshape back
    y_test_real = target_scaler.inverse_transform(y_test_np.reshape(-1, 1)).reshape(y_test_np.shape)
    predictions_real = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(predictions_scaled.shape)

    # 7. Calculate metrics (MAE, RMSE, MAPE) across the full test set
    # MAE: Mean Absolute Error - average of absolute differences
    absolute_errors = np.abs(predictions_real - y_test_real)
    mae = np.mean(absolute_errors)

    # RMSE: Root Mean Squared Error - penalizes larger errors more
    squared_errors = np.square(predictions_real - y_test_real)
    rmse = np.sqrt(np.mean(squared_errors))

    # MAPE: Mean Absolute Percentage Error - error relative to actual value
    # We add a small epsilon to denominator to prevent division by zero in case of 0 sales
    mape = np.mean(absolute_errors / (np.abs(y_test_real) + 1e-8)) * 100

    print("\n--- Overall Test Metrics ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f} units")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} units")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    # 9. Save a CSV file
    os.makedirs('reports', exist_ok=True)
    
    csv_data = []
    # Loop over all samples and all forecast days to create a long format CSV
    for sample_idx in range(len(y_test_real)):
        for day_idx in range(forecast_days):
            actual = y_test_real[sample_idx, day_idx]
            predicted = predictions_real[sample_idx, day_idx]
            abs_err = absolute_errors[sample_idx, day_idx]
            
            csv_data.append({
                'sample_index': sample_idx,
                'forecast_day': day_idx + 1,
                'actual_units_sold': actual,
                'predicted_units_sold': predicted,
                'absolute_error': abs_err
            })
            
    df_results = pd.DataFrame(csv_data)
    csv_path = 'reports/lstm_test_predictions.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nDetailed predictions saved to {csv_path}")

    # 10. Create charts
    
    # Chart 1: Average error by forecast day
    # Purpose: See if predicting further into the future (Day 7) is harder than near future (Day 1)
    error_by_day = [np.mean(absolute_errors[:, d]) for d in range(forecast_days)]
    
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, forecast_days + 1), error_by_day, color='coral')
    plt.title('Average Absolute Error by Forecast Day')
    plt.xlabel('Forecast Day (1 to 7)')
    plt.ylabel('Mean Absolute Error (Units)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    chart1_path = 'reports/lstm_error_by_forecast_day.png'
    plt.savefig(chart1_path)
    plt.close()
    print(f"Chart saved to {chart1_path}")

    # Chart 2: Flattened actual vs predicted values
    # Purpose: Visualize how well the model captures the overall variance across all test samples
    plt.figure(figsize=(12, 6))
    
    # We plot the first 100 flattened points to avoid an overly dense chart
    plot_limit = min(100, len(y_test_real.flatten()))
    
    plt.plot(y_test_real.flatten()[:plot_limit], label='Actual Sales', marker='.', alpha=0.8)
    plt.plot(predictions_real.flatten()[:plot_limit], label='Predicted Sales', marker='.', alpha=0.8)
    plt.title('Actual vs Predicted Sales (Flattened Test Samples Subset)')
    plt.xlabel('Time Step (in flattened sequence)')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chart2_path = 'reports/lstm_actual_vs_predicted_all_test.png'
    plt.savefig(chart2_path)
    plt.close()
    print(f"Chart saved to {chart2_path}")

if __name__ == "__main__":
    main()
