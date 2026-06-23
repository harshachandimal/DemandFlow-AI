import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Directories
os.makedirs("reports/real", exist_ok=True)

# 1. Load sequence arrays
data = np.load("data/real/processed/rossmann_store_1_sequences.npz")
X_past_test = data['X_past_test']
X_future_test = data['X_future_test']
y_test = data['y_test']

# 2. Load the saved target scaler
target_scaler = joblib.load("models/real/rossmann_target_scaler.pkl")

# 3. Recreate the same RossmannFutureAwareLSTM architecture used during training
class RossmannFutureAwareLSTM(nn.Module):
    def __init__(self, past_features, future_features, forecast_days):
        super().__init__()
        
        self.forecast_days = forecast_days
        
        # A. Past encoder
        self.past_lstm = nn.LSTM(
            input_size=past_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # B. Future feature encoder
        self.future_encoder = nn.Sequential(
            nn.Linear(future_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # D. Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_past, x_future):
        # A. Encode past
        past_out, (hn, cn) = self.past_lstm(x_past)
        
        # C. Fusion
        past_context = hn[-1]
        past_context_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1)
        future_encoded = self.future_encoder(x_future) 
        fused = torch.cat((past_context_repeated, future_encoded), dim=2)
        
        # D. Predict
        predictions = self.prediction_head(fused)
        return predictions.squeeze(-1)

past_features = X_past_test.shape[2]
future_features = X_future_test.shape[2]
forecast_days = y_test.shape[1]

model = RossmannFutureAwareLSTM(past_features, future_features, forecast_days)

# 4. Load the saved model weights
model.load_state_dict(torch.load("models/real/rossmann_store_1_lstm_model.pth"))
model.eval()

# Convert arrays to PyTorch tensors
X_past_test_t = torch.tensor(X_past_test, dtype=torch.float32)
X_future_test_t = torch.tensor(X_future_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(X_past_test_t, X_future_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. Predict on the full test set
test_predictions = []
test_actuals = []

with torch.no_grad():
    for x_p, x_f, y in test_loader:
        preds = model(x_p, x_f)
        test_predictions.append(preds.numpy())
        test_actuals.append(y.numpy())
        
test_predictions = np.vstack(test_predictions)
test_actuals = np.vstack(test_actuals)

# 6. Inverse transform
y_test_inv = target_scaler.inverse_transform(test_actuals)
predictions_inv = target_scaler.inverse_transform(test_predictions)

# 7. Calculate overall MAE, RMSE, MAPE
overall_mae = np.mean(np.abs(y_test_inv - predictions_inv))
overall_rmse = np.sqrt(np.mean((y_test_inv - predictions_inv) ** 2))
overall_mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100

print(f"Overall MAE: {overall_mae:.2f}")
print(f"Overall RMSE: {overall_rmse:.2f}")
print(f"Overall MAPE: {overall_mape:.2f}%")

# 8. Calculate error by forecast day
day_maes = []
day_rmses = []
day_mapes = []

print("\n9. Error by Forecast Day:")
print(f"{'Forecast Day':<15} | {'MAE':<10} | {'RMSE':<10} | {'MAPE':<10}")
print("-" * 55)
for d in range(forecast_days):
    d_actual = y_test_inv[:, d]
    d_pred = predictions_inv[:, d]
    
    d_mae = np.mean(np.abs(d_actual - d_pred))
    d_rmse = np.sqrt(np.mean((d_actual - d_pred) ** 2))
    d_mape = np.mean(np.abs((d_actual - d_pred) / d_actual)) * 100
    
    day_maes.append(d_mae)
    day_rmses.append(d_rmse)
    day_mapes.append(d_mape)
    
    print(f"Day {d+1:<11} | {d_mae:<10.2f} | {d_rmse:<10.2f} | {d_mape:.2f}%")

# 10. Save detailed predictions to CSV
records = []
for i in range(len(y_test_inv)):
    for d in range(forecast_days):
        actual = y_test_inv[i, d]
        predicted = predictions_inv[i, d]
        abs_err = np.abs(actual - predicted)
        pct_err = abs_err / actual * 100
        records.append({
            'sample_index': i,
            'forecast_day': d + 1,
            'actual_sales': actual,
            'predicted_sales': predicted,
            'absolute_error': abs_err,
            'percentage_error': pct_err
        })

df_detailed = pd.DataFrame(records)
df_detailed.to_csv("reports/real/rossmann_store_1_detailed_predictions.csv", index=False)

# 11. Find the worst 10 predictions by absolute error and save to CSV
df_worst = df_detailed.sort_values(by='absolute_error', ascending=False).head(10)
df_worst.to_csv("reports/real/rossmann_store_1_worst_predictions.csv", index=False)

# 12. Create charts
# A. reports/real/rossmann_store_1_error_by_forecast_day.png
plt.figure(figsize=(10, 6))
plt.bar(range(1, forecast_days + 1), day_maes, color='skyblue')
plt.title("LSTM MAE by Forecast Day (Store 1)")
plt.xlabel("Forecast Day")
plt.ylabel("Mean Absolute Error (MAE)")
plt.xticks(range(1, forecast_days + 1))
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_error_by_forecast_day.png")
plt.close()

# B. reports/real/rossmann_store_1_actual_vs_predicted_all_test.png
plt.figure(figsize=(15, 6))
# Plot first 30 sequences, flattened to show the continuous sequence effect
flattened_actual = y_test_inv[:30].flatten()
flattened_pred = predictions_inv[:30].flatten()
plt.plot(flattened_actual, label="Actual Sales", alpha=0.8)
plt.plot(flattened_pred, label="Predicted Sales", alpha=0.8)
plt.title("Actual vs Predicted - First 30 Test Sequences Flattened (Store 1)")
plt.xlabel("Days")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_actual_vs_predicted_all_test.png")
plt.close()

# C. reports/real/rossmann_store_1_baseline_vs_lstm_mape.png
baseline_last_value_mape = 20.96
baseline_weekly_mape = 25.79

plt.figure(figsize=(8, 6))
models = ['Last Value Baseline', 'Weekly Seasonal Baseline', 'LSTM']
mapes = [baseline_last_value_mape, baseline_weekly_mape, overall_mape]
colors = ['lightcoral', 'orange', 'lightgreen']

bars = plt.bar(models, mapes, color=colors)
plt.title("MAPE Comparison: Baselines vs LSTM (Store 1)")
plt.ylabel("MAPE (%)")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_baseline_vs_lstm_mape.png")
plt.close()

# 13. Save evaluation summary to reports/real/rossmann_store_1_lstm_detailed_metrics.txt
with open("reports/real/rossmann_store_1_lstm_detailed_metrics.txt", "w") as f:
    f.write("Rossmann Store 1 - LSTM Detailed Evaluation\n")
    f.write("===========================================\n\n")
    f.write(f"Overall MAE: {overall_mae:.2f}\n")
    f.write(f"Overall RMSE: {overall_rmse:.2f}\n")
    f.write(f"Overall MAPE: {overall_mape:.2f}%\n\n")
    f.write("Error by Forecast Day:\n")
    f.write(f"{'Forecast Day':<15} | {'MAE':<10} | {'RMSE':<10} | {'MAPE':<10}\n")
    f.write("-" * 55 + "\n")
    for d in range(forecast_days):
        f.write(f"Day {d+1:<11} | {day_maes[d]:<10.2f} | {day_rmses[d]:<10.2f} | {day_mapes[d]:.2f}%\n")
    
    f.write("\nBaseline Comparison:\n")
    f.write(f"Last Value Baseline MAPE: {baseline_last_value_mape:.2f}%\n")
    f.write(f"Weekly Seasonal Baseline MAPE: {baseline_weekly_mape:.2f}%\n")
    f.write(f"LSTM MAPE: {overall_mape:.2f}%\n")
    f.write("\nConclusion:\n")
    if overall_mape < min(baseline_last_value_mape, baseline_weekly_mape):
        f.write("LSTM beats both baselines significantly, proving its value over heuristics.\n")
    else:
        f.write("LSTM does NOT beat both baselines.\n")
