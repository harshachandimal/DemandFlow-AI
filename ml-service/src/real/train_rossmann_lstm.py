import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Directories
os.makedirs("models/real", exist_ok=True)
os.makedirs("reports/real", exist_ok=True)

# 1. Load sequence arrays
data = np.load("data/real/processed/rossmann_store_1_sequences.npz")
X_past_train_full = data['X_past_train']
X_future_train_full = data['X_future_train']
y_train_full = data['y_train']
X_past_test = data['X_past_test']
X_future_test = data['X_future_test']
y_test = data['y_test']

target_scaler = joblib.load("models/real/rossmann_target_scaler.pkl")

# 2. Print shapes
print("Initial loaded shapes:")
print(f"X_past_train_full shape: {X_past_train_full.shape}")
print(f"X_future_train_full shape: {X_future_train_full.shape}")
print(f"y_train_full shape: {y_train_full.shape}")
print(f"X_past_test shape: {X_past_test.shape}")
print(f"X_future_test shape: {X_future_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 3. Split the existing training set chronologically into 80% train, 20% validation
# Important: Do not randomly split time-series data.
split_idx = int(len(X_past_train_full) * 0.8)

X_past_train = X_past_train_full[:split_idx]
X_future_train = X_future_train_full[:split_idx]
y_train = y_train_full[:split_idx]

X_past_val = X_past_train_full[split_idx:]
X_future_val = X_future_train_full[split_idx:]
y_val = y_train_full[split_idx:]

print("\nAfter split:")
print(f"X_past_train shape: {X_past_train.shape}")
print(f"X_future_train shape: {X_future_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_past_val shape: {X_past_val.shape}")
print(f"X_future_val shape: {X_future_val.shape}")
print(f"y_val shape: {y_val.shape}")

# 4. Convert arrays to PyTorch tensors
X_past_train_t = torch.tensor(X_past_train, dtype=torch.float32)
X_future_train_t = torch.tensor(X_future_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_past_val_t = torch.tensor(X_past_val, dtype=torch.float32)
X_future_val_t = torch.tensor(X_future_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

X_past_test_t = torch.tensor(X_past_test, dtype=torch.float32)
X_future_test_t = torch.tensor(X_future_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# 5. Create TensorDataset and DataLoader
batch_size = 16

train_dataset = TensorDataset(X_past_train_t, X_future_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_past_val_t, X_future_val_t, y_val_t)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_past_test_t, X_future_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 6. Build model class
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
        # past_out: (batch_size, seq_len, hidden_size)
        past_out, (hn, cn) = self.past_lstm(x_past)
        
        # C. Fusion
        # Take last hidden output from LSTM as past context
        # hn[-1] shape: (batch_size, 64)
        past_context = hn[-1]
        
        # Repeat past context for forecast_days
        # Shape: (batch_size, forecast_days, 64)
        past_context_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1)
        
        # B. Encode future features
        # x_future shape: (batch_size, forecast_days, future_features)
        future_encoded = self.future_encoder(x_future) # Shape: (batch_size, forecast_days, 32)
        
        # Concatenate repeated past context with encoded future features
        # Shape: (batch_size, forecast_days, 64 + 32)
        fused = torch.cat((past_context_repeated, future_encoded), dim=2)
        
        # D. Predict
        # Shape: (batch_size, forecast_days, 1)
        predictions = self.prediction_head(fused)
        
        # Output shape: (batch_size, forecast_days)
        return predictions.squeeze(-1)

past_features = X_past_train.shape[2]
future_features = X_future_train.shape[2]
forecast_days = y_train.shape[1]

model = RossmannFutureAwareLSTM(past_features, future_features, forecast_days)

# 7. Use: MSELoss, Adam, learning_rate=0.001, weight_decay=1e-4, max_epochs=300, patience=25
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

max_epochs = 300
early_stopping_patience = 25

# 8. Train with early stopping based on validation loss
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
stopped_epoch = max_epochs

train_losses = []
val_losses = []

print("\nStarting training...")
for epoch in range(1, max_epochs + 1):
    # Train
    model.train()
    epoch_train_loss = 0
    for x_p, x_f, y in train_loader:
        optimizer.zero_grad()
        preds = model(x_p, x_f)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * len(y)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validate
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for x_p, x_f, y in val_loader:
            preds = model(x_p, x_f)
            loss = criterion(preds, y)
            epoch_val_loss += loss.item() * len(y)
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    # Early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        
    # Print every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{max_epochs}] - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Best Val Loss: {best_val_loss:.4f}")
        
    if patience_counter >= early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch}")
        stopped_epoch = epoch
        break

# 9. Restore best validation model weights
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# 10. Predict on test set
model.eval()
test_predictions = []
test_actuals = []

test_mse = 0
with torch.no_grad():
    for x_p, x_f, y in test_loader:
        preds = model(x_p, x_f)
        loss = criterion(preds, y)
        test_mse += loss.item() * len(y)
        test_predictions.append(preds.numpy())
        test_actuals.append(y.numpy())
        
test_mse /= len(test_loader.dataset)
test_predictions = np.vstack(test_predictions)
test_actuals = np.vstack(test_actuals)

# 11. Inverse transform
y_test_inv = target_scaler.inverse_transform(test_actuals)
predictions_inv = target_scaler.inverse_transform(test_predictions)

# 12. Calculate MAE, RMSE, MAPE
mae = np.mean(np.abs(y_test_inv - predictions_inv))
rmse = np.sqrt(np.mean((y_test_inv - predictions_inv) ** 2))
mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100

# 13. Print final metrics
print("\n--- Training Complete ---")
print(f"Stopped Epoch: {stopped_epoch}")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Final Test MSE: {test_mse:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

print("\nSample Next 7 Days (Actual):")
print(np.round(y_test_inv[0]).astype(int))
print("Sample Next 7 Days (Predicted):")
print(np.round(predictions_inv[0]).astype(int))

# 14. Compare against baselines
baseline_last_value_mape = 20.96
baseline_weekly_mape = 25.79

print("\n--- Baseline Comparison ---")
if mape < baseline_last_value_mape:
    print(f"LSTM ({mape:.2f}%) BEATS Last Value Baseline ({baseline_last_value_mape:.2f}%)")
else:
    print(f"LSTM ({mape:.2f}%) DOES NOT beat Last Value Baseline ({baseline_last_value_mape:.2f}%)")

if mape < baseline_weekly_mape:
    print(f"LSTM ({mape:.2f}%) BEATS Weekly Seasonal Baseline ({baseline_weekly_mape:.2f}%)")
else:
    print(f"LSTM ({mape:.2f}%) DOES NOT beat Weekly Seasonal Baseline ({baseline_weekly_mape:.2f}%)")

# 15. Save model
torch.save(model.state_dict(), "models/real/rossmann_store_1_lstm_model.pth")

# 16. Save chart: prediction vs actual
plt.figure(figsize=(10, 6))
plt.plot(y_test_inv[0], label="Actual (Sample 0)", marker='o')
plt.plot(predictions_inv[0], label="Predicted (Sample 0)", marker='x')
plt.title("LSTM Prediction vs Actual (Store 1)")
plt.xlabel("Days into Future")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_lstm_prediction_vs_actual.png")
plt.close()

# 17. Save loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
best_epoch_idx = stopped_epoch - patience_counter if stopped_epoch != max_epochs else max_epochs
# Need to be careful with patience counter vs best epoch if we haven't stopped yet
# but since stopped_epoch is handled, we can just use the index of best_val_loss
best_epoch_true = np.argmin(val_losses) + 1
plt.axvline(x=best_epoch_true, color='r', linestyle='--', label='Best Model')
plt.title("LSTM Training & Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_lstm_loss_curve.png")
plt.close()

# 18. Save metrics to text file
with open("reports/real/rossmann_store_1_lstm_metrics.txt", "w") as f:
    f.write("Rossmann Store 1 - LSTM v1 Metrics\n")
    f.write("==================================\n")
    f.write(f"Stopped Epoch: {stopped_epoch}\n")
    f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
    f.write(f"Final Test MSE: {test_mse:.4f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")
    f.write("\nBaseline Comparison:\n")
    f.write(f"Last Value Baseline MAPE: {baseline_last_value_mape:.2f}%\n")
    f.write(f"Weekly Seasonal Baseline MAPE: {baseline_weekly_mape:.2f}%\n")
