"""
train_lstm_residual.py
======================
Phase 1.11 – Train LSTM v4: Residual Forecasting Model

New in v4 vs v3:
  • Instead of predicting units_sold directly, the model predicts the residual correction:
    residual = actual_sales - weekly_baseline
  • final_prediction = weekly_baseline + predicted_residual
  • Uses chronological validation split (80/20) and Early Stopping (patience=20)
  • Integrates weight_decay = 1e-4 and Dropout (0.3/0.2) for regularization
  • Evaluates the hybrid model against the standalone Weekly Seasonal Baseline

Learning goals:
  • Understand what residual modeling is and why it's powerful for seasonal datasets.
  • Understand how to combine linear heuristics (baselines) with non-linear neural networks (LSTMs).
  • Follow the math of splitting, training, reconstructing, and comparing residual errors.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================================================================
# MODEL DEFINITION
# ===========================================================================

class ResidualFutureAwareLSTM(nn.Module):
    """
    Architecture identical to SeasonalFutureAwareLSTM, but trained to output
    residual deviations (corrections) instead of absolute units_sold values.
    """
    def __init__(
        self,
        past_input_size,
        future_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        forecast_days=7
    ):
        super(ResidualFutureAwareLSTM, self).__init__()
        self.forecast_days = forecast_days

        # A. Past LSTM Encoder
        self.past_encoder = nn.LSTM(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # B. Future Encoder
        self.future_encoder = nn.Sequential(
            nn.Linear(future_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # D. Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_past, x_future):
        # x_past shape: (batch_size, 30, past_input_size)
        # x_future shape: (batch_size, 7, future_input_size)
        
        # Stream A: Process history
        lstm_out, _ = self.past_encoder(x_past)
        past_context = lstm_out[:, -1, :] # (batch_size, 64)

        # Stream B: Encode future features
        encoded_future = self.future_encoder(x_future) # (batch_size, 7, 32)

        # Fusion: repeat past context for each forecast day
        past_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1) # (batch_size, 7, 64)
        combined = torch.cat((past_repeated, encoded_future), dim=2) # (batch_size, 7, 96)

        # Head outputs predicted residuals
        out = self.prediction_head(combined) # (batch_size, 7, 1)
        return out.squeeze(2) # (batch_size, 7)


# ===========================================================================
# METRIC HELPERS
# ===========================================================================

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
    mape = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100
    return mae, rmse, mape


# ===========================================================================
# MAIN TRAINING & EVALUATION
# ===========================================================================

def main():
    # 1. Load residual sequence arrays
    data_path = 'data/processed/synthetic_residual_sequences.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Run: python src/preprocess_residual_sequences.py")
        return

    print("Loading residual sequence dataset...")
    with np.load(data_path) as data:
        X_past_train_full   = data['X_past_train']
        X_future_train_full = data['X_future_train']
        y_residual_train_full = data['y_residual_train']
        baseline_train_full = data['baseline_train']
        y_actual_train_full = data['y_actual_train']
        
        X_past_test         = data['X_past_test']
        X_future_test       = data['X_future_test']
        y_residual_test     = data['y_residual_test']
        baseline_test       = data['baseline_test']
        y_actual_test       = data['y_actual_test']

    # 2. Load target scaler (fitted on raw residuals)
    scaler_path = 'models/residual_target_scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found.")
        return
    residual_target_scaler = joblib.load(scaler_path)
    print(f"Loaded target scaler from {scaler_path}")

    # 4. Split training data chronologically: 80% training, 20% validation
    n_total = len(X_past_train_full)
    n_train = int(n_total * 0.8)

    X_past_train = X_past_train_full[:n_train]
    X_future_train = X_future_train_full[:n_train]
    y_residual_train = y_residual_train_full[:n_train]
    baseline_train = baseline_train_full[:n_train]
    y_actual_train = y_actual_train_full[:n_train]

    X_past_val = X_past_train_full[n_train:]
    X_future_val = X_future_train_full[n_train:]
    y_residual_val = y_residual_train_full[n_train:]
    baseline_val = baseline_train_full[n_train:]
    y_actual_val = y_actual_train_full[n_train:]

    print("\n--- Split Shapes ---")
    print(f"Train samples: {len(X_past_train)}")
    print(f"Val samples  : {len(X_past_val)}")
    print(f"Test samples : {len(X_past_test)}")

    # 3. Convert arrays to PyTorch tensors
    X_past_train_t = torch.tensor(X_past_train, dtype=torch.float32)
    X_future_train_t = torch.tensor(X_future_train, dtype=torch.float32)
    y_residual_train_t = torch.tensor(y_residual_train, dtype=torch.float32)

    X_past_val_t = torch.tensor(X_past_val, dtype=torch.float32)
    X_future_val_t = torch.tensor(X_future_val, dtype=torch.float32)
    y_residual_val_t = torch.tensor(y_residual_val, dtype=torch.float32)

    X_past_test_t = torch.tensor(X_past_test, dtype=torch.float32)
    X_future_test_t = torch.tensor(X_future_test, dtype=torch.float32)
    y_residual_test_t = torch.tensor(y_residual_test, dtype=torch.float32)

    # Dataloaders
    batch_size = 16
    train_loader = DataLoader(TensorDataset(X_past_train_t, X_future_train_t, y_residual_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_past_val_t, X_future_val_t, y_residual_val_t),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_past_test_t, X_future_test_t, y_residual_test_t),
                             batch_size=batch_size, shuffle=False)

    # 5. Build model
    past_input_size = X_past_train.shape[2]
    future_input_size = X_future_train.shape[2]
    forecast_days = y_residual_train.shape[1]

    model = ResidualFutureAwareLSTM(
        past_input_size=past_input_size,
        future_input_size=future_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        forecast_days=forecast_days
    )

    print("\n--- Model Architecture ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 6. Loss and Optimiser
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    max_epochs = 300
    early_stopping_patience = 20
    best_val_loss = float('inf')
    best_model_wts = None
    patience_counter = 0
    stopped_epoch = max_epochs

    train_loss_history = []
    val_loss_history   = []

    print(f"\nTraining Residual model for max {max_epochs} epochs...")
    print("-" * 65)

    # 7. Training Loop with early stopping
    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        running_train_loss = 0.0
        for batch_x_past, batch_x_future, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x_past, batch_x_future)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x_past.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Val
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x_past, batch_x_future, batch_y in val_loader:
                outputs = model(batch_x_past, batch_x_future)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_x_past.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)

        # Monitor validation loss improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Print loss progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{max_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Best Val: {best_val_loss:.6f}")

        # Check early stopping patience
        if patience_counter >= early_stopping_patience:
            stopped_epoch = epoch
            print(f"\n[Early Stopping] Triggered at epoch {stopped_epoch} (patience={early_stopping_patience}).")
            break

    print("-" * 65)

    # Reload best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print("Restored best validation model weights.")

    # 8. Evaluation on test set
    model.eval()
    all_pred_residuals = []
    with torch.no_grad():
        for batch_x_past, batch_x_future, _ in test_loader:
            outputs = model(batch_x_past, batch_x_future)
            all_pred_residuals.append(outputs.numpy())

    all_pred_residuals = np.concatenate(all_pred_residuals, axis=0)

    # Inverse transform predictions back to real residual counts
    pred_residuals_real = residual_target_scaler.inverse_transform(
        all_pred_residuals.reshape(-1, 1)
    ).reshape(-1, forecast_days)

    # Reconstruct final predictions: final_prediction = baseline + residual_correction
    final_predictions = baseline_test + pred_residuals_real

    # 9. Calculate model metrics
    mae_model, rmse_model, mape_model = calculate_metrics(y_actual_test, final_predictions)

    # 10. Calculate metrics for weekly seasonal baseline alone
    mae_baseline, rmse_baseline, mape_baseline = calculate_metrics(y_actual_test, baseline_test)

    # Compare differences
    mae_imp  = mae_baseline - mae_model
    rmse_imp = rmse_baseline - rmse_model
    mape_imp = mape_baseline - mape_model

    # 11. Print all metrics
    print(f"\nStopped at epoch          : {stopped_epoch}")
    print(f"Best Validation Loss      : {best_val_loss:.6f}")
    
    print("\n--- Weekly Seasonal Baseline Alone ---")
    print(f"MAE  : {mae_baseline:.2f} units")
    print(f"RMSE : {rmse_baseline:.2f} units")
    print(f"MAPE : {mape_baseline:.2f}%")

    print("\n--- LSTM v4 Residual Forecasting Model ---")
    print(f"MAE  : {mae_model:.2f} units")
    print(f"RMSE : {rmse_model:.2f} units")
    print(f"MAPE : {mape_model:.2f}%")

    print("\n--- Performance Improvement over Heuristic Baseline ---")
    print(f"MAE Improvement  : {mae_imp:+.2f} units ({'IMPROVEMENT' if mae_imp > 0 else 'DEGRADATION'})")
    print(f"RMSE Improvement : {rmse_imp:+.2f} units ({'IMPROVEMENT' if rmse_imp > 0 else 'DEGRADATION'})")
    print(f"MAPE Improvement : {mape_imp:+.2f}% ({'IMPROVEMENT' if mape_imp > 0 else 'DEGRADATION'})")

    sample_idx = 0
    print("\n--- Forecast Comparison on First Test Sample ---")
    print("Actual Next 7 Days         :", np.round(y_actual_test[sample_idx], 1))
    print("Weekly Baseline Next 7 Days:", np.round(baseline_test[sample_idx], 1))
    print("LSTM v4 Predict Next 7 Days:", np.round(final_predictions[sample_idx], 1))

    # 12. Save model
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/lstm_v4_residual_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # 13. Save loss curves chart
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', color='royalblue')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='tomato')
    plt.axvline(x=stopped_epoch, color='grey', linestyle='--', label=f'Early Stop ({stopped_epoch})')
    plt.title('LSTM v4 Residual Model: Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()

    # Save comparison chart
    chart_path = 'reports/lstm_v4_residual_prediction_vs_actual.png'
    days = range(1, forecast_days + 1)
    
    plt.figure(figsize=(9, 5))
    plt.plot(days, y_actual_test[sample_idx], marker='o', label='Actual Sales', color='royalblue')
    plt.plot(days, baseline_test[sample_idx], marker='s', label='Weekly Baseline', color='grey', linestyle=':')
    plt.plot(days, final_predictions[sample_idx], marker='x', label='LSTM v4 Residual Prediction', color='tomato', linestyle='--')
    plt.title('LSTM v4 Residual model vs Weekly Baseline vs Actuals')
    plt.xlabel('Forecast Day')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"Chart saved to {chart_path}")

    # 14. Save metrics file
    metrics_path = 'reports/lstm_v4_residual_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("LSTM v4 Residual Forecasting Model Metrics\n")
        f.write("=" * 45 + "\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Stopped Epoch       : {stopped_epoch}\n")
        f.write("\nWeekly Baseline Metrics:\n")
        f.write(f"  MAE  : {mae_baseline:.2f} units\n")
        f.write(f"  RMSE : {rmse_baseline:.2f} units\n")
        f.write(f"  MAPE : {mape_baseline:.2f}%\n")
        f.write("\nLSTM v4 Residual Model Metrics:\n")
        f.write(f"  MAE  : {mae_model:.2f} units\n")
        f.write(f"  RMSE : {rmse_model:.2f} units\n")
        f.write(f"  MAPE : {mape_model:.2f}%\n")
        f.write("\nRelative Lift Over Baseline:\n")
        f.write(f"  MAE  : {mae_imp:+.2f} units ({'Improved' if mae_imp > 0 else 'Degraded'})\n")
        f.write(f"  RMSE : {rmse_imp:+.2f} units ({'Improved' if rmse_imp > 0 else 'Degraded'})\n")
        f.write(f"  MAPE : {mape_imp:+.2f}% ({'Improved' if mape_imp > 0 else 'Degraded'})\n")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
