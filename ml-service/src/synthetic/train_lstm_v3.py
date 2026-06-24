"""
train_lstm_v3.py
================
Phase 1.9 – Train LSTM v3: Seasonal + Future-Aware Architecture

New in v3 vs v2 (train_lstm_future.py):
  • Past encoder now sees 8 features (added is_weekend + rolling_7_day_average)
  • Future encoder now sees 7 features (added is_weekend, same_day_last_week_sales,
    rolling_7_day_average) instead of 4
  • Future encoder hidden size expanded from 16 → 32 (more capacity for richer features)
  • Prediction head expanded: (64+32→64→1) instead of (64+16→32→1)
  • Dropout added to future encoder + prediction head for better regularisation
  • 150 training epochs (was 120) to allow the larger head to converge

Learning goals:
  • Understand how two-stream (past + future) LSTMs work
  • See how lag features (same_day_last_week_sales) act as a seasonal prior
  • Observe how rolling averages smooth short-term noise for the model
  • Follow the full PyTorch training loop from tensors → metrics → saved artefacts
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend – safe for scripts
import matplotlib.pyplot as plt


# ===========================================================================
# MODEL DEFINITION
# ===========================================================================

class SeasonalFutureAwareLSTM(nn.Module):
    """
    A two-stream neural network that combines:
      Stream A – an LSTM that reads the last 30 days of historical data.
      Stream B – a linear encoder that reads 7 days of known future features.

    The two streams are fused (concatenated) and passed through a prediction
    head that outputs one sales value per forecast day.

    Why two streams?
    ----------------
    Demand forecasting requires two kinds of knowledge:
      1. What happened recently? (trends, promotions, weekend effects)
      2. What is scheduled to happen? (future price, promotion, day-of-week)

    Merging both kinds of information leads to much more accurate 7-day forecasts
    than using only the past.
    """

    def __init__(
        self,
        past_input_size,    # Number of features per past time step (8 in v3)
        future_input_size,  # Number of features per future time step (7 in v3)
        hidden_size=64,     # LSTM hidden state dimension
        num_layers=2,       # Stack two LSTM layers for deeper temporal modelling
        dropout=0.2,        # Dropout rate inside the stacked LSTM
        forecast_days=7     # Number of days to predict
    ):
        super(SeasonalFutureAwareLSTM, self).__init__()

        # Store forecast_days so the forward() method can use it
        self.forecast_days = forecast_days

        # -----------------------------------------------------------------
        # A. PAST ENCODER
        # -----------------------------------------------------------------
        # nn.LSTM reads the 30-day sequence and compresses it into a single
        # hidden vector that captures trends, weekly cycles, and promotions.
        #
        # Key parameters:
        #   batch_first=True  → input shape is (batch, seq_len, features)
        #   num_layers=2      → two stacked LSTM cells for richer representations
        #   dropout=0.2       → randomly zero 20% of connections between layers
        #                       to prevent co-adaptation (overfitting)
        # -----------------------------------------------------------------
        self.past_encoder = nn.LSTM(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # dropout only between layers
        )

        # -----------------------------------------------------------------
        # B. FUTURE FEATURE ENCODER
        # -----------------------------------------------------------------
        # We know several things about the future in advance: price, promotion,
        # day_of_week, month, is_weekend, same_day_last_week_sales, and
        # rolling_7_day_average.
        #
        # A simple Linear → ReLU → Dropout block projects each day's 7-feature
        # vector into a compact 32-dimensional representation.
        #
        # Why Linear and not another LSTM here?
        #   Future features are mostly independent per day (day_of_week on day 3
        #   doesn't "flow" into day 4 in a causal sense). A feedforward layer
        #   captures the per-day context without unnecessary complexity.
        # -----------------------------------------------------------------
        self.future_encoder = nn.Sequential(
            nn.Linear(future_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # -----------------------------------------------------------------
        # C. FUSION
        # -----------------------------------------------------------------
        # After encoding, past context is (batch, hidden_size=64) and the
        # encoded future is (batch, 7, 32).
        #
        # We repeat the past context 7 times so it can be compared/combined
        # with each of the 7 forecast days individually.
        #
        # Final fused tensor shape: (batch, 7, 64 + 32) = (batch, 7, 96)
        # -----------------------------------------------------------------
        # (No learnable parameters here; fusion happens in forward())

        # -----------------------------------------------------------------
        # D. PREDICTION HEAD
        # -----------------------------------------------------------------
        # The fused vector (64 + 32 = 96 dims) passes through two linear
        # layers to produce a single scalar per day.
        #
        # Architecture: 96 → 64 → ReLU → Dropout → 1
        #
        # Why a two-layer head?
        #   A single linear layer directly from 96 → 1 would be too shallow
        #   to learn the non-linear interactions between past context and
        #   future features. The intermediate 64-unit layer adds capacity.
        # -----------------------------------------------------------------
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x_past, x_future):
        """
        Forward pass through both streams and the prediction head.

        Parameters
        ----------
        x_past   : Tensor of shape (batch_size, 30, past_input_size)
        x_future : Tensor of shape (batch_size, 7,  future_input_size)

        Returns
        -------
        Tensor of shape (batch_size, 7) – one sales prediction per forecast day
        """

        # ---- Stream A: Process past 30 days through the LSTM ----
        # lstm_out shape: (batch_size, 30, hidden_size)
        # We discard the hidden/cell state tuple because we only need the
        # output at the final time step.
        lstm_out, _ = self.past_encoder(x_past)

        # Extract the last time step's output as the "past context" vector.
        # This single vector summarises everything the LSTM learned from
        # the 30-day history.
        # Shape: (batch_size, hidden_size=64)
        past_context = lstm_out[:, -1, :]

        # ---- Stream B: Encode each of the 7 future days ----
        # The Sequential (Linear→ReLU→Dropout) is applied independently to
        # each day's feature vector via broadcasting.
        # Shape: (batch_size, 7, 32)
        encoded_future = self.future_encoder(x_future)

        # ---- Fusion (C): Repeat past context for every forecast day ----
        # unsqueeze(1) adds the time dimension: (batch, 1, 64)
        # repeat(1, 7, 1) copies it 7 times:   (batch, 7, 64)
        past_context_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1)

        # Concatenate along the feature dimension: (batch, 7, 64+32=96)
        combined = torch.cat((past_context_repeated, encoded_future), dim=2)

        # ---- Prediction head (D) ----
        # Shape after head: (batch, 7, 1)
        out = self.prediction_head(combined)

        # Remove the trailing size-1 dimension → (batch, 7)
        return out.squeeze(2)


# ===========================================================================
# METRIC HELPERS
# ===========================================================================

def calculate_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE, and MAPE on raw (inverse-transformed) unit values.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples, forecast_days)
    y_pred : np.ndarray of shape (n_samples, forecast_days)

    Returns
    -------
    (mae, rmse, mape) – all floats
    """
    # Mean Absolute Error: average number of units we are off by
    mae = np.mean(np.abs(y_pred - y_true))

    # Root Mean Squared Error: penalises large errors more heavily
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))

    # Mean Absolute Percentage Error: how far off we are, as a percentage
    # Add 1e-8 to the denominator to avoid division-by-zero on days with 0 sales
    mape = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100

    return mae, rmse, mape


# ===========================================================================
# MAIN TRAINING ROUTINE
# ===========================================================================

def main():

    # -----------------------------------------------------------------------
    # 1. LOAD PREPROCESSED ARRAYS
    # -----------------------------------------------------------------------
    # These arrays were created by preprocess_future_sequences_v3.py.
    # They are already:
    #   • Scaled to [0, 1] using MinMaxScaler
    #   • Split chronologically (first 80% train / last 20% test)
    # -----------------------------------------------------------------------
    data_path = 'data/processed/synthetic_future_sequences_v3.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Please run: python src/preprocess_future_sequences_v3.py")
        return

    print("Loading v3 sequences (seasonal + future-aware)...")
    with np.load(data_path) as data:
        X_past_train_np   = data['X_past_train']
        X_future_train_np = data['X_future_train']
        y_train_np        = data['y_train']
        X_past_test_np    = data['X_past_test']
        X_future_test_np  = data['X_future_test']
        y_test_np         = data['y_test']

    # 12. Print shapes so the student can verify dimensionality
    print("\n--- Data Shapes ---")
    print(f"X_past_train shape  : {X_past_train_np.shape}")   # (257, 30, 8)
    print(f"X_future_train shape: {X_future_train_np.shape}") # (257,  7, 7)
    print(f"y_train shape       : {y_train_np.shape}")        # (257,  7)
    print(f"X_past_test shape   : {X_past_test_np.shape}")    # ( 65, 30, 8)
    print(f"X_future_test shape : {X_future_test_np.shape}")  # ( 65,  7, 7)
    print(f"y_test shape        : {y_test_np.shape}")         # ( 65,  7)

    # -----------------------------------------------------------------------
    # 2. LOAD TARGET SCALER
    # -----------------------------------------------------------------------
    # The scaler was fitted on units_sold during preprocessing.
    # We need it later to inverse-transform predictions back to real unit counts.
    # -----------------------------------------------------------------------
    scaler_path = 'models/v3_target_scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found.")
        return
    target_scaler = joblib.load(scaler_path)
    print(f"\nLoaded target scaler from {scaler_path}")

    # -----------------------------------------------------------------------
    # 3. CONVERT NUMPY ARRAYS TO PYTORCH TENSORS
    # -----------------------------------------------------------------------
    # PyTorch operates on Tensor objects, not raw NumPy arrays.
    # dtype=torch.float32 is the standard precision for neural network weights.
    # -----------------------------------------------------------------------
    X_past_train   = torch.tensor(X_past_train_np,   dtype=torch.float32)
    X_future_train = torch.tensor(X_future_train_np, dtype=torch.float32)
    y_train        = torch.tensor(y_train_np,         dtype=torch.float32)

    X_past_test    = torch.tensor(X_past_test_np,    dtype=torch.float32)
    X_future_test  = torch.tensor(X_future_test_np,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test_np,          dtype=torch.float32)

    # -----------------------------------------------------------------------
    # 4. CREATE TENSORDATASET
    # -----------------------------------------------------------------------
    # TensorDataset bundles the three input arrays together so DataLoader
    # can yield aligned mini-batches (same indices across all three tensors).
    # -----------------------------------------------------------------------
    train_dataset = TensorDataset(X_past_train, X_future_train, y_train)
    test_dataset  = TensorDataset(X_past_test,  X_future_test,  y_test_tensor)

    # -----------------------------------------------------------------------
    # 5. CREATE DATALOADER
    # -----------------------------------------------------------------------
    # DataLoader handles:
    #   • Batching  – groups samples into mini-batches of size 16
    #   • Shuffling – randomises order each epoch to prevent the model from
    #                 memorising the exact sequence of training examples
    # Test loader does NOT shuffle – we want to compare predictions in order.
    # -----------------------------------------------------------------------
    batch_size   = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # -----------------------------------------------------------------------
    # 6. BUILD MODEL
    # -----------------------------------------------------------------------
    # Derive input sizes from the data rather than hard-coding them.
    # This makes the script robust if the preprocessing changes.
    # -----------------------------------------------------------------------
    past_input_size   = X_past_train_np.shape[2]    # 8
    future_input_size = X_future_train_np.shape[2]  # 7
    forecast_days     = y_train_np.shape[1]          # 7

    model = SeasonalFutureAwareLSTM(
        past_input_size=past_input_size,
        future_input_size=future_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        forecast_days=forecast_days
    )

    # 12. Print model architecture
    print("\n--- Model Architecture ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # 7. LOSS FUNCTION & OPTIMISER
    # -----------------------------------------------------------------------
    # MSELoss (Mean Squared Error) squares the difference between prediction
    # and target, then averages. Squaring penalises large errors more heavily,
    # pushing the model to avoid catastrophic misses.
    #
    # Adam (Adaptive Moment Estimation) adjusts the learning rate for each
    # parameter individually. It combines momentum and RMSProp, making it
    # faster and more stable than plain Stochastic Gradient Descent (SGD).
    # -----------------------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs    = 150

    print(f"\nStarting training for {epochs} epochs (batch_size={batch_size}, lr=0.001)...")
    print("-" * 55)

    # -----------------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        # Put model in training mode (enables dropout layers)
        model.train()
        running_loss = 0.0

        for batch_x_past, batch_x_future, batch_y in train_loader:
            # Zero out gradients from the previous step.
            # PyTorch accumulates gradients by default; we must reset them.
            optimizer.zero_grad()

            # Forward pass – compute predictions for this mini-batch
            outputs = model(batch_x_past, batch_x_future)

            # Compute loss between predictions and ground truth
            loss = criterion(outputs, batch_y)

            # Backward pass – compute gradients via backpropagation
            loss.backward()

            # Update model weights using the computed gradients
            optimizer.step()

            # Accumulate loss weighted by actual batch size
            # (last batch may be smaller than batch_size)
            running_loss += loss.item() * batch_x_past.size(0)

        # Average loss across all training samples this epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        # 8. Print training loss every 10 epochs (and on the first epoch)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{epochs} | Train Loss (MSE): {epoch_loss:.6f}")

    print("-" * 55)
    print("Training complete.\n")

    # -----------------------------------------------------------------------
    # 9. EVALUATE ON TEST SET
    # -----------------------------------------------------------------------
    # torch.no_grad() disables gradient tracking during inference.
    # This saves memory and speeds up evaluation (we aren't learning here).
    # model.eval() disables dropout, so predictions are deterministic.
    # -----------------------------------------------------------------------
    model.eval()
    test_loss      = 0.0
    all_predictions = []
    all_actuals     = []

    with torch.no_grad():
        for batch_x_past, batch_x_future, batch_y in test_loader:
            outputs   = model(batch_x_past, batch_x_future)
            loss      = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x_past.size(0)

            # Collect scaled predictions and actuals for metric computation
            all_predictions.append(outputs.numpy())
            all_actuals.append(batch_y.numpy())

    # Average test MSE across all test samples
    test_loss /= len(test_loader.dataset)

    # Stack list of arrays into single 2-D arrays: (n_test_samples, 7)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_actuals     = np.concatenate(all_actuals,     axis=0)

    # -----------------------------------------------------------------------
    # 10. INVERSE TRANSFORM
    # -----------------------------------------------------------------------
    # The model outputs values in the scaled [0, 1] range.
    # We reshape to (n*7, 1) because the scaler expects a 2-D array,
    # then reshape back to (n, 7) after transforming.
    # -----------------------------------------------------------------------
    n_test       = all_actuals.shape[0]
    y_test_real  = target_scaler.inverse_transform(
                       all_actuals.reshape(-1, 1)
                   ).reshape(n_test, forecast_days)

    predictions_real = target_scaler.inverse_transform(
                           all_predictions.reshape(-1, 1)
                       ).reshape(n_test, forecast_days)

    # -----------------------------------------------------------------------
    # 11. CALCULATE METRICS
    # -----------------------------------------------------------------------
    mae, rmse, mape = calculate_metrics(y_test_real, predictions_real)

    # -----------------------------------------------------------------------
    # 12. PRINT ALL RESULTS
    # -----------------------------------------------------------------------
    print(f"Final Test Loss (MSE, scaled): {test_loss:.6f}")
    print("\n--- Overall Test Metrics (Seasonal + Future-Aware LSTM v3) ---")
    print(f"MAE  : {mae:.2f}  units  (average absolute error per day)")
    print(f"RMSE : {rmse:.2f}  units  (penalises large errors more)")
    print(f"MAPE : {mape:.2f}% (relative percentage error)")

    # Show one sample prediction so we can visually sanity-check the model
    sample_idx = 0
    print("\n--- Prediction vs Actual on First Test Sample ---")
    print("Actual    Next 7 Days (units):", np.round(y_test_real[sample_idx],     1))
    print("Predicted Next 7 Days (units):", np.round(predictions_real[sample_idx], 1))

    # -----------------------------------------------------------------------
    # 13. SAVE MODEL WEIGHTS
    # -----------------------------------------------------------------------
    # torch.save saves only the state_dict (learned weights), not the full
    # model object. This is the recommended PyTorch convention: it's portable
    # and not tied to the Python class path.
    # To reload:
    #   model = SeasonalFutureAwareLSTM(...)
    #   model.load_state_dict(torch.load('models/lstm_v3_seasonal_model.pth'))
    # -----------------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lstm_v3_seasonal_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # -----------------------------------------------------------------------
    # 14. SAVE PREDICTION CHART
    # -----------------------------------------------------------------------
    os.makedirs('reports', exist_ok=True)
    chart_path = 'reports/lstm_v3_prediction_vs_actual.png'
    days = range(1, forecast_days + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('LSTM v3 (Seasonal + Future-Aware): Prediction vs Actual', fontsize=14)

    # Left panel: first test sample
    axes[0].plot(days, y_test_real[sample_idx],     marker='o', label='Actual',    color='royalblue')
    axes[0].plot(days, predictions_real[sample_idx], marker='x', label='Predicted', color='tomato', linestyle='--')
    axes[0].set_title('First Test Sample')
    axes[0].set_xlabel('Forecast Day')
    axes[0].set_ylabel('Units Sold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Right panel: averaged across ALL test samples
    mean_actual    = y_test_real.mean(axis=0)
    mean_predicted = predictions_real.mean(axis=0)
    axes[1].plot(days, mean_actual,    marker='o', label='Actual (avg)',    color='royalblue')
    axes[1].plot(days, mean_predicted, marker='x', label='Predicted (avg)', color='tomato', linestyle='--')
    axes[1].set_title('Average Across All Test Samples')
    axes[1].set_xlabel('Forecast Day')
    axes[1].set_ylabel('Units Sold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"Chart saved to {chart_path}")

    # -----------------------------------------------------------------------
    # 15. SAVE METRICS TO TEXT FILE
    # -----------------------------------------------------------------------
    metrics_path = 'reports/lstm_v3_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("LSTM v3 – Seasonal + Future-Aware Model\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test MSE (scaled) : {test_loss:.6f}\n")
        f.write(f"MAE               : {mae:.2f} units\n")
        f.write(f"RMSE              : {rmse:.2f} units\n")
        f.write(f"MAPE              : {mape:.2f}%\n")
        f.write("\nModel Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"past_input_size   : {past_input_size}\n")
        f.write(f"future_input_size : {future_input_size}\n")
        f.write(f"hidden_size       : 64\n")
        f.write(f"num_layers        : 2\n")
        f.write(f"dropout           : 0.2\n")
        f.write(f"forecast_days     : {forecast_days}\n")
        f.write(f"epochs            : {epochs}\n")
        f.write(f"batch_size        : {batch_size}\n")
        f.write(f"learning_rate     : 0.001\n")
        f.write(f"total_params      : {total_params:,}\n")
    print(f"Metrics saved to {metrics_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
