"""
train_lstm_v3_regularized.py
=============================
Phase 1.10 – Regularized LSTM v3: Fighting Overfitting

Why this script exists
----------------------
In Phase 1.9, LSTM v3 achieved a very low training loss (0.002) but a high
test MSE (0.046). That large gap is the classic symptom of OVERFITTING:
the model memorised the training examples so well that it stopped generalising
to unseen data.

This script adds three regularisation techniques to close that gap:

  1. Validation split  – holds out 20% of the training data to monitor
                         how well the model generalises *during* training.
  2. Early stopping    – halts training the moment validation loss stops
                         improving, before the model has a chance to
                         memorise the training set.
  3. Weight decay      – adds an L2 penalty to all model weights inside
                         the Adam optimiser, preventing any single weight
                         from growing unreasonably large.

Student learning goals
----------------------
After reading this script you should understand:
  • The difference between training loss and validation loss.
  • Why a model that scores perfectly on training data can still fail.
  • How early stopping works as an automatic regulariser.
  • What weight decay (L2 regularisation) does mathematically.
  • How to save the *best* checkpoint and reload it for final evaluation.
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
matplotlib.use('Agg')          # Non-interactive backend – safe in scripts
import matplotlib.pyplot as plt


# ===========================================================================
# MODEL DEFINITION  (identical architecture to v3, but with stronger dropout)
# ===========================================================================

class SeasonalFutureAwareLSTM(nn.Module):
    """
    Two-stream network: LSTM for past history + Linear encoder for known future.

    Changes vs. Phase 1.9:
      • LSTM dropout:           0.2  → 0.3
      • Future encoder dropout: 0.1  → 0.2
      • Prediction head dropout:0.1  → 0.2

    Higher dropout means that during each training forward pass, 20-30% of
    neurons are randomly turned off. This forces the remaining neurons to
    learn redundant, robust representations rather than memorising specific
    training samples.
    """

    def __init__(
        self,
        past_input_size,    # Number of features per past time step   (8 in v3)
        future_input_size,  # Number of features per future time step  (7 in v3)
        hidden_size=64,
        num_layers=2,
        dropout=0.3,        # ← increased from 0.2
        forecast_days=7
    ):
        super(SeasonalFutureAwareLSTM, self).__init__()
        self.forecast_days = forecast_days

        # -----------------------------------------------------------------
        # A. PAST ENCODER
        # -----------------------------------------------------------------
        # A stacked (2-layer) LSTM reads the 30-day history and compresses
        # it into a single hidden vector representing the "past context".
        #
        # dropout=0.3 applies between the two LSTM layers. During training,
        # 30% of activations passing from layer 1 → layer 2 are zeroed out
        # randomly each forward pass. This prevents the layers from learning
        # to rely on specific neurons and encourages diverse representations.
        # -----------------------------------------------------------------
        self.past_encoder = nn.LSTM(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # -----------------------------------------------------------------
        # B. FUTURE FEATURE ENCODER
        # -----------------------------------------------------------------
        # Encodes the 7 known future days (price, promotion, day_of_week,
        # month, is_weekend, same_day_last_week_sales, rolling_7_day_avg)
        # into compact 32-dimensional per-day vectors.
        #
        # Dropout(0.2) here means 20% of the encoded future values are
        # randomly masked during training, forcing the prediction head to
        # not rely on any single future feature too heavily.
        # -----------------------------------------------------------------
        self.future_encoder = nn.Sequential(
            nn.Linear(future_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2)     # ← increased from 0.1
        )

        # -----------------------------------------------------------------
        # C. FUSION  (no learnable parameters – happens in forward())
        # -----------------------------------------------------------------
        # The past context vector (shape: batch × 64) is repeated 7 times
        # to match the 7 forecast days, then concatenated with the encoded
        # future features (batch × 7 × 32).
        # Result: (batch × 7 × 96)
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # D. PREDICTION HEAD
        # -----------------------------------------------------------------
        # Takes the fused 96-dim vector and predicts one sales value per day.
        #
        # Dropout(0.2) between the two linear layers adds a final layer of
        # regularisation right before the output.
        # -----------------------------------------------------------------
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),    # ← increased from 0.1
            nn.Linear(64, 1)
        )

    def forward(self, x_past, x_future):
        """
        x_past   : (batch, 30, past_input_size)
        x_future : (batch,  7, future_input_size)
        returns  : (batch,  7)  — one prediction per forecast day
        """
        # Stream A: compress 30-day history into a single context vector
        lstm_out, _ = self.past_encoder(x_past)
        past_context = lstm_out[:, -1, :]                            # (batch, 64)

        # Stream B: encode each future day independently
        encoded_future = self.future_encoder(x_future)               # (batch, 7, 32)

        # Fusion: broadcast past context across all 7 forecast days
        past_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1)  # (batch, 7, 64)
        combined = torch.cat((past_repeated, encoded_future), dim=2)  # (batch, 7, 96)

        # Predict
        out = self.prediction_head(combined)                          # (batch, 7, 1)
        return out.squeeze(2)                                         # (batch, 7)


# ===========================================================================
# METRIC HELPERS
# ===========================================================================

def calculate_metrics(y_true, y_pred):
    """MAE, RMSE, MAPE on inverse-transformed (real-unit) arrays."""
    mae  = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
    mape = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100
    return mae, rmse, mape


# ===========================================================================
# MAIN
# ===========================================================================

def main():

    # -----------------------------------------------------------------------
    # 1. LOAD PREPROCESSED ARRAYS
    # -----------------------------------------------------------------------
    data_path = 'data/processed/synthetic_future_sequences_v3.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Run: python src/preprocess_future_sequences_v3.py")
        return

    print("Loading v3 sequences...")
    with np.load(data_path) as data:
        X_past_train_full_np   = data['X_past_train']
        X_future_train_full_np = data['X_future_train']
        y_train_full_np        = data['y_train']
        X_past_test_np         = data['X_past_test']
        X_future_test_np       = data['X_future_test']
        y_test_np              = data['y_test']

    # -----------------------------------------------------------------------
    # 2. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
    # -----------------------------------------------------------------------
    # WHY CHRONOLOGICAL?
    # ------------------
    # Time-series data has a strict arrow of time: the past causes the future,
    # not the other way around. If we randomly shuffled and split the data,
    # some future samples would leak into the training set. The model would
    # then "see" future patterns during training and produce unrealistically
    # optimistic validation scores — a form of data leakage.
    #
    # Correct approach: use the *earliest* 80% of sequences for training
    # and the *latest* 20% for validation. This mirrors how the model will
    # be used in production (always trained on the past, evaluated on the future).
    #
    # WHY DO WE NEED VALIDATION AT ALL?
    # -----------------------------------
    # The test set is held out completely until the very end. We cannot use
    # it to tune hyperparameters or stop training early — that would leak
    # information about the test set into training decisions.
    # The validation set acts as a proxy for the test set:
    #   - Training loss tells us whether the model is learning at all.
    #   - Validation loss tells us whether it is generalising.
    #   - If training loss ↓ but validation loss ↑, the model is overfitting.
    # -----------------------------------------------------------------------
    n_total  = len(X_past_train_full_np)
    n_train  = int(n_total * 0.80)   # first 80% → training
    # remaining 20% → validation (latest samples, closer to the test period)

    X_past_train_np   = X_past_train_full_np[:n_train]
    X_future_train_np = X_future_train_full_np[:n_train]
    y_train_np        = y_train_full_np[:n_train]

    X_past_val_np   = X_past_train_full_np[n_train:]
    X_future_val_np = X_future_train_full_np[n_train:]
    y_val_np        = y_train_full_np[n_train:]

    print("\n--- Data Shapes ---")
    print(f"X_past_train shape  : {X_past_train_np.shape}")
    print(f"X_future_train shape: {X_future_train_np.shape}")
    print(f"y_train shape       : {y_train_np.shape}")
    print(f"X_past_val shape    : {X_past_val_np.shape}")
    print(f"X_future_val shape  : {X_future_val_np.shape}")
    print(f"y_val shape         : {y_val_np.shape}")
    print(f"X_past_test shape   : {X_past_test_np.shape}")
    print(f"X_future_test shape : {X_future_test_np.shape}")
    print(f"y_test shape        : {y_test_np.shape}")

    # -----------------------------------------------------------------------
    # 3. LOAD TARGET SCALER
    # -----------------------------------------------------------------------
    scaler_path = 'models/v3_target_scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found.")
        return
    target_scaler = joblib.load(scaler_path)
    print(f"\nLoaded target scaler from {scaler_path}")

    # -----------------------------------------------------------------------
    # 4. CONVERT TO PYTORCH TENSORS
    # -----------------------------------------------------------------------
    X_past_train   = torch.tensor(X_past_train_np,   dtype=torch.float32)
    X_future_train = torch.tensor(X_future_train_np, dtype=torch.float32)
    y_train        = torch.tensor(y_train_np,         dtype=torch.float32)

    X_past_val   = torch.tensor(X_past_val_np,   dtype=torch.float32)
    X_future_val = torch.tensor(X_future_val_np, dtype=torch.float32)
    y_val        = torch.tensor(y_val_np,         dtype=torch.float32)

    X_past_test   = torch.tensor(X_past_test_np,   dtype=torch.float32)
    X_future_test = torch.tensor(X_future_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np,         dtype=torch.float32)

    # DataLoaders
    # Shuffle=True on training: randomising the order each epoch prevents
    # the optimiser from learning spurious correlations tied to batch order.
    # Shuffle=False on val/test: we want deterministic, ordered evaluation.
    batch_size   = 16
    train_loader = DataLoader(TensorDataset(X_past_train, X_future_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_past_val,   X_future_val,   y_val),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_past_test,  X_future_test,  y_test_tensor),
                              batch_size=batch_size, shuffle=False)

    # -----------------------------------------------------------------------
    # 5. BUILD MODEL
    # -----------------------------------------------------------------------
    past_input_size   = X_past_train_np.shape[2]    # 8
    future_input_size = X_future_train_np.shape[2]  # 7
    forecast_days     = y_train_np.shape[1]          # 7

    model = SeasonalFutureAwareLSTM(
        past_input_size=past_input_size,
        future_input_size=future_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,            # ← stronger than v3
        forecast_days=forecast_days
    )

    print("\n--- Model Architecture ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # 6. LOSS, OPTIMISER, AND HYPERPARAMETERS
    # -----------------------------------------------------------------------

    criterion = nn.MSELoss()

    # WEIGHT DECAY (L2 regularisation)
    # ----------------------------------
    # What it does mathematically:
    #   At every update step, Adam not only moves weights to reduce training
    #   loss, but also applies a small penalty proportional to the weight
    #   magnitude:
    #
    #     new_weight = old_weight × (1 − lr × weight_decay) − lr × gradient
    #
    # Why this prevents overfitting:
    #   Large weights allow the model to memorise very specific patterns
    #   (high-frequency noise) in the training data. By penalising large
    #   weights, we force the model to find solutions that work with smaller,
    #   more general weight values. It's like Occam's Razor applied to
    #   neural networks: prefer simpler solutions.
    #
    #   weight_decay=1e-4 is a mild penalty; it gently discourages large
    #   weights without preventing the model from learning.
    # -----------------------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4       # L2 regularisation coefficient
    )

    max_epochs              = 300
    early_stopping_patience = 20  # stop if val loss doesn't improve for 20 epochs

    # -----------------------------------------------------------------------
    # 7. TRAINING LOOP WITH VALIDATION AND EARLY STOPPING
    # -----------------------------------------------------------------------

    # EARLY STOPPING EXPLAINED
    # -------------------------
    # Idea: we keep a "patience counter". Every time the validation loss
    # reaches a new minimum we reset the counter to 0. Every time it does
    # NOT improve, we increment the counter. Once the counter exceeds
    # `patience`, we stop training and restore the best weights.
    #
    # Why this works:
    #   Without early stopping, training always continues until max_epochs.
    #   After the validation loss starts rising (the model is overfitting),
    #   every extra epoch makes the generalisation worse. Early stopping
    #   freezes training at the exact epoch where the model generalised best,
    #   even if it could still lower the training loss further.
    # -----------------------------------------------------------------------

    best_val_loss   = float('inf')      # track the lowest validation loss seen
    patience_counter = 0                # how many consecutive epochs without improvement
    best_model_wts  = None              # deep copy of the weights at the best epoch
    stopped_epoch   = max_epochs       # will be updated if we stop early

    train_loss_history = []
    val_loss_history   = []

    print(f"\nStarting training (max {max_epochs} epochs, patience={early_stopping_patience})...")
    print("-" * 65)

    for epoch in range(1, max_epochs + 1):

        # --- TRAINING PHASE ---
        # model.train() enables dropout (neurons randomly turned off).
        # This is critical: dropout is only active during training, not eval.
        model.train()
        running_train_loss = 0.0

        for batch_x_past, batch_x_future, batch_y in train_loader:
            optimizer.zero_grad()                          # clear stale gradients
            outputs = model(batch_x_past, batch_x_future) # forward pass
            loss    = criterion(outputs, batch_y)          # compute MSE loss
            loss.backward()                                # backpropagate gradients
            optimizer.step()                               # update weights
            running_train_loss += loss.item() * batch_x_past.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- VALIDATION PHASE ---
        # model.eval() disables dropout → all neurons are active for
        # deterministic, stable loss measurement.
        # torch.no_grad() disables gradient computation (saves memory + time).
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch_x_past, batch_x_future, batch_y in val_loader:
                outputs = model(batch_x_past, batch_x_future)
                loss    = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_x_past.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # Record history for the loss-curve chart
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)

        # --- EARLY STOPPING CHECK ---
        if epoch_val_loss < best_val_loss:
            # New best found: save weights and reset patience counter
            best_val_loss  = epoch_val_loss
            patience_counter = 0
            # copy.deepcopy ensures we save a *separate* snapshot,
            # not a reference to the model (which keeps changing).
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        # 8. Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3}/{max_epochs} | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Val Loss: {epoch_val_loss:.6f} | "
                f"Best Val: {best_val_loss:.6f}"
            )

        # Trigger early stopping once patience is exhausted
        if patience_counter >= early_stopping_patience:
            stopped_epoch = epoch
            print(f"\n[Early Stopping] No val improvement for {early_stopping_patience} epochs.")
            print(f"Stopped at epoch {stopped_epoch}. Best val loss: {best_val_loss:.6f}")
            break

    else:
        # Loop finished naturally without early stopping
        stopped_epoch = max_epochs
        print(f"\nCompleted all {max_epochs} epochs. Best val loss: {best_val_loss:.6f}")

    print("-" * 65)

    # -----------------------------------------------------------------------
    # RELOAD BEST MODEL WEIGHTS
    # -----------------------------------------------------------------------
    # After the loop, `model` contains the weights from the LAST epoch,
    # which is NOT necessarily the best. We reload the checkpoint we saved
    # when val loss was lowest.
    #
    # This is the key payoff of early stopping + checkpointing:
    # we get the generalisable version, not the overfit version.
    # -----------------------------------------------------------------------
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print("\nRestored best model weights for final evaluation.")

    # -----------------------------------------------------------------------
    # 9. EVALUATE ON TEST SET
    # -----------------------------------------------------------------------
    model.eval()
    test_loss       = 0.0
    all_predictions = []
    all_actuals     = []

    with torch.no_grad():
        for batch_x_past, batch_x_future, batch_y in test_loader:
            outputs   = model(batch_x_past, batch_x_future)
            loss      = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x_past.size(0)
            all_predictions.append(outputs.numpy())
            all_actuals.append(batch_y.numpy())

    test_loss /= len(test_loader.dataset)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_actuals     = np.concatenate(all_actuals,     axis=0)

    # -----------------------------------------------------------------------
    # 10. INVERSE TRANSFORM
    # -----------------------------------------------------------------------
    n_test = all_actuals.shape[0]
    y_test_real      = target_scaler.inverse_transform(
                           all_actuals.reshape(-1, 1)
                       ).reshape(n_test, forecast_days)

    predictions_real = target_scaler.inverse_transform(
                           all_predictions.reshape(-1, 1)
                       ).reshape(n_test, forecast_days)

    # -----------------------------------------------------------------------
    # 11 & 12. CALCULATE AND PRINT METRICS
    # -----------------------------------------------------------------------
    mae, rmse, mape = calculate_metrics(y_test_real, predictions_real)

    print(f"\nBest Validation Loss      : {best_val_loss:.6f}")
    print(f"Stopped at epoch          : {stopped_epoch}")
    print(f"Final Test Loss (MSE)     : {test_loss:.6f}")
    print("\n--- Overall Test Metrics (Regularized LSTM v3) ---")
    print(f"MAE  : {mae:.2f}  units")
    print(f"RMSE : {rmse:.2f}  units")
    print(f"MAPE : {mape:.2f}%")

    sample_idx = 0
    print("\n--- Prediction vs Actual on First Test Sample ---")
    print("Actual    Next 7 Days:", np.round(y_test_real[sample_idx],     1))
    print("Predicted Next 7 Days:", np.round(predictions_real[sample_idx], 1))

    # -----------------------------------------------------------------------
    # 13. SAVE BEST MODEL
    # -----------------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lstm_v3_regularized_best_model.pth'
    torch.save(best_model_wts if best_model_wts is not None else model.state_dict(),
               model_path)
    print(f"\nModel saved to {model_path}")

    os.makedirs('reports', exist_ok=True)

    # -----------------------------------------------------------------------
    # 14. SAVE LOSS CURVE
    # -----------------------------------------------------------------------
    # This chart is the most important diagnostic plot in this phase.
    # A healthy training run shows:
    #   • Training loss steadily decreasing.
    #   • Validation loss decreasing in parallel, then flattening.
    # An overfit run shows:
    #   • Training loss keeps decreasing past the point where val loss rises.
    # The early-stopping vertical line shows exactly where we stopped.
    # -----------------------------------------------------------------------
    loss_curve_path = 'reports/lstm_v3_regularized_loss_curve.png'
    epochs_range    = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_loss_history, label='Training Loss',   color='royalblue')
    plt.plot(epochs_range, val_loss_history,   label='Validation Loss', color='tomato')
    plt.axvline(x=stopped_epoch, color='grey', linestyle='--',
                label=f'Early Stop (epoch {stopped_epoch})')
    plt.title('Regularized LSTM v3: Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (scaled)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=120)
    plt.close()
    print(f"Loss curve saved to {loss_curve_path}")

    # -----------------------------------------------------------------------
    # 15. SAVE PREDICTION CHART
    # -----------------------------------------------------------------------
    pred_chart_path = 'reports/lstm_v3_regularized_prediction_vs_actual.png'
    days = range(1, forecast_days + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Regularized LSTM v3: Prediction vs Actual', fontsize=14)

    # Left panel: first test sample
    axes[0].plot(days, y_test_real[sample_idx],     marker='o', label='Actual',    color='royalblue')
    axes[0].plot(days, predictions_real[sample_idx], marker='x', label='Predicted', color='tomato', linestyle='--')
    axes[0].set_title('First Test Sample')
    axes[0].set_xlabel('Forecast Day')
    axes[0].set_ylabel('Units Sold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Right panel: average across all test samples
    axes[1].plot(days, y_test_real.mean(axis=0),     marker='o', label='Actual (avg)',    color='royalblue')
    axes[1].plot(days, predictions_real.mean(axis=0), marker='x', label='Predicted (avg)', color='tomato', linestyle='--')
    axes[1].set_title('Average Across All Test Samples')
    axes[1].set_xlabel('Forecast Day')
    axes[1].set_ylabel('Units Sold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(pred_chart_path, dpi=120)
    plt.close()
    print(f"Prediction chart saved to {pred_chart_path}")

    # -----------------------------------------------------------------------
    # 16. SAVE METRICS FILE
    # -----------------------------------------------------------------------
    metrics_path = 'reports/lstm_v3_regularized_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("LSTM v3 Regularized – Validation Split + Early Stopping\n")
        f.write("=" * 56 + "\n")
        f.write(f"Best Val Loss (scaled) : {best_val_loss:.6f}\n")
        f.write(f"Stopped at epoch       : {stopped_epoch}\n")
        f.write(f"Test MSE (scaled)      : {test_loss:.6f}\n")
        f.write(f"MAE                    : {mae:.2f} units\n")
        f.write(f"RMSE                   : {rmse:.2f} units\n")
        f.write(f"MAPE                   : {mape:.2f}%\n")
        f.write("\nRegularisation Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"LSTM dropout           : 0.3\n")
        f.write(f"Future encoder dropout : 0.2\n")
        f.write(f"Prediction head dropout: 0.2\n")
        f.write(f"Weight decay (L2)      : 1e-4\n")
        f.write(f"Early stopping patience: {early_stopping_patience}\n")
        f.write(f"Max epochs             : {max_epochs}\n")
        f.write(f"Batch size             : {batch_size}\n")
        f.write(f"Learning rate          : 0.001\n")
        f.write(f"Total params           : {total_params:,}\n")
    print(f"Metrics saved to {metrics_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
