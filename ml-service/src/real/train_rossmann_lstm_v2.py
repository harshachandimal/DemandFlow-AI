"""
train_rossmann_lstm_v2.py
==========================
Phase 2.7 - Train Rossmann LSTM v2 with Enhanced Event/Context Features

WHAT IS NEW IN V2 vs V1:
--------------------------
Phase 2.5 diagnosis revealed specific failure conditions for the v1 model:
  * April (Easter) caused the highest errors - no holiday flag existed in v1
  * Promo x SchoolHoliday coincidences were underpredicted
  * Weekend volatility was underestimated
  * The model had no information about promo timing (lead/lag)

V2 adds 11 new features to address exactly these failure modes.
The architecture is unchanged so we can fairly attribute any improvement
to the richer feature set rather than a model capacity increase.

V1 baseline to beat:
  MAE  = 351.74
  RMSE = 479.05
  MAPE = 7.76%
"""

import os
import random
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY SEED
#
# WHY WE FIX THE RANDOM SEED:
# Neural network training is non-deterministic by default:
#   - Weight initialisation uses random numbers
#   - DataLoader shuffling uses random numbers
# Without a fixed seed, each run produces different results.
# This makes it impossible to know whether v2 "beats" v1 consistently
# or whether the result is just a lucky draw.
# Fixing the seed guarantees the same weights, same batch order, and
# therefore the same final metric on every run.
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed fixed at {SEED} for reproducibility.")

os.makedirs("models/real", exist_ok=True)
os.makedirs("reports/real", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load the v2 sequence arrays
# ─────────────────────────────────────────────────────────────────────────────
data = np.load("data/real/processed/rossmann_store_1_sequences_v2.npz")
X_past_train_full  = data['X_past_train']
X_future_train_full = data['X_future_train']
y_train_full       = data['y_train']
X_past_test        = data['X_past_test']
X_future_test      = data['X_future_test']
y_test             = data['y_test']

# Load the v2 target scaler for inverse-transforming predictions
target_scaler = joblib.load("models/real/rossmann_v2_target_scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Print loaded shapes
# ─────────────────────────────────────────────────────────────────────────────
print("Loaded v2 sequence shapes:")
print(f"  X_past_train  : {X_past_train_full.shape}")
print(f"  X_future_train: {X_future_train_full.shape}")
print(f"  y_train       : {y_train_full.shape}")
print(f"  X_past_test   : {X_past_test.shape}")
print(f"  X_future_test : {X_future_test.shape}")
print(f"  y_test        : {y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Chronological 80/20 train / validation split
#
# WHY CHRONOLOGICAL, NOT RANDOM:
# Shuffling would mix future timestamps into the training set.
# The model would then "remember" future demand patterns during training,
# producing misleadingly low validation loss — a form of look-ahead bias.
# We always keep time order intact for time-series data.
# ─────────────────────────────────────────────────────────────────────────────
split_idx = int(len(X_past_train_full) * 0.80)

X_past_train    = X_past_train_full[:split_idx]
X_future_train  = X_future_train_full[:split_idx]
y_train         = y_train_full[:split_idx]

X_past_val      = X_past_train_full[split_idx:]
X_future_val    = X_future_train_full[split_idx:]
y_val           = y_train_full[split_idx:]

print(f"\nTrain  : {X_past_train.shape[0]} samples")
print(f"Val    : {X_past_val.shape[0]} samples")
print(f"Test   : {X_past_test.shape[0]} samples")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Convert arrays to PyTorch tensors
#
# float32 is the standard precision for neural network weights and activations.
# float64 would double memory consumption with no practical accuracy benefit
# for LSTM training on tabular data.
# ─────────────────────────────────────────────────────────────────────────────
X_past_train_t   = torch.tensor(X_past_train,   dtype=torch.float32)
X_future_train_t = torch.tensor(X_future_train, dtype=torch.float32)
y_train_t        = torch.tensor(y_train,         dtype=torch.float32)

X_past_val_t     = torch.tensor(X_past_val,     dtype=torch.float32)
X_future_val_t   = torch.tensor(X_future_val,   dtype=torch.float32)
y_val_t          = torch.tensor(y_val,           dtype=torch.float32)

X_past_test_t    = torch.tensor(X_past_test,    dtype=torch.float32)
X_future_test_t  = torch.tensor(X_future_test,  dtype=torch.float32)
y_test_t         = torch.tensor(y_test,          dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TensorDatasets and DataLoaders
#
# WHY shuffle=True for training only:
# Shuffling training batches prevents the model from learning a spurious
# "time order = better prediction" shortcut. Each epoch sees the training
# windows in a different random order, improving generalisation.
# Validation and test must not be shuffled so results are reproducible
# and comparable across runs.
# ─────────────────────────────────────────────────────────────────────────────
batch_size = 16

train_loader = DataLoader(
    TensorDataset(X_past_train_t, X_future_train_t, y_train_t),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_past_val_t, X_future_val_t, y_val_t),
    batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(X_past_test_t, X_future_test_t, y_test_t),
    batch_size=batch_size, shuffle=False
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Model definition: RossmannEnhancedFutureAwareLSTM
#
# ARCHITECTURAL CHANGES VS V1:
# The future encoder is now a 2-layer MLP instead of a 1-layer linear.
#   v1: Linear(8  → 32) → ReLU → Dropout
#   v2: Linear(18 → 64) → ReLU → Dropout → Linear(64 → 32) → ReLU
#
# Why the deeper encoder?
# V2 future features include interaction terms and timing features that
# have non-linear relationships (e.g., days_until_next_promo × is_april).
# A two-layer MLP can approximate these non-linearities before the
# fusion step, whereas a single linear layer can only do weighted averaging.
# ─────────────────────────────────────────────────────────────────────────────
class RossmannEnhancedFutureAwareLSTM(nn.Module):
    def __init__(self, past_features: int, future_features: int, forecast_days: int):
        super().__init__()
        self.forecast_days = forecast_days

        # A. Past encoder — LSTM reads the 30-day historical window
        #    hidden_size=64: each hidden state is a 64-dimensional context vector
        #    num_layers=2: stacked LSTMs; the second layer learns higher-level patterns
        #    dropout=0.3: randomly zero 30% of activations between LSTM layers
        #                 during training to prevent over-reliance on any single neuron
        self.past_lstm = nn.LSTM(
            input_size=past_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # B. Future feature encoder — deeper MLP for richer feature interactions
        #    18 input features → 64 hidden units → 32 output units
        #    The extra layer helps the network pre-process interaction terms
        #    (e.g., promo_schoolholiday, days_until_next_promo) non-linearly.
        self.future_encoder = nn.Sequential(
            nn.Linear(future_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # D. Prediction head — takes fused (past + future) context and outputs 1 value
        #    Input: 64 (past context) + 32 (future context) = 96 dimensions per day
        self.prediction_head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        # ── A. Encode the 30-day past window ──────────────────────────────────
        # past_out shape: (batch, seq_len=30, hidden=64)  — all time steps
        # hn shape:       (num_layers=2, batch, hidden=64) — final hidden states
        _, (hn, _) = self.past_lstm(x_past)

        # ── C. Fusion: replicate past context across the 7 forecast days ──────
        # We take hn[-1] = the last LSTM layer's final hidden state.
        # This is the compressed "summary" of everything seen in the past window.
        # Shape: (batch, 64)
        past_context = hn[-1]

        # Expand to (batch, forecast_days, 64) so each forecast day gets
        # the same historical context vector.
        past_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1)

        # ── B. Encode the 7-day known future features ────────────────────────
        # x_future shape: (batch, forecast_days=7, future_features=18)
        # future_encoded: (batch, 7, 32)
        future_encoded = self.future_encoder(x_future)

        # ── Concatenate past context + future encoding per forecast day ───────
        # fused shape: (batch, 7, 96)
        fused = torch.cat([past_repeated, future_encoded], dim=2)

        # ── D. Predict one sales value per forecast day ───────────────────────
        # predictions shape: (batch, 7, 1) → squeeze → (batch, 7)
        return self.prediction_head(fused).squeeze(-1)


past_features   = X_past_train.shape[2]    # 20
future_features = X_future_train.shape[2]  # 18
forecast_days   = y_train.shape[1]         # 7

model = RossmannEnhancedFutureAwareLSTM(past_features, future_features, forecast_days)
print(f"\nModel: RossmannEnhancedFutureAwareLSTM")
print(f"  Past input size:   {past_features}")
print(f"  Future input size: {future_features}")
print(f"  Forecast horizon:  {forecast_days} days")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Loss function, optimizer, and hyperparameters
#
# MSELoss: penalizes large errors quadratically — useful for demand forecasting
#   where a huge overestimate or underestimate is much worse than a small one.
#
# Adam with weight_decay=1e-4: L2 regularisation gently shrinks large weights
#   at each step, discouraging the model from memorizing training noise.
#   weight_decay=1e-4 is a mild setting that rarely hurts convergence.
# ─────────────────────────────────────────────────────────────────────────────
criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

max_epochs              = 300
early_stopping_patience = 25

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Training loop with early stopping
#
# WHY EARLY STOPPING:
# Adding 11 new features gives the v2 model more capacity — but also more
# opportunity to memorize training noise (overfitting).
# Early stopping monitors validation loss after every epoch:
#   • If val loss improves → save model weights, reset patience counter.
#   • If val loss does not improve → increment patience counter.
#   • If patience counter reaches 25 → stop training, restore best weights.
#
# This guarantees we keep the model at the epoch where it GENERALISES best,
# not the epoch where training loss is lowest.
# ─────────────────────────────────────────────────────────────────────────────
best_val_loss    = float('inf')
patience_counter = 0
best_model_state = None
stopped_epoch    = max_epochs
train_losses     = []
val_losses       = []

print("\nStarting training...")

for epoch in range(1, max_epochs + 1):

    # ── Training phase ────────────────────────────────────────────────────────
    # model.train() enables dropout layers (they are disabled during eval).
    model.train()
    epoch_train_loss = 0.0
    for x_p, x_f, y_b in train_loader:
        optimizer.zero_grad()           # clear stale gradients
        preds = model(x_p, x_f)        # forward pass
        loss  = criterion(preds, y_b)  # compute MSE loss
        loss.backward()                 # backpropagate
        optimizer.step()                # update weights
        epoch_train_loss += loss.item() * len(y_b)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # ── Validation phase ──────────────────────────────────────────────────────
    # model.eval() disables dropout so validation is deterministic.
    # torch.no_grad() skips building the computation graph, saving memory.
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for x_p, x_f, y_b in val_loader:
            preds = model(x_p, x_f)
            loss  = criterion(preds, y_b)
            epoch_val_loss += loss.item() * len(y_b)
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    # ── Early stopping logic ──────────────────────────────────────────────────
    if epoch_val_loss < best_val_loss:
        best_val_loss    = epoch_val_loss
        patience_counter = 0
        # Deep-copy state dict so we can restore after early stopping
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if epoch % 10 == 0:
        print(f"Epoch [{epoch:>3}/{max_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}  "
              f"Val Loss: {epoch_val_loss:.4f}  "
              f"Best Val: {best_val_loss:.4f}")

    if patience_counter >= early_stopping_patience:
        print(f"\nEarly stopping triggered at epoch {epoch}.")
        stopped_epoch = epoch
        break

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Restore best model weights
# ─────────────────────────────────────────────────────────────────────────────
if best_model_state is not None:
    model.load_state_dict(best_model_state)
print(f"Restored best weights from epoch {stopped_epoch - patience_counter}.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Predict on test set
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
all_preds   = []
all_actuals = []
test_mse    = 0.0

with torch.no_grad():
    for x_p, x_f, y_b in test_loader:
        preds = model(x_p, x_f)
        loss  = criterion(preds, y_b)
        test_mse    += loss.item() * len(y_b)
        all_preds.append(preds.numpy())
        all_actuals.append(y_b.numpy())

test_mse    /= len(test_loader.dataset)
all_preds    = np.vstack(all_preds)
all_actuals  = np.vstack(all_actuals)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Inverse-transform predictions back to real sales units
#
# We scaled Sales to [0, 1] during preprocessing.
# inverse_transform maps the predictions back to the original unit scale
# (number of items sold), which is what business stakeholders care about.
# ─────────────────────────────────────────────────────────────────────────────
y_test_inv   = target_scaler.inverse_transform(all_actuals)
preds_inv    = target_scaler.inverse_transform(all_preds)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Compute evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────
mae  = np.mean(np.abs(y_test_inv - preds_inv))
rmse = np.sqrt(np.mean((y_test_inv - preds_inv) ** 2))
mape = np.mean(np.abs((y_test_inv - preds_inv) / y_test_inv)) * 100

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — Print final results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("TRAINING COMPLETE - v2 RESULTS")
print("=" * 55)
print(f"Stopped epoch      : {stopped_epoch}")
print(f"Best val loss      : {best_val_loss:.4f}")
print(f"Final test MSE     : {test_mse:.4f}")
print(f"MAE                : {mae:.2f}")
print(f"RMSE               : {rmse:.2f}")
print(f"MAPE               : {mape:.2f}%")
print(f"\nSample [0] - Actual  next 7 days: {np.round(y_test_inv[0]).astype(int)}")
print(f"Sample [0] - Predict next 7 days: {np.round(preds_inv[0]).astype(int)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — Compare against v1 and baselines
# ─────────────────────────────────────────────────────────────────────────────
v1_mae, v1_rmse, v1_mape    = 351.74, 479.05, 7.76
baseline_lv_mape            = 20.96
baseline_weekly_mape        = 25.79

print("\n" + "-" * 55)
print("COMPARISON")
print("-" * 55)
print(f"v1 MAPE = {v1_mape:.2f}%   ->  v2 MAPE = {mape:.2f}%")
if mape < v1_mape:
    print(f"[PASS] v2 BEATS LSTM v1   (improvement: {v1_mape - mape:.2f} pp)")
else:
    print(f"[FAIL] v2 does NOT beat LSTM v1  (regression: {mape - v1_mape:.2f} pp)")

if mape < baseline_lv_mape:
    print(f"[PASS] v2 BEATS Last Value Baseline ({baseline_lv_mape:.2f}%)")
else:
    print(f"[FAIL] v2 does NOT beat Last Value Baseline ({baseline_lv_mape:.2f}%)")

if mape < baseline_weekly_mape:
    print(f"[PASS] v2 BEATS Weekly Seasonal Baseline ({baseline_weekly_mape:.2f}%)")
else:
    print(f"[FAIL] v2 does NOT beat Weekly Seasonal Baseline ({baseline_weekly_mape:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15 — Save model
# ─────────────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "models/real/rossmann_store_1_lstm_v2_model.pth")
print("\nSaved model -> models/real/rossmann_store_1_lstm_v2_model.pth")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 16 — Prediction vs Actual chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test_inv[0],  marker='o', label="Actual (Sample 0)")
ax.plot(preds_inv[0],   marker='x', label="Predicted (Sample 0)")
ax.set_title("LSTM v2 - Prediction vs Actual (Store 1, Sample 0)")
ax.set_xlabel("Forecast Day")
ax.set_ylabel("Sales")
ax.legend()
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_lstm_v2_prediction_vs_actual.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 17 — Loss curve
# ─────────────────────────────────────────────────────────────────────────────
best_epoch = int(np.argmin(val_losses)) + 1   # 1-indexed

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, label="Training Loss")
ax.plot(val_losses,   label="Validation Loss")
ax.axvline(x=best_epoch - 1, color='red', linestyle='--',
           label=f"Best model (epoch {best_epoch})")
ax.set_title("LSTM v2 - Training & Validation Loss Curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_lstm_v2_loss_curve.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 18 — Save metrics text file
# ─────────────────────────────────────────────────────────────────────────────
metrics_path = "reports/real/rossmann_store_1_lstm_v2_metrics.txt"
with open(metrics_path, "w") as f:
    f.write("Rossmann Store 1 - LSTM v2 Metrics\n")
    f.write("===================================\n\n")
    f.write(f"Stopped Epoch     : {stopped_epoch}\n")
    f.write(f"Best Val Loss     : {best_val_loss:.4f}\n")
    f.write(f"Final Test MSE    : {test_mse:.4f}\n")
    f.write(f"MAE               : {mae:.2f}\n")
    f.write(f"RMSE              : {rmse:.2f}\n")
    f.write(f"MAPE              : {mape:.2f}%\n\n")
    f.write("Comparison:\n")
    f.write(f"  LSTM v1 MAPE              : {v1_mape:.2f}%\n")
    f.write(f"  LSTM v2 MAPE              : {mape:.2f}%\n")
    f.write(f"  Last Value Baseline MAPE  : {baseline_lv_mape:.2f}%\n")
    f.write(f"  Weekly Seasonal MAPE      : {baseline_weekly_mape:.2f}%\n\n")
    beat_v1 = "YES" if mape < v1_mape else "NO"
    beat_lv = "YES" if mape < baseline_lv_mape else "NO"
    beat_ws = "YES" if mape < baseline_weekly_mape else "NO"
    f.write(f"  Beats LSTM v1             : {beat_v1}\n")
    f.write(f"  Beats Last Value Baseline : {beat_lv}\n")
    f.write(f"  Beats Weekly Baseline     : {beat_ws}\n")

print(f"Saved metrics -> {metrics_path}")
print("\nPhase 2.7 - LSTM v2 training complete.")
