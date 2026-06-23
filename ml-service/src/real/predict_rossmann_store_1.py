"""
predict_rossmann_store_1.py
============================
Phase 2.9 - Champion Model Inference Pipeline

DIFFERENCE BETWEEN TRAINING AND INFERENCE:
-------------------------------------------
During TRAINING, the model sees thousands of historical windows where the
true future Sales are known (they're in the dataset). It learns the mapping:
  (past 30 days context) + (known future features) -> next 7 days Sales

During INFERENCE, we have NO future Sales. We only have:
  - The real historical data up to TODAY (last 30 rows = X_past)
  - A business plan for the next 7 days (promotions, holidays = X_future)

The model's job is to fill in the gap -- predict what sales WILL BE.
The scalers must be the same ones fit on training data so the model
receives feature values in the same [0,1] range it was trained on.

WHY THE MODEL NEEDS THE LATEST 30 DAYS (X_past):
-------------------------------------------------
The LSTM encoder summarises the recent demand history into a 64-dimensional
context vector. Without this historical context, the model cannot distinguish:
  - A high-sales week from a post-closure week
  - An upward trend from a seasonal dip
The last 30 days of actual sales are the model's "memory" going into the forecast.

WHY FUTURE KNOWN FEATURES ARE REQUIRED (X_future):
----------------------------------------------------
The second stream of the FutureAwareLSTM reads 7 days of known business
context: Is tomorrow a promotion day? Is it a school holiday? Is it April?
These features are provided by the business BEFORE the forecast period begins.
Without them, the model would have to guess all context -- severely reducing
forecast accuracy on event-driven demand spikes.
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FORECAST_OPEN_DAYS_ONLY = True


os.makedirs("reports/real", exist_ok=True)

# =============================================================================
# STEP 1 -- Load the processed Store 1 CSV
# =============================================================================
print("Loading rossmann_store_1_processed.csv...")
df = pd.read_csv("data/real/processed/rossmann_store_1_processed.csv")

# STEP 2 -- Parse Date and sort chronologically
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded {len(df)} rows | Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# =============================================================================
# STEP 3 -- Recreate EXACTLY the same v2 feature engineering as in
#           preprocess_rossmann_sequences_v2.py
#
# WHY THIS MUST MATCH EXACTLY:
# The past feature scaler (MinMaxScaler) was fit on columns in this exact
# order with these exact transformations. If we compute any column differently
# (e.g., using a different rolling window size or forgetting the shift),
# the scaler will map values into the wrong [0,1] range and the model
# will receive garbage inputs -- silent, hard-to-debug errors.
# =============================================================================

# Lag and rolling features (all use shift(1) to prevent leakage)
df["lag_7_sales"]       = df["Sales"].shift(7)
df["rolling_7_sales"]   = df["Sales"].shift(1).rolling(window=7).mean()
df["rolling_14_sales"]  = df["Sales"].shift(1).rolling(window=14).mean()
df["rolling_7_std_sales"] = df["Sales"].shift(1).rolling(window=7).std()

# Calendar event flags
df["is_month_start"]    = df["Date"].dt.is_month_start.astype(int)
df["is_month_end"]      = df["Date"].dt.is_month_end.astype(int)
df["is_april"]          = (df["Month"] == 4).astype(int)

# Interaction features
df["promo_schoolholiday"]  = df["Promo"] * df["SchoolHoliday"]
df["weekend_schoolholiday"] = df["IsWeekend"] * df["SchoolHoliday"]
df["promo_weekend"]        = df["Promo"] * df["IsWeekend"]

# Promo timing: days since last promo
days_since = []
counter = 9999
for _, row in df.iterrows():
    counter = 0 if row["Promo"] == 1 else counter + 1
    days_since.append(counter)
df["days_since_last_promo"] = days_since

# Promo timing: days until next promo (iterate backwards)
days_until = []
counter = 9999
for _, row in df.iloc[::-1].iterrows():
    counter = 0 if row["Promo"] == 1 else counter + 1
    days_until.append(counter)
days_until.reverse()
df["days_until_next_promo"] = days_until

# Sales momentum (past only -- uses current Sales so excluded from X_future)
df["sales_vs_lag7"] = df["Sales"] - df["lag_7_sales"]

# STEP 4 -- Drop rows with missing lag/rolling values (same as preprocessing)
lag_cols = ["lag_7_sales", "rolling_7_sales", "rolling_14_sales",
            "rolling_7_std_sales", "sales_vs_lag7"]
df = df.dropna(subset=lag_cols).reset_index(drop=True)
print(f"Rows after dropping NaN lag/rolling: {len(df)}")

# =============================================================================
# STEP 5 -- Use the latest 30 rows as X_past
#
# These are the most recent 30 open-day sales records for Store 1.
# They form the historical context window fed to the LSTM encoder.
# =============================================================================
lookback_days = 30
forecast_days = 7

past_feature_cols = [
    "Sales", "Promo", "SchoolHoliday", "DayOfWeek", "Month", "Day", "IsWeekend",
    "lag_7_sales", "rolling_7_sales", "rolling_14_sales", "rolling_7_std_sales",
    "is_month_start", "is_month_end", "is_april",
    "promo_schoolholiday", "weekend_schoolholiday", "promo_weekend",
    "days_since_last_promo", "days_until_next_promo",
    "sales_vs_lag7",
]

future_feature_cols = [
    "Promo", "SchoolHoliday", "DayOfWeek", "Month", "Day", "IsWeekend",
    "lag_7_sales", "rolling_7_sales", "rolling_14_sales", "rolling_7_std_sales",
    "is_month_start", "is_month_end", "is_april",
    "promo_schoolholiday", "weekend_schoolholiday", "promo_weekend",
    "days_since_last_promo", "days_until_next_promo",
]

# Latest 30 rows for the historical context window
df_past = df.tail(lookback_days).copy()
latest_date = df["Date"].max()
print(f"\nLatest available historical date: {latest_date.date()}")
print(f"X_past window: {df_past['Date'].min().date()} to {df_past['Date'].max().date()}")

# Raw (unscaled) past window -- will be scaled in Step 9
X_past_raw = df_past[past_feature_cols].values  # shape (30, 20)

# =============================================================================
# STEP 6 -- Build the 7-day future business plan
# =============================================================================

# Detect whether Store 1 usually closes on Sundays from historical data.
# In the processed data, closed days (Sales=0) were removed. So if DayOfWeek=7
# is missing, it means the store is historically closed on Sundays.
sundays_in_history = len(df[df["DayOfWeek"] == 7])
closes_on_sundays = (sundays_in_history == 0)

forecast_mode = "open days only" if FORECAST_OPEN_DAYS_ONLY else "calendar days"
if closes_on_sundays:
    if FORECAST_OPEN_DAYS_ONLY:
        print("Store 1 appears to be closed on Sundays. Forecasting next 7 open days.")
    else:
        print("Store 1 appears to be closed on Sundays. Forecasting next 7 calendar days.")

future_dates = []
current_dt = latest_date
while len(future_dates) < forecast_days:
    current_dt += pd.Timedelta(days=1)
    if FORECAST_OPEN_DAYS_ONLY and closes_on_sundays and current_dt.weekday() == 6:
        continue  # Skip Sundays if forecasting open days only
    future_dates.append(current_dt)

# Build a DataFrame for the 7 future days
future_rows = []
for future_dt in future_dates:
    row = {}
    row["Date"]          = future_dt
    row["DayOfWeek"]     = future_dt.weekday() + 1   # 1=Mon, 7=Sun (Rossmann format)
    row["Month"]         = future_dt.month
    row["Day"]           = future_dt.day
    row["IsWeekend"]     = 1 if row["DayOfWeek"] in [6, 7] else 0
    row["Open"]          = 0 if (closes_on_sundays and row["DayOfWeek"] == 7) else 1

    # Default business plan -- no promotion, no school holiday
    row["Promo"]         = 0
    row["SchoolHoliday"] = 0

    # Calendar event flags
    row["is_month_start"] = int(future_dt.is_month_start)
    row["is_month_end"]   = int(future_dt.is_month_end)
    row["is_april"]       = int(future_dt.month == 4)

    # Interaction terms
    row["promo_schoolholiday"]   = row["Promo"] * row["SchoolHoliday"]
    row["weekend_schoolholiday"] = row["IsWeekend"] * row["SchoolHoliday"]
    row["promo_weekend"]         = row["Promo"] * row["IsWeekend"]

    # lag_7_sales: look up actual sales from 7 days before this future date
    lag_date = future_dt - pd.Timedelta(days=7)
    lag_match = df[df["Date"] == lag_date]
    if not lag_match.empty:
        row["lag_7_sales"] = float(lag_match["Sales"].values[0])
    else:
        row["lag_7_sales"] = float(df["Sales"].tail(7).mean())

    # rolling_7/14_sales and std: use the most recent known values
    row["rolling_7_sales"]    = float(df["rolling_7_sales"].iloc[-1])
    row["rolling_14_sales"]   = float(df["rolling_14_sales"].iloc[-1])
    row["rolling_7_std_sales"] = float(df["rolling_7_std_sales"].iloc[-1])

    future_rows.append(row)

df_future = pd.DataFrame(future_rows)

# Promo timing for future window
# IMPORTANT: the scaler was fit with days_since/until max = 10.
# Never-promo default of 9999 would scale to ~999, completely out of range.
# We cap at 10 to stay within the distribution the scaler knows.
MAX_PROMO_TIMING = 10

last_days_since = int(df["days_since_last_promo"].iloc[-1])
future_days_since = []
counter = last_days_since
for _, row in df_future.iterrows():
    counter = 0 if row["Promo"] == 1 else counter + 1
    future_days_since.append(min(counter, MAX_PROMO_TIMING))
df_future["days_since_last_promo"] = future_days_since

future_days_until = []
counter = MAX_PROMO_TIMING  # cap: no promo in 7-day plan = max distance
for _, row in df_future.iloc[::-1].iterrows():
    counter = 0 if row["Promo"] == 1 else counter + 1
    future_days_until.append(min(counter, MAX_PROMO_TIMING))
future_days_until.reverse()
df_future["days_until_next_promo"] = future_days_until

# Raw (unscaled) future window
X_future_raw = df_future[future_feature_cols].values  # shape (7, 18)

print(f"\nFuture forecast dates: {future_dates[0].date()} to {future_dates[-1].date()}")

# =============================================================================
# STEP 8 -- Load scalers (fitted on training data in preprocessing v2)
# =============================================================================
past_scaler   = joblib.load("models/real/rossmann_v2_past_feature_scaler.pkl")
future_scaler = joblib.load("models/real/rossmann_v2_future_feature_scaler.pkl")
target_scaler = joblib.load("models/real/rossmann_v2_target_scaler.pkl")

# =============================================================================
# STEP 9 -- Scale X_past and X_future using the training scalers
#
# WHY WE PASS A DATAFRAME (NOT A RAW ARRAY):
# The MinMaxScaler was originally fit on a pandas DataFrame with named columns.
# When we pass a plain numpy array, sklearn cannot verify column order and
# silently applies the wrong min/max to each column -- producing garbage outputs.
# Wrapping in a DataFrame with the same column names guarantees the scaler
# maps each feature using the correct stored range.
# =============================================================================
df_past_for_scale   = pd.DataFrame(X_past_raw,   columns=past_feature_cols)
df_future_for_scale = pd.DataFrame(X_future_raw, columns=future_feature_cols)

X_past_scaled   = past_scaler.transform(df_past_for_scale)    # shape (30, 20)
X_future_scaled = future_scaler.transform(df_future_for_scale) # shape (7, 18)

# Safety clip: ensure all scaled values stay in [0, 1].
# Out-of-range values (e.g., a DayOfWeek=7 where max was 6) produce >1.0
# which the model has never seen and can cause wild predictions.
X_past_scaled   = np.clip(X_past_scaled,   0.0, 1.0)
X_future_scaled = np.clip(X_future_scaled, 0.0, 1.0)

# Add batch dimension: model expects (batch_size, seq_len, features)
X_past_t   = torch.tensor(X_past_scaled[np.newaxis, :, :],   dtype=torch.float32)  # (1, 30, 20)
X_future_t = torch.tensor(X_future_scaled[np.newaxis, :, :], dtype=torch.float32)  # (1, 7, 18)

# =============================================================================
# STEP 10 -- Recreate the EXACT same architecture as in training
# =============================================================================
class RossmannEnhancedFutureAwareLSTM(nn.Module):
    def __init__(self, past_features: int, future_features: int, forecast_days: int):
        super().__init__()
        self.forecast_days = forecast_days
        self.past_lstm = nn.LSTM(
            input_size=past_features, hidden_size=64, num_layers=2,
            batch_first=True, dropout=0.3
        )
        self.future_encoder = nn.Sequential(
            nn.Linear(future_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(64 + 32, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, x_past, x_future):
        _, (hn, _) = self.past_lstm(x_past)
        past_context  = hn[-1]
        past_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1)
        future_enc    = self.future_encoder(x_future)
        fused         = torch.cat([past_repeated, future_enc], dim=2)
        return self.prediction_head(fused).squeeze(-1)


model = RossmannEnhancedFutureAwareLSTM(
    past_features   = len(past_feature_cols),    # 20
    future_features = len(future_feature_cols),  # 18
    forecast_days   = forecast_days              # 7
)

# =============================================================================
# STEP 11 -- Load model weights
# =============================================================================
model.load_state_dict(torch.load(
    "models/real/rossmann_store_1_lstm_v2_model.pth",
    map_location="cpu"
))
# IMPORTANT: call eval() to disable dropout for deterministic inference
model.eval()

# =============================================================================
# STEP 12 -- Predict next 7 days
#
# WHY torch.no_grad():
# During inference we do not compute gradients (no backpropagation).
# no_grad() saves memory and speeds up the forward pass.
# =============================================================================
with torch.no_grad():
    predictions_scaled = model(X_past_t, X_future_t)  # shape: (1, 7)

# predictions_scaled is a 2D tensor (1 batch x 7 days)
predictions_scaled_np = predictions_scaled.numpy()    # shape: (1, 7)

# =============================================================================
# STEP 13 -- Inverse transform back to real Sales units
# =============================================================================
predicted_sales = target_scaler.inverse_transform(predictions_scaled_np).flatten()  # (7,)

# Set predicted_sales = 0 for closed days if forecasting calendar days
for i, row in enumerate(future_rows):
    if row["Open"] == 0:
        predicted_sales[i] = 0.0

# =============================================================================
# STEP 14 -- Print forecast summary
# =============================================================================
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
total_sales   = predicted_sales.sum()
avg_sales     = predicted_sales.mean()
peak_idx      = int(np.argmax(predicted_sales))
trough_idx    = int(np.argmin(predicted_sales))

print("\n" + "=" * 55)
print("ROSSMANN STORE 1 - NEXT 7 DAY SALES FORECAST")
print("=" * 55)
print(f"Latest historical date : {latest_date.date()}")
print(f"Forecast mode          : {forecast_mode}")
print(f"Forecast period        : {future_dates[0].date()} to {future_dates[-1].date()}")
print()
print(f"{'Date':<14} {'Weekday':<10} {'Promo':<8} {'Predicted Sales':>15}")
print("-" * 50)
for i, (dt, sales) in enumerate(zip(future_dates, predicted_sales)):
    weekday = day_names[dt.weekday()]
    promo   = int(df_future["Promo"].iloc[i])
    is_open = df_future["Open"].iloc[i]
    if is_open == 0:
        print(f"{str(dt.date()):<14} {weekday:<10} {'Yes' if promo else 'No':<8} {'0 (Closed)':>15}")
    else:
        marker  = " <<< PEAK" if i == peak_idx else (" <<< LOW" if i == trough_idx else "")
        print(f"{str(dt.date()):<14} {weekday:<10} {'Yes' if promo else 'No':<8} {sales:>12,.0f}{marker}")

print("-" * 50)
print(f"{'Total predicted sales:':<35} {total_sales:>12,.0f}")
print(f"{'Average predicted sales:':<35} {avg_sales:>12,.0f}")
print(f"{'Highest demand day:':<35} {str(future_dates[peak_idx].date())}")
print(f"{'Lowest demand day:':<35} {str(future_dates[trough_idx].date())}")

# =============================================================================
# STEP 15 -- Save forecast CSV
# =============================================================================
columns = ["Date", "DayOfWeek", "IsWeekend", "Promo", "SchoolHoliday"]
if not FORECAST_OPEN_DAYS_ONLY:
    columns.append("Open")
columns.append("predicted_sales")

df_forecast = pd.DataFrame({
    "Date"           : [d.date() for d in future_dates],
    "DayOfWeek"      : df_future["DayOfWeek"].values,
    "IsWeekend"      : df_future["IsWeekend"].values,
    "Promo"          : df_future["Promo"].values,
    "SchoolHoliday"  : df_future["SchoolHoliday"].values,
})

if not FORECAST_OPEN_DAYS_ONLY:
    df_forecast["Open"] = df_future["Open"].values

df_forecast["predicted_sales"] = np.round(predicted_sales).astype(int)

# Reorder columns to exactly match prompt for required columns
# Prompt asked for: Date, DayOfWeek, IsWeekend, Promo, SchoolHoliday, predicted_sales
# (and Open if calendar days)
df_forecast = df_forecast[columns]

forecast_csv = "reports/real/rossmann_store_1_next_7_day_forecast.csv"
df_forecast.to_csv(forecast_csv, index=False)
print(f"\nSaved forecast CSV -> {forecast_csv}")

# =============================================================================
# STEP 16 -- Save forecast chart
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5))

bar_colors = ["steelblue" if not p else "tomato"
              for p in df_future["Promo"].values]

bars = ax.bar(range(forecast_days), predicted_sales, color=bar_colors,
              edgecolor="white", width=0.65)

# Annotate each bar with the predicted value
for i, (bar, val) in enumerate(zip(bars, predicted_sales)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

x_labels = [f"{day_names[d.weekday()]}\n{d.date()}" for d in future_dates]
ax.set_xticks(range(forecast_days))
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_title("Rossmann Store 1 - Next 7 Day Sales Forecast\n(LSTM v2 Champion Model)",
             fontsize=13)
ax.set_ylabel("Predicted Sales (units)")
ax.set_ylim(0, predicted_sales.max() * 1.3)

# Legend
promo_patch    = mpatches.Patch(color="tomato",    label="Promotion Active")
no_promo_patch = mpatches.Patch(color="steelblue", label="No Promotion")
ax.legend(handles=[promo_patch, no_promo_patch], loc="upper right")

plt.tight_layout()
forecast_chart = "reports/real/rossmann_store_1_next_7_day_forecast.png"
plt.savefig(forecast_chart, dpi=120)
plt.close()
print(f"Saved forecast chart -> {forecast_chart}")

print("\nPhase 2.9 - Inference pipeline complete.")
