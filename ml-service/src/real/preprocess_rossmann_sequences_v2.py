"""
preprocess_rossmann_sequences_v2.py
====================================
Phase 2.6 — Enhanced Real-Data Preprocessing with Event/Context Features

WHY WE ADD MORE FEATURES AFTER DIAGNOSIS
------------------------------------------
In Phase 2.5 we discovered that the largest model errors cluster around:
  • April (Easter public holidays)
  • Promotion days with school holidays active simultaneously
  • Saturday (weekend spike volatility)

The v1 model had no way to "see" these conditions explicitly.
This v2 preprocessing adds:
  • Calendar event flags (month start/end, is_april)
  • Interaction terms (promo × school_holiday, weekend × school_holiday)
  • Promo timing context (days_since_last_promo, days_until_next_promo)
  • Extended rolling features (14-day average, 7-day std dev)
  • Sales momentum (how today compares to last week — past only)

Together these give the LSTM richer signals to understand volatile days
WITHOUT leaking any information from the future into the input.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

os.makedirs("data/real/processed", exist_ok=True)
os.makedirs("models/real", exist_ok=True)
os.makedirs("reports/real", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load processed CSV
# ─────────────────────────────────────────────────────────────────────────────
input_csv = "data/real/processed/rossmann_store_1_processed.csv"
print(f"Loading {input_csv}...")
df = pd.read_csv(input_csv)
print(f"Total rows before feature engineering: {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 & 3 — Parse Date as datetime and sort chronologically
# ─────────────────────────────────────────────────────────────────────────────
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Basic lag and rolling features
#
# WHY WE USE shift(1) BEFORE EVERY ROLLING WINDOW:
# If we compute rolling(7).mean() on the raw Sales column, the value at
# row i includes Sales at row i itself — that is future leakage, because
# we're trying to *predict* Sales and cannot use today's value as an input.
# shift(1) shifts every row down by one position, so the rolling window at
# row i only looks at rows i-1, i-2, … i-7 (all in the past).
# ─────────────────────────────────────────────────────────────────────────────

# lag_7_sales: exact sales value 7 days ago (same weekday, same seasonal slot)
# ML purpose: gives the model the "weekly seasonal baseline" as an explicit feature
df['lag_7_sales'] = df['Sales'].shift(7)

# rolling_7_sales: smooth 7-day average of past sales (excludes today via shift)
# ML purpose: captures the short-term sales trend; reduces daily noise
df['rolling_7_sales'] = df['Sales'].shift(1).rolling(window=7).mean()

# rolling_14_sales: smooth 14-day average of past sales
# ML purpose: captures a medium-term trend; complements the 7-day window
df['rolling_14_sales'] = df['Sales'].shift(1).rolling(window=14).mean()

# rolling_7_std_sales: standard deviation of sales over the past 7 days
# ML purpose: measures recent sales VOLATILITY — high std means unstable
# demand, which helps the model widen its uncertainty appropriately
df['rolling_7_std_sales'] = df['Sales'].shift(1).rolling(window=7).std()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Event / context features
#
# WHY WE ADDED THESE AFTER PHASE 2.5 DIAGNOSIS:
# The diagnosis revealed that errors clustered around specific calendar events
# (April / Easter, month boundaries, promo+school-holiday combinations).
# Adding explicit flags lets the LSTM immediately identify those conditions
# instead of having to infer them from noisy historical values alone.
# ─────────────────────────────────────────────────────────────────────────────

# is_month_start: 1 if date is the first calendar day of the month
# ML purpose: month-start days often have different buying patterns
# (payday effects, restocking cycles)
df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)

# is_month_end: 1 if date is the last calendar day of the month
# ML purpose: end-of-month clearance sales, budget depletion effects
df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)

# is_april: 1 if the date falls in April
# ML purpose: Phase 2.5 showed April has by far the highest average error
# because of Easter. A dedicated flag gives the model an explicit warning
# that April predictions are structurally different from other months.
df['is_april'] = (df['Month'] == 4).astype(int)

# ── Interaction features ──────────────────────────────────────────────────────
# WHY INTERACTION FEATURES HELP:
# Promotions alone raise demand. School holidays alone raise demand.
# But Promo × SchoolHoliday can produce a MULTIPLICATIVE effect — both
# parents and children are available and a promotion is running.
# An LSTM could theoretically learn this from two separate features, but
# providing the product directly reduces the learning burden and speeds
# convergence, especially with limited data.

# promo_schoolholiday: 1 only when BOTH Promo == 1 AND SchoolHoliday == 1
df['promo_schoolholiday'] = df['Promo'] * df['SchoolHoliday']

# weekend_schoolholiday: 1 only when weekend AND school holiday coincide
df['weekend_schoolholiday'] = df['IsWeekend'] * df['SchoolHoliday']

# promo_weekend: 1 only when a promotion runs on a weekend
# ML purpose: weekend promotions in retail generate much higher footfall
# than identical weekday promotions
df['promo_weekend'] = df['Promo'] * df['IsWeekend']

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Promo timing features
#
# WHY PROMO TIMING HELPS:
# Demand often anticipates or decays around promotions.
# "Sales are high 3 days BEFORE a promo starts" (stock-up behavior).
# "Sales are low 2 days AFTER a promo ends" (demand brought forward).
# These effects are invisible to a model that only sees a binary Promo flag.
#
# days_until_next_promo is ALLOWED in future features because businesses
# know their own promotion schedule in advance.
# ─────────────────────────────────────────────────────────────────────────────

# ── days_since_last_promo ─────────────────────────────────────────────────────
days_since = []
counter = 9999   # Large default for rows before the first promo
for _, row in df.iterrows():
    if row['Promo'] == 1:
        counter = 0
    else:
        counter += 1
    days_since.append(counter)
df['days_since_last_promo'] = days_since

# ── days_until_next_promo ─────────────────────────────────────────────────────
days_until = []
counter = 9999   # Large default for rows after the last promo
for _, row in df.iloc[::-1].iterrows():   # iterate backwards
    if row['Promo'] == 1:
        counter = 0
    else:
        counter += 1
    days_until.append(counter)
days_until.reverse()
df['days_until_next_promo'] = days_until

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Sales momentum feature (PAST ONLY)
#
# WHY WE ONLY PUT THIS IN X_PAST:
# sales_vs_lag7 = today's Sales − last week's Sales.
# It measures whether demand is accelerating or decelerating relative to
# the same weekday last week (upward vs downward momentum).
# We CANNOT include this in X_future because future Sales values are unknown —
# that is exactly what we are trying to predict.
# Including future Sales in X_future would be DATA LEAKAGE.
# ─────────────────────────────────────────────────────────────────────────────
df['sales_vs_lag7'] = df['Sales'] - df['lag_7_sales']

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Drop rows with missing lag/rolling values
# The first 14 rows will have NaN in rolling_14_sales.
# dropna removes them so every training sample is complete.
# ─────────────────────────────────────────────────────────────────────────────
lag_cols = ['lag_7_sales', 'rolling_7_sales', 'rolling_14_sales',
            'rolling_7_std_sales', 'sales_vs_lag7']
df = df.dropna(subset=lag_cols).reset_index(drop=True)
print(f"Total rows after dropping missing lag/rolling values: {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Define lookback and forecast windows (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
lookback_days = 30
forecast_days = 7

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Past feature columns (all features, including sales_vs_lag7)
# ─────────────────────────────────────────────────────────────────────────────
past_feature_cols = [
    # Core sales signal
    'Sales',
    # Marketing / events
    'Promo', 'SchoolHoliday',
    # Calendar basics
    'DayOfWeek', 'Month', 'Day', 'IsWeekend',
    # Lag & rolling signals (all past-only, no leakage)
    'lag_7_sales', 'rolling_7_sales', 'rolling_14_sales', 'rolling_7_std_sales',
    # Calendar event flags (added after Phase 2.5 diagnosis)
    'is_month_start', 'is_month_end', 'is_april',
    # Interaction features
    'promo_schoolholiday', 'weekend_schoolholiday', 'promo_weekend',
    # Promo timing
    'days_since_last_promo', 'days_until_next_promo',
    # Momentum (past-only because it uses current Sales)
    'sales_vs_lag7',
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Future known feature columns (NO sales, NO sales_vs_lag7)
#
# WHY future Sales and sales_vs_lag7 are EXCLUDED:
# The entire purpose of this model is to predict future Sales.
# If we included future Sales as an input feature, the model would simply
# copy it to the output — perfect training accuracy, zero real-world utility.
# This is called DATA LEAKAGE and must be prevented at all costs.
# ─────────────────────────────────────────────────────────────────────────────
future_feature_cols = [
    # Marketing / events (known in advance from promotion schedule)
    'Promo', 'SchoolHoliday',
    # Calendar basics (known in advance trivially)
    'DayOfWeek', 'Month', 'Day', 'IsWeekend',
    # Lag and rolling (historical values, known at forecast time)
    'lag_7_sales', 'rolling_7_sales', 'rolling_14_sales', 'rolling_7_std_sales',
    # Calendar event flags
    'is_month_start', 'is_month_end', 'is_april',
    # Interaction features
    'promo_schoolholiday', 'weekend_schoolholiday', 'promo_weekend',
    # Promo timing (known from the schedule)
    'days_since_last_promo', 'days_until_next_promo',
    # NOTE: 'Sales' and 'sales_vs_lag7' are intentionally absent
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Target
# ─────────────────────────────────────────────────────────────────────────────
target_col = 'Sales'

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — Safety check: remove any feature columns missing from df
# ─────────────────────────────────────────────────────────────────────────────
for col in list(past_feature_cols):
    if col not in df.columns:
        print(f"WARNING: Past feature '{col}' not found in DataFrame — removing.")
        past_feature_cols.remove(col)

for col in list(future_feature_cols):
    if col not in df.columns:
        print(f"WARNING: Future feature '{col}' not found in DataFrame — removing.")
        future_feature_cols.remove(col)

print(f"\nPast feature count:   {len(past_feature_cols)}")
print(f"Future feature count: {len(future_feature_cols)}")
print(f"\nPast feature columns:\n  {past_feature_cols}")
print(f"\nFuture feature columns:\n  {future_feature_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — Normalize all features independently
#
# WHY MinMaxScaler for neural networks:
# LSTMs use sigmoid/tanh activations that saturate near ±1.
# Features with raw values in the thousands (e.g., Sales = 6000) would
# push activations into saturation zones, killing the gradient signal.
# MinMaxScaler maps all values to [0, 1], keeping activations in the
# sensitive, linear region where gradients flow cleanly.
#
# WHY separate scalers for past, future, and target:
# We need to inverse-transform the target independently to get back real
# sales numbers in units for evaluation (MAE, RMSE, MAPE).
# ─────────────────────────────────────────────────────────────────────────────
past_scaler   = MinMaxScaler()
future_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

df_past_scaled   = pd.DataFrame(
    past_scaler.fit_transform(df[past_feature_cols]),
    columns=past_feature_cols
)
df_future_scaled = pd.DataFrame(
    future_scaler.fit_transform(df[future_feature_cols]),
    columns=future_feature_cols
)
target_scaled = target_scaler.fit_transform(df[[target_col]].values)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15 — Build sliding-window sequences
#
# For each starting index i in the cleaned df:
#   X_past[i]   = rows [i .. i+lookback_days-1]   of past features  → shape (30, 20)
#   X_future[i] = rows [i+lookback_days .. i+lookback_days+forecast_days-1]
#                 of future features                                 → shape (7, 18)
#   y[i]        = rows [i+lookback_days .. i+lookback_days+forecast_days-1]
#                 of scaled Sales                                    → shape (7,)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding sliding-window sequences...")
X_past_list, X_future_list, y_list = [], [], []

for i in range(len(df) - lookback_days - forecast_days + 1):
    X_past_list.append(df_past_scaled.iloc[i : i + lookback_days].values)
    X_future_list.append(df_future_scaled.iloc[i + lookback_days : i + lookback_days + forecast_days].values)
    y_list.append(target_scaled[i + lookback_days : i + lookback_days + forecast_days].flatten())

X_past   = np.array(X_past_list)
X_future = np.array(X_future_list)
y        = np.array(y_list)

print(f"X_past shape:   {X_past.shape}")
print(f"X_future shape: {X_future.shape}")
print(f"y shape:        {y.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 16 — Chronological 80/20 train/test split
#
# WHY CHRONOLOGICAL (not random):
# Random splitting would mix future dates into the training set.
# The model would learn from data it would never have seen in real deployment.
# This produces deceptively low test errors — a phenomenon called look-ahead bias.
# ─────────────────────────────────────────────────────────────────────────────
split_idx = int(len(X_past) * 0.80)

X_past_train   = X_past[:split_idx]
X_future_train = X_future[:split_idx]
y_train        = y[:split_idx]

X_past_test    = X_past[split_idx:]
X_future_test  = X_future[split_idx:]
y_test         = y[split_idx:]

print(f"\nTrain shapes: X_past: {X_past_train.shape} | X_future: {X_future_train.shape} | y: {y_train.shape}")
print(f"Test  shapes: X_past: {X_past_test.shape}  | X_future: {X_future_test.shape}  | y: {y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 17 — Save arrays
# ─────────────────────────────────────────────────────────────────────────────
out_npz = "data/real/processed/rossmann_store_1_sequences_v2.npz"
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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 18 — Save scalers
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(past_scaler,   "models/real/rossmann_v2_past_feature_scaler.pkl")
joblib.dump(future_scaler, "models/real/rossmann_v2_future_feature_scaler.pkl")
joblib.dump(target_scaler, "models/real/rossmann_v2_target_scaler.pkl")
print("Saved scalers to models/real/")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 20 — Save feature list
# ─────────────────────────────────────────────────────────────────────────────
feature_report = "reports/real/rossmann_store_1_v2_feature_list.txt"
with open(feature_report, "w") as f:
    f.write("Rossmann Store 1 — v2 Feature List\n")
    f.write("====================================\n\n")
    f.write(f"Past feature count:   {len(past_feature_cols)}\n")
    f.write(f"Future feature count: {len(future_feature_cols)}\n\n")
    f.write("Past features:\n")
    for i, col in enumerate(past_feature_cols, 1):
        f.write(f"  {i:02d}. {col}\n")
    f.write("\nFuture features:\n")
    for i, col in enumerate(future_feature_cols, 1):
        f.write(f"  {i:02d}. {col}\n")
    f.write("\nFeatures in Past but NOT in Future (leak risk or unknown ahead):\n")
    past_only = [c for c in past_feature_cols if c not in future_feature_cols]
    for col in past_only:
        f.write(f"  - {col}\n")

print(f"Saved feature list to {feature_report}")
print("\nPhase 2.6 — Enhanced preprocessing complete.")
