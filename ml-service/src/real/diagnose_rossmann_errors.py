"""
diagnose_rossmann_errors.py
===========================
Phase 2.4 — Worst-Prediction Diagnosis with Dates and Business Context

WHY ERROR DIAGNOSIS MATTERS IN ML
----------------------------------
A model's aggregate MAPE (7.76%) tells us the average story.
But business decisions live at the individual sample level.
When a model is badly wrong on a specific day, inventory planners
suffer stock-outs or wasteful overstock on THAT day.

Diagnosing the worst predictions helps us answer:
  • WHEN does the model fail? (Which calendar dates?)
  • WHY does it fail? (Promotions? Weekends? Unusual sales spikes?)
  • HOW OFTEN do certain calendar conditions trigger large errors?

These answers directly guide the next model improvement iteration.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports/real", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load the processed CSV (the cleaned real-data source of truth)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading processed CSV...")
df = pd.read_csv("data/real/processed/rossmann_store_1_processed.csv")

# STEP 2 — Parse Date and sort chronologically
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# STEP 3 — Recreate the SAME feature engineering used in preprocessing
# This is critical: we must reproduce the exact same dropna rows so that
# our row indices in df align perfectly with what the sequences saw.
df['lag_7_sales']     = df['Sales'].shift(7)
df['rolling_7_sales'] = df['Sales'].shift(1).rolling(window=7).mean()

# STEP 4 — Drop rows with missing lag/rolling values (mirrors preprocessing exactly)
df = df.dropna(subset=['lag_7_sales', 'rolling_7_sales']).reset_index(drop=True)
print(f"Rows after dropping NaN lag/rolling: {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Define lookback / forecast windows (must match preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
lookback_days = 30
forecast_days = 7

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Recreate mapping: (test_sample_index, forecast_day) → calendar date
#
# How sequence indexing works:
#   Global sequence index i means:
#     X_past  covers df rows  [i .. i + lookback_days - 1]  (30 rows)
#     y       covers df rows  [i + lookback_days .. i + lookback_days + forecast_days - 1]  (7 rows)
#
#   Chronological 80/20 split:
#     total sequences N = len(df) - lookback_days - forecast_days + 1
#     train_size        = int(N * 0.80)
#     test sample k     ↔ global index  train_size + k
#
#   For forecast day d  (1-indexed):
#     target row index in df = global_i + lookback_days + (d - 1)
# ─────────────────────────────────────────────────────────────────────────────
total_sequences = len(df) - lookback_days - forecast_days + 1
train_size      = int(total_sequences * 0.80)

print(f"Total sequences: {total_sequences}")
print(f"Train size: {train_size}  |  Test size: {total_sequences - train_size}")

def get_row_index(test_sample_k: int, forecast_day_d: int) -> int:
    """
    Given a test sample index (0-based) and a forecast day (1-based),
    return the row index in df that corresponds to that prediction target.
    """
    global_i   = train_size + test_sample_k          # global sequence index
    row_idx    = global_i + lookback_days + (forecast_day_d - 1)
    return row_idx

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Load detailed predictions (all test samples × all 7 forecast days)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading detailed predictions CSV...")
df_preds = pd.read_csv("reports/real/rossmann_store_1_detailed_predictions.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Select worst 20 rows by absolute_error
# ─────────────────────────────────────────────────────────────────────────────
df_worst = df_preds.sort_values(by='absolute_error', ascending=False).head(20).copy()
df_worst = df_worst.reset_index(drop=True)

print(f"\nWorst 20 predictions selected. Max error: {df_worst['absolute_error'].max():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — For each worst row, look up calendar date + business context
# ─────────────────────────────────────────────────────────────────────────────
context_cols = [
    'Date', 'DayOfWeek', 'Month', 'Day', 'Promo',
    'SchoolHoliday', 'IsWeekend', 'Sales', 'lag_7_sales', 'rolling_7_sales'
]

# Build list of context dicts
context_rows = []
for _, row in df_worst.iterrows():
    k = int(row['sample_index'])
    d = int(row['forecast_day'])
    
    row_idx = get_row_index(k, d)
    
    if row_idx >= len(df):
        # Safety guard — should not happen if shapes match
        print(f"  WARNING: row_idx {row_idx} out of bounds for sample {k}, day {d}")
        context_rows.append({c: None for c in context_cols})
        continue

    ctx = df.iloc[row_idx][context_cols].to_dict()
    context_rows.append(ctx)

df_context = pd.DataFrame(context_rows)

# Merge the original prediction columns with the context columns side-by-side
df_enriched = pd.concat([df_worst.reset_index(drop=True), df_context.reset_index(drop=True)], axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Save enriched worst errors
# ─────────────────────────────────────────────────────────────────────────────
out_enriched = "reports/real/rossmann_store_1_worst_predictions_with_context.csv"
df_enriched.to_csv(out_enriched, index=False)
print(f"\nSaved enriched worst predictions to: {out_enriched}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Print readable table for the worst 20 rows
# ─────────────────────────────────────────────────────────────────────────────
display_cols = [
    'sample_index', 'forecast_day', 'Date',
    'actual_sales', 'predicted_sales', 'absolute_error', 'percentage_error',
    'Promo', 'SchoolHoliday', 'DayOfWeek', 'IsWeekend',
    'lag_7_sales', 'rolling_7_sales'
]
print("\n--- Worst 20 Predictions with Business Context ---")
print(df_enriched[display_cols].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Build summary analysis using ALL test predictions (not just worst 20)
#
# WHY ALL predictions and not just worst 20?
# Calculating group averages on 20 rows would be statistically meaningless.
# We must attach context to ALL test predictions, then group by business fields.
# ─────────────────────────────────────────────────────────────────────────────
print("\nAttaching context to ALL test predictions for group analysis...")

all_contexts = []
for _, row in df_preds.iterrows():
    k = int(row['sample_index'])
    d = int(row['forecast_day'])
    row_idx = get_row_index(k, d)
    
    if row_idx >= len(df):
        all_contexts.append({c: None for c in context_cols})
        continue
    
    ctx = df.iloc[row_idx][context_cols].to_dict()
    all_contexts.append(ctx)

df_all_context = pd.DataFrame(all_contexts)
df_full = pd.concat([df_preds.reset_index(drop=True), df_all_context.reset_index(drop=True)], axis=1)

# Drop any rows where context lookup failed (safety)
df_full = df_full.dropna(subset=['Date'])

# ── Promo analysis
promo_err = df_full.groupby('Promo')['absolute_error'].mean()
promo_labels = {0: 'No Promo', 1: 'Promo Active'}

# ── Weekend analysis
weekend_err = df_full.groupby('IsWeekend')['absolute_error'].mean()
weekend_labels = {0: 'Weekday', 1: 'Weekend'}

# ── DayOfWeek analysis (Rossmann: 1=Mon … 7=Sun)
dow_err = df_full.groupby('DayOfWeek')['absolute_error'].mean()

# ── Month analysis
month_err = df_full.groupby('Month')['absolute_error'].mean()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — Save error context summary
# ─────────────────────────────────────────────────────────────────────────────
summary_path = "reports/real/rossmann_store_1_error_context_summary.txt"
with open(summary_path, 'w') as f:
    f.write("Rossmann Store 1 — Error Context Summary\n")
    f.write("=========================================\n\n")

    f.write("Average Absolute Error by Promo Status:\n")
    for k, v in promo_err.items():
        f.write(f"  {promo_labels.get(k, k)}: {v:.2f}\n")

    f.write("\nAverage Absolute Error by Weekend vs Weekday:\n")
    for k, v in weekend_err.items():
        f.write(f"  {weekend_labels.get(k, k)}: {v:.2f}\n")

    f.write("\nAverage Absolute Error by Day of Week (1=Mon, 7=Sun):\n")
    for k, v in dow_err.items():
        f.write(f"  Day {int(k)}: {v:.2f}\n")

    f.write("\nAverage Absolute Error by Month:\n")
    for k, v in month_err.items():
        f.write(f"  Month {int(k):02d}: {v:.2f}\n")

print(f"\nSaved error context summary to: {summary_path}")

# Print summary to console too
print("\n--- Error Context Summary ---")
print("By Promo Status:")
for k, v in promo_err.items():
    print(f"  {promo_labels.get(k, k)}: {v:.2f}")

print("\nBy Weekend vs Weekday:")
for k, v in weekend_err.items():
    print(f"  {weekend_labels.get(k, k)}: {v:.2f}")

print("\nBy Day of Week:")
for k, v in dow_err.items():
    print(f"  Day {int(k)}: {v:.2f}")

print("\nBy Month:")
for k, v in month_err.items():
    print(f"  Month {int(k):02d}: {v:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — Create charts
# ─────────────────────────────────────────────────────────────────────────────

# Chart A — Error by Promo
fig, ax = plt.subplots(figsize=(7, 5))
promo_x = [promo_labels.get(k, k) for k in promo_err.index]
bars = ax.bar(promo_x, promo_err.values, color=['steelblue', 'tomato'])
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=11)
ax.set_title("Avg Absolute Error: Promo vs No Promo (Store 1)", fontsize=13)
ax.set_ylabel("Mean Absolute Error")
ax.set_ylim(0, promo_err.max() * 1.25)
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_error_by_promo.png")
plt.close()

# Chart B — Error by Weekend
fig, ax = plt.subplots(figsize=(7, 5))
weekend_x = [weekend_labels.get(k, k) for k in weekend_err.index]
bars = ax.bar(weekend_x, weekend_err.values, color=['mediumseagreen', 'darkorange'])
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=11)
ax.set_title("Avg Absolute Error: Weekday vs Weekend (Store 1)", fontsize=13)
ax.set_ylabel("Mean Absolute Error")
ax.set_ylim(0, weekend_err.max() * 1.25)
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_error_by_weekend.png")
plt.close()

# Chart C — Error by Day of Week
fig, ax = plt.subplots(figsize=(9, 5))
dow_labels = [f"Day {int(d)}" for d in dow_err.index]
bars = ax.bar(dow_labels, dow_err.values, color='mediumpurple')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=10)
ax.set_title("Avg Absolute Error by Day of Week (1=Mon, 7=Sun) — Store 1", fontsize=12)
ax.set_ylabel("Mean Absolute Error")
ax.set_ylim(0, dow_err.max() * 1.25)
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_error_by_day_of_week.png")
plt.close()

# Chart D — Error by Month
fig, ax = plt.subplots(figsize=(11, 5))
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
month_x = [month_names.get(int(m), str(int(m))) for m in month_err.index]
bars = ax.bar(month_x, month_err.values, color='cornflowerblue')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=9)
ax.set_title("Avg Absolute Error by Month — Store 1", fontsize=13)
ax.set_ylabel("Mean Absolute Error")
ax.set_ylim(0, month_err.max() * 1.25)
plt.tight_layout()
plt.savefig("reports/real/rossmann_store_1_error_by_month.png")
plt.close()

print("\nAll charts saved to reports/real/")
print("\nPhase 2.5 — Diagnosis complete.")
