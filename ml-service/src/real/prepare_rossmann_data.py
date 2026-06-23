"""
prepare_rossmann_data.py
========================
Phase 2.1 – Prepare the Real Rossmann Store Sales Dataset

This is the very first step of working with real-world data.  We move from
the clean, synthetic sample_sales.csv (Phase 1) to the much noisier, richer
Rossmann dataset that contains 1,115 stores across Germany over ~2.5 years.

What this script does
---------------------
1.  Loads train.csv (daily sales records) and store.csv (static store metadata).
2.  Merges them on the Store column so each row has both sales info and store info.
3.  Filters down to **Store 1 only** so we can build a clean, understandable
    pipeline before scaling to multiple stores.
4.  Removes closed-store rows (Open == 0) where Sales is always 0.
5.  Drops the Customers column to avoid target leakage.
6.  Handles missing values and encodes categorical columns.
7.  Saves a clean CSV and generates four EDA charts.

Learning goals
--------------
•  Understand what target leakage is and why Customers must be excluded.
•  See how real data differs from synthetic data (missing values, categoricals).
•  Learn why closed-store rows are removed for the first model.
•  Practice standard data-preparation patterns used in industry.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt


def main():
    # -----------------------------------------------------------------------
    # 1. CHECK THAT RAW FILES EXIST
    # -----------------------------------------------------------------------
    train_path = 'data/real/raw/train.csv'
    store_path = 'data/real/raw/store.csv'

    missing = []
    if not os.path.exists(train_path):
        missing.append(train_path)
    if not os.path.exists(store_path):
        missing.append(store_path)

    if missing:
        print("ERROR — Missing files:")
        for f in missing:
            print(f"  • {f}")
        print("\nPlease download the Rossmann Store Sales dataset and place")
        print("train.csv and store.csv inside data/real/raw/")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. LOAD RAW CSVs
    # -----------------------------------------------------------------------
    print("Loading raw datasets...")
    train_df = pd.read_csv(train_path, low_memory=False)
    store_df = pd.read_csv(store_path)

    # -----------------------------------------------------------------------
    # 3. PRINT RAW SHAPES
    # -----------------------------------------------------------------------
    # train.csv has one row per store per day (1,115 stores × ~940 days).
    # store.csv has one row per store (static metadata like StoreType).
    print(f"\ntrain.csv shape : {train_df.shape}  ({train_df.shape[0]:,} rows x {train_df.shape[1]} columns)")
    print(f"train.csv columns: {list(train_df.columns)}")
    print(f"\nstore.csv shape : {store_df.shape}  ({store_df.shape[0]:,} rows x {store_df.shape[1]} columns)")
    print(f"store.csv columns: {list(store_df.columns)}")

    # -----------------------------------------------------------------------
    # 4. PARSE DATE
    # -----------------------------------------------------------------------
    # The Date column is a string like "2015-07-31".  We convert it to a
    # proper datetime so pandas can sort, filter, and extract calendar info.
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    # -----------------------------------------------------------------------
    # 5. MERGE TRAIN + STORE
    # -----------------------------------------------------------------------
    # A SQL-style LEFT JOIN on the Store column.  Every daily-sales row gains
    # the static store attributes (StoreType, Assortment, CompetitionDistance…).
    merged_df = train_df.merge(store_df, on='Store', how='left')

    # -----------------------------------------------------------------------
    # 6. SORT BY DATE
    # -----------------------------------------------------------------------
    # Time-series order is critical.  We ensure rows are ascending by date
    # so that our later sliding-window sequences respect causality.
    merged_df = merged_df.sort_values(by='Date').reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 7. PRINT INITIAL DATASET INFORMATION
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("MERGED DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"Merged shape       : {merged_df.shape}")
    print(f"Date range         : {merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}")
    print(f"Unique stores      : {merged_df['Store'].nunique()}")

    print(f"\n--- Missing Values Per Column ---")
    missing_counts = merged_df.isnull().sum()
    for col, cnt in missing_counts.items():
        if cnt > 0:
            print(f"  {col:35s}: {cnt:>7,}  ({cnt/len(merged_df)*100:.1f}%)")
    if missing_counts.sum() == 0:
        print("  (none)")

    print(f"\n--- First 5 Rows ---")
    print(merged_df.head().to_string(index=False))

    print(f"\n--- Sales Statistics (all stores) ---")
    print(f"  Mean : {merged_df['Sales'].mean():,.1f}")
    print(f"  Min  : {merged_df['Sales'].min():,.0f}")
    print(f"  Max  : {merged_df['Sales'].max():,.0f}")

    # -----------------------------------------------------------------------
    # 8. FILTER TO STORE 1 ONLY
    # -----------------------------------------------------------------------
    # WHY ONE STORE FIRST?
    # --------------------
    # Multi-store forecasting introduces a massive jump in complexity:
    #   • Each store has its own demand patterns, local holidays, and
    #     competition dynamics.
    #   • The number of training samples increases 1,115×, requiring longer
    #     training and more memory.
    #   • Debugging and visualising results across many stores is hard.
    #
    # By starting with a single store, we can:
    #   1. Validate the entire data → sequence → LSTM pipeline end-to-end.
    #   2. Deeply inspect the charts and predictions for one store.
    #   3. Build intuition about the data quality before scaling.
    #
    # Once the pipeline works for Store 1, extending it to all stores is
    # mostly a matter of looping or reshaping the inputs.
    # -----------------------------------------------------------------------
    store_id = 1
    df = merged_df[merged_df['Store'] == store_id].copy()
    print(f"\nFiltered to Store {store_id}: {len(df)} rows")

    # -----------------------------------------------------------------------
    # 9. REMOVE CLOSED-STORE ROWS (Open == 0)
    # -----------------------------------------------------------------------
    # WHY REMOVE CLOSED DAYS?
    # -----------------------
    # When a store is closed, Sales is always 0.  If we include these rows:
    #   • The model will learn a trivial pattern: "if closed → predict 0".
    #   • The average Sales will be dragged down, biasing all predictions.
    #   • More importantly, for *forecasting* we almost always know in advance
    #     whether the store will be open (it's a scheduled decision).
    #
    # In a real deployment, the application layer would simply output 0 for
    # closed days and only invoke the model for open days.  Our training
    # data should therefore only contain open-day patterns.
    # -----------------------------------------------------------------------
    rows_before = len(df)
    df = df[df['Open'] == 1].copy()
    rows_removed = rows_before - len(df)
    print(f"Removed {rows_removed} closed-store rows (Open==0). Remaining: {len(df)}")

    # -----------------------------------------------------------------------
    # 10. DROP THE CUSTOMERS COLUMN
    # -----------------------------------------------------------------------
    # WHAT IS TARGET LEAKAGE?
    # -----------------------
    # Target leakage occurs when information that would not be available at
    # prediction time leaks into the training features.
    #
    # In the Rossmann dataset, the Customers column records how many customers
    # visited the store on that day.  Customer count is strongly correlated
    # with Sales (more customers → more sales), but it is NOT known in advance.
    # You cannot know how many customers will walk in next Tuesday until
    # next Tuesday actually happens.
    #
    # If we train with Customers as a feature:
    #   • The model will rely heavily on it (it's an almost-perfect predictor).
    #   • At inference time, Customers won't exist → predictions collapse.
    #   • The model never learns the real underlying drivers (promos, weekday…).
    #
    # This is one of the most common ML mistakes in Kaggle competitions and
    # in real-world data science projects.  Always ask: "Would I know this
    # value BEFORE making the prediction?"
    # -----------------------------------------------------------------------
    if 'Customers' in df.columns:
        df = df.drop(columns=['Customers'])
        print("Dropped 'Customers' column (target leakage risk).")

    # -----------------------------------------------------------------------
    # 11. SELECT USEFUL COLUMNS
    # -----------------------------------------------------------------------
    # We keep only the columns that are either:
    #   • Known in advance (Date, DayOfWeek, Promo, StateHoliday…)
    #   • Static store attributes (StoreType, Assortment, CompetitionDistance…)
    #   • The target variable (Sales)
    # -----------------------------------------------------------------------
    columns_to_keep = [
        'Date',
        'Store',
        'DayOfWeek',
        'Sales',
        'Open',
        'Promo',
        'StateHoliday',
        'SchoolHoliday',
        'StoreType',
        'Assortment',
        'CompetitionDistance',
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'Promo2',
        'Promo2SinceWeek',
        'Promo2SinceYear',
    ]

    # Only keep columns that actually exist in the dataframe
    columns_present = [c for c in columns_to_keep if c in df.columns]
    df = df[columns_present].copy()
    print(f"Selected {len(columns_present)} columns.")

    # -----------------------------------------------------------------------
    # 12. HANDLE MISSING VALUES
    # -----------------------------------------------------------------------
    # Competition and Promo2 columns have legitimate missing values:
    #   • CompetitionDistance: a few stores have no nearby competitor.
    #     → fill with the median (a robust central tendency that isn't
    #       distorted by outliers the way the mean would be).
    #   • CompetitionOpenSinceMonth/Year: the month/year the competitor opened.
    #     → fill with 0 to indicate "no competitor opening recorded".
    #   • Promo2SinceWeek/Year: when the store joined a recurring promo program.
    #     → fill with 0 to indicate "not participating in Promo2".
    #
    # WHY NOT DROP ROWS?
    #   In time-series, dropping rows creates gaps in the timeline.  Gaps
    #   break sliding-window sequences and destroy temporal continuity.
    #   Imputation (filling) preserves the date sequence.
    # -----------------------------------------------------------------------
    if 'CompetitionDistance' in df.columns:
        median_cd = df['CompetitionDistance'].median()
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(median_cd)
        print(f"Filled CompetitionDistance NaNs with median ({median_cd:.0f}).")

    fill_zero_cols = [
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'Promo2SinceWeek',
        'Promo2SinceYear',
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(0)
                print(f"Filled {col} NaNs with 0 ({n_missing} values).")

    # -----------------------------------------------------------------------
    # 13. ENCODE CATEGORICAL COLUMNS
    # -----------------------------------------------------------------------
    # StateHoliday can be '0', 'a', 'b', or 'c' (different holiday types).
    # StoreType can be 'a', 'b', 'c', or 'd'.
    # Assortment can be 'a', 'b', or 'c'.
    #
    # Neural networks operate on numbers, not letters.  One-hot encoding
    # (pd.get_dummies) converts each category into a separate binary column:
    #   StoreType_a = 1, StoreType_b = 0, StoreType_c = 0, StoreType_d = 0
    #
    # This avoids the ordinal assumption (a < b < c) that would be implied
    # by mapping categories to integers 0, 1, 2.
    # -----------------------------------------------------------------------
    categorical_cols = ['StateHoliday', 'StoreType', 'Assortment']
    existing_cats = [c for c in categorical_cols if c in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, dtype=int)
        print(f"One-hot encoded: {existing_cats}")

    # Re-sort after all transformations (paranoia check)
    df = df.sort_values(by='Date').reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 14. CREATE EXTRA DATE FEATURES
    # -----------------------------------------------------------------------
    # Extracting calendar components from the Date column gives the model
    # explicit access to yearly seasonality (Month), long-term trends (Year),
    # within-month patterns (Day), and the weekend effect (IsWeekend).
    #
    # Why not rely on DayOfWeek alone?
    #   DayOfWeek tells the model which day of the week it is (1-7), but it
    #   doesn't convey monthly or yearly cycles.  December sales may be very
    #   different from July sales due to holiday shopping.  Adding Month and
    #   Year explicitly saves the model from having to infer these cycles
    #   from the raw date values.
    #
    # IsWeekend is a simple binary flag: 1 if Saturday (6) or Sunday (7).
    #   The Rossmann DayOfWeek column uses 1=Mon ... 7=Sun.
    # -----------------------------------------------------------------------
    df['Month'] = df['Date'].dt.month
    df['Year']  = df['Date'].dt.year
    df['Day']   = df['Date'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [6, 7] else 0)
    print(f"Created date features: Month, Year, Day, IsWeekend")

    # -----------------------------------------------------------------------
    # 15. SAVE PROCESSED DATASET
    # -----------------------------------------------------------------------
    os.makedirs('data/real/processed', exist_ok=True)
    output_path = 'data/real/processed/rossmann_store_1_processed.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed dataset to {output_path}")

    # -----------------------------------------------------------------------
    # 15. PRINT PROCESSED DATASET INFORMATION
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PROCESSED DATASET (Store 1, Open Days Only)")
    print(f"{'='*60}")
    print(f"Shape              : {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>2}. {col}")

    print(f"\n--- First 5 Processed Rows ---")
    print(df.head().to_string(index=False))

    remaining_missing = df.isnull().sum()
    total_missing = remaining_missing.sum()
    print(f"\n--- Missing Values After Cleaning ---")
    if total_missing == 0:
        print("  None — all clean!")
    else:
        for col, cnt in remaining_missing.items():
            if cnt > 0:
                print(f"  {col}: {cnt}")

    print(f"\n--- Sales Statistics (Store 1, Open Days) ---")
    print(f"  Average Sales : {df['Sales'].mean():,.1f}")
    print(f"  Min Sales     : {df['Sales'].min():,.0f}")
    print(f"  Max Sales     : {df['Sales'].max():,.0f}")
    print(f"  Std Dev       : {df['Sales'].std():,.1f}")

    # -----------------------------------------------------------------------
    # 16. EDA CHARTS
    # -----------------------------------------------------------------------
    os.makedirs('reports/real', exist_ok=True)
    chart_color = 'royalblue'

    # ----- Chart 1: Sales Over Time -----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['Date'], df['Sales'], linewidth=0.5, color=chart_color, alpha=0.8)
    ax.set_title(f'Store {store_id} — Daily Sales Over Time', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales (€)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = 'reports/real/rossmann_store_1_sales_over_time.png'
    plt.savefig(path1, dpi=120)
    plt.close()
    print(f"\nChart saved: {path1}")

    # ----- Chart 2: Sales by Day of Week -----
    fig, ax = plt.subplots(figsize=(8, 5))
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_avg = df.groupby('DayOfWeek')['Sales'].mean()
    ax.bar(day_avg.index, day_avg.values, color=chart_color, alpha=0.8)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(day_names)
    ax.set_title(f'Store {store_id} — Average Sales by Day of Week', fontsize=14)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Sales (€)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path2 = 'reports/real/rossmann_store_1_sales_by_day_of_week.png'
    plt.savefig(path2, dpi=120)
    plt.close()
    print(f"Chart saved: {path2}")

    # ----- Chart 3: Promotion Effect -----
    fig, ax = plt.subplots(figsize=(6, 5))
    promo_avg = df.groupby('Promo')['Sales'].mean()
    labels = ['No Promo', 'Promo']
    colors = ['grey', 'tomato']
    ax.bar(labels, promo_avg.values, color=colors, alpha=0.8)
    ax.set_title(f'Store {store_id} — Average Sales: Promo vs No Promo', fontsize=14)
    ax.set_ylabel('Average Sales (€)')
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate the values on top of the bars
    for i, v in enumerate(promo_avg.values):
        ax.text(i, v + 50, f'{v:,.0f}', ha='center', fontweight='bold')
    plt.tight_layout()
    path3 = 'reports/real/rossmann_store_1_promo_effect.png'
    plt.savefig(path3, dpi=120)
    plt.close()
    print(f"Chart saved: {path3}")

    # ----- Chart 4: Monthly Sales -----
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_avg = df.groupby('Month')['Sales'].mean()
    ax.bar(monthly_avg.index, monthly_avg.values, color=chart_color, alpha=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_title(f'Store {store_id} -- Average Sales by Month', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Sales')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path4 = 'reports/real/rossmann_store_1_monthly_sales.png'
    plt.savefig(path4, dpi=120)
    plt.close()
    print(f"Chart saved: {path4}")

    print(f"\n{'='*60}")
    print("Phase 2.1 complete — Real dataset is prepared.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
