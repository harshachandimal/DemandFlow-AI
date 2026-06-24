import pandas as pd

def apply_past_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Lag and rolling features
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

    # Sales momentum
    df["sales_vs_lag7"] = df["Sales"] - df["lag_7_sales"]

    # Drop rows with missing lag/rolling values
    lag_cols = ["lag_7_sales", "rolling_7_sales", "rolling_14_sales", "rolling_7_std_sales", "sales_vs_lag7"]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    
    return df
