import os
import numpy as np
import pandas as pd
import torch
from .schemas import ForecastRequest, ForecastResponse, ForecastDay
from .utils.date_utils import get_day_of_week, get_weekday_label, is_weekend, generate_future_dates
from .utils.feature_engineering import apply_past_feature_engineering
from .business_insights import generate_business_insights

def generate_forecast(request: ForecastRequest, model, past_scaler, future_scaler, target_scaler) -> ForecastResponse:
    if request.forecast_days != 7:
        raise ValueError("This model currently supports 7-day forecasts only.")
        
    data_path = "data/real/processed/rossmann_store_1_processed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df_engineered = apply_past_feature_engineering(df)

    lookback_days = 30
    forecast_days = request.forecast_days

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

    df_past = df_engineered.tail(lookback_days).copy()
    latest_date = df_engineered["Date"].max()

    sundays_in_history = len(df_engineered[df_engineered["DayOfWeek"] == 7])
    closes_on_sundays = (sundays_in_history == 0)

    future_dates = generate_future_dates(
        latest_date, forecast_days, request.forecast_open_days_only, closes_on_sundays
    )

    promo_dates_set = set(pd.to_datetime(request.promo_dates).date) if request.promo_dates else set()
    school_holiday_dates_set = set(pd.to_datetime(request.school_holiday_dates).date) if request.school_holiday_dates else set()

    future_rows = []
    for future_dt in future_dates:
        row = {}
        row["Date"]          = future_dt
        row["DayOfWeek"]     = get_day_of_week(future_dt)
        row["Month"]         = future_dt.month
        row["Day"]           = future_dt.day
        row["IsWeekend"]     = is_weekend(row["DayOfWeek"])
        row["Open"]          = 0 if (closes_on_sundays and row["DayOfWeek"] == 7) else 1

        row["Promo"]         = 1 if future_dt.date() in promo_dates_set else 0
        row["SchoolHoliday"] = 1 if future_dt.date() in school_holiday_dates_set else 0

        row["is_month_start"] = int(future_dt.is_month_start)
        row["is_month_end"]   = int(future_dt.is_month_end)
        row["is_april"]       = int(future_dt.month == 4)

        row["promo_schoolholiday"]   = row["Promo"] * row["SchoolHoliday"]
        row["weekend_schoolholiday"] = row["IsWeekend"] * row["SchoolHoliday"]
        row["promo_weekend"]         = row["Promo"] * row["IsWeekend"]

        lag_date = future_dt - pd.Timedelta(days=7)
        lag_match = df_engineered[df_engineered["Date"] == lag_date]
        if not lag_match.empty:
            row["lag_7_sales"] = float(lag_match["Sales"].values[0])
        else:
            row["lag_7_sales"] = float(df_engineered["Sales"].tail(7).mean())

        row["rolling_7_sales"]    = float(df_engineered["rolling_7_sales"].iloc[-1])
        row["rolling_14_sales"]   = float(df_engineered["rolling_14_sales"].iloc[-1])
        row["rolling_7_std_sales"] = float(df_engineered["rolling_7_std_sales"].iloc[-1])

        future_rows.append(row)

    df_future = pd.DataFrame(future_rows)

    MAX_PROMO_TIMING = 10

    last_days_since = int(df_engineered["days_since_last_promo"].iloc[-1])
    future_days_since = []
    counter = last_days_since
    for _, row in df_future.iterrows():
        counter = 0 if row["Promo"] == 1 else counter + 1
        future_days_since.append(min(counter, MAX_PROMO_TIMING))
    df_future["days_since_last_promo"] = future_days_since

    future_days_until = []
    counter = MAX_PROMO_TIMING
    for _, row in df_future.iloc[::-1].iterrows():
        counter = 0 if row["Promo"] == 1 else counter + 1
        future_days_until.append(min(counter, MAX_PROMO_TIMING))
    future_days_until.reverse()
    df_future["days_until_next_promo"] = future_days_until

    X_past_raw = df_past[past_feature_cols].values
    X_future_raw = df_future[future_feature_cols].values

    df_past_for_scale   = pd.DataFrame(X_past_raw,   columns=past_feature_cols)
    df_future_for_scale = pd.DataFrame(X_future_raw, columns=future_feature_cols)

    X_past_scaled   = past_scaler.transform(df_past_for_scale)
    X_future_scaled = future_scaler.transform(df_future_for_scale)

    X_past_scaled   = np.clip(X_past_scaled,   0.0, 1.0)
    X_future_scaled = np.clip(X_future_scaled, 0.0, 1.0)

    X_past_t   = torch.tensor(X_past_scaled[np.newaxis, :, :],   dtype=torch.float32)
    X_future_t = torch.tensor(X_future_scaled[np.newaxis, :, :], dtype=torch.float32)

    with torch.no_grad():
        predictions_scaled = model(X_past_t, X_future_t)

    predictions_scaled_np = predictions_scaled.numpy()
    predicted_sales = target_scaler.inverse_transform(predictions_scaled_np).flatten()

    for i, row in enumerate(future_rows):
        if row["Open"] == 0:
            predicted_sales[i] = 0.0

    predicted_sales_rounded = np.round(predicted_sales).astype(int)

    forecast_days_list = []
    for i, future_dt in enumerate(future_dates):
        sales = int(predicted_sales_rounded[i])
        if sales >= 4500:
            demand_level = "high"
        elif sales >= 3500:
            demand_level = "medium"
        else:
            demand_level = "low"
            
        forecast_days_list.append(ForecastDay(
            date=str(future_dt.date()),
            day_of_week=int(df_future["DayOfWeek"].iloc[i]),
            weekday=get_weekday_label(df_future["DayOfWeek"].iloc[i]),
            is_weekend=int(df_future["IsWeekend"].iloc[i]),
            promo=int(df_future["Promo"].iloc[i]),
            school_holiday=int(df_future["SchoolHoliday"].iloc[i]),
            predicted_sales=sales,
            demand_level=demand_level
        ))

    total_sales = int(predicted_sales_rounded.sum())
    avg_sales = int(predicted_sales_rounded.mean())
    peak_idx = int(np.argmax(predicted_sales_rounded))
    trough_idx = int(np.argmin(predicted_sales_rounded))

    business_insights_data = generate_business_insights(
        predictions=[int(p) for p in predicted_sales_rounded],
        current_stock=request.current_stock,
        unit_price=request.unit_price,
        reorder_lead_time_days=request.reorder_lead_time_days,
        safety_stock_percentage=request.safety_stock_percentage
    )

    return ForecastResponse(
        store_id=1,
        model_name="RossmannEnhancedFutureAwareLSTM",
        model_version="v2",
        forecast_mode="open days only" if request.forecast_open_days_only else "calendar days",
        latest_historical_date=str(latest_date.date()),
        forecast_days=forecast_days,
        total_predicted_sales=total_sales,
        average_predicted_sales=avg_sales,
        highest_demand_day=str(future_dates[peak_idx].date()),
        lowest_demand_day=str(future_dates[trough_idx].date()),
        forecast=forecast_days_list,
        business_insights=business_insights_data
    )
