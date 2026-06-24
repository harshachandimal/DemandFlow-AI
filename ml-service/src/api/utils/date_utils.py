import pandas as pd

def get_day_of_week(dt: pd.Timestamp) -> int:
    """Returns Rossmann DayOfWeek encoding: Monday = 1, ..., Sunday = 7"""
    return dt.weekday() + 1

def get_weekday_label(day_of_week: int) -> str:
    """Returns weekday label"""
    labels = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    return labels.get(day_of_week, "")

def is_weekend(day_of_week: int) -> int:
    """Returns 1 if weekend else 0"""
    return 1 if day_of_week in [6, 7] else 0

def generate_future_dates(latest_date: pd.Timestamp, forecast_days: int, forecast_open_days_only: bool, closes_on_sundays: bool) -> list[pd.Timestamp]:
    future_dates = []
    current_dt = latest_date
    while len(future_dates) < forecast_days:
        current_dt += pd.Timedelta(days=1)
        if forecast_open_days_only and closes_on_sundays and current_dt.weekday() == 6:
            continue  # Skip Sundays
        future_dates.append(current_dt)
    return future_dates
