from pydantic import BaseModel
from typing import List, Optional

class ForecastRequest(BaseModel):
    forecast_days: int = 7
    forecast_open_days_only: bool = True
    promo_dates: List[str] = []
    school_holiday_dates: List[str] = []
    current_stock: Optional[int] = None
    unit_price: Optional[float] = None
    reorder_lead_time_days: int = 3
    safety_stock_percentage: float = 0.15

class ForecastDay(BaseModel):
    date: str
    day_of_week: int
    weekday: str
    is_weekend: int
    promo: int
    school_holiday: int
    predicted_sales: int
    demand_level: str

class BusinessInsights(BaseModel):
    total_predicted_sales: int
    average_predicted_sales: int
    expected_revenue: Optional[float]
    current_stock: Optional[int]
    projected_stock_after_7_days: Optional[int]
    stockout_risk: Optional[str]
    recommended_reorder_quantity: Optional[int]
    reorder_needed: Optional[bool]
    recommendation: str

class ForecastResponse(BaseModel):
    store_id: int
    model_name: str
    model_version: str
    forecast_mode: str
    latest_historical_date: str
    forecast_days: int
    total_predicted_sales: int
    average_predicted_sales: int
    highest_demand_day: str
    lowest_demand_day: str
    forecast: List[ForecastDay]
    business_insights: BusinessInsights
