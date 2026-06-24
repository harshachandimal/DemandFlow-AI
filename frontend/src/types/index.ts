export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_name?: string;
  timestamp?: string;
}

export interface ModelMetrics {
  mae: number;
  rmse: number;
  mape: number;
  r2?: number;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  store_id: number;
  features_used: string[];
  metrics: ModelMetrics;
  supports_business_insights: boolean;
  timestamp?: string;
}

export interface ForecastPayload {
  forecast_days: number;
  forecast_open_days_only: boolean;
  promo_dates: string[];
  school_holiday_dates: string[];
  current_stock: number;
  unit_price: number;
  reorder_lead_time_days: number;
  safety_stock_percentage: number;
}

export interface ForecastResult {
  date: string;
  predicted_sales: number;
  is_promo?: boolean;
  promo?: boolean;
  is_school_holiday?: boolean;
  school_holiday?: boolean;
  demand_level?: 'High' | 'Medium' | 'Low';
}

export interface ReorderRecommendation {
  reorder_needed: boolean;
  recommended_reorder_date: string | null;
  suggested_order_quantity?: number;
  recommended_reorder_quantity?: number;
}

export interface BusinessInsights {
  total_predicted_sales: number;
  average_predicted_sales: number;
  stockout_risk: 'high' | 'medium' | 'low' | 'High' | 'Medium' | 'Low';
  expected_revenue?: number;
  projected_revenue?: number;
  projected_stock_after_7_days?: number;
  projected_stock_after_forecast?: number;
  current_stock: number;
  reorder_needed: boolean;
  recommended_reorder_date?: string | null;
  recommended_reorder_quantity?: number;
  recommendation?: string;
}

export interface ForecastResponse {
  forecast: ForecastResult[];
  business_insights: BusinessInsights;
}

export interface ForecastLogSummary {
  id: number;
  store_id: number;
  total_predicted_sales: number;
  average_predicted_sales: number;
  stockout_risk: 'high' | 'medium' | 'low' | 'High' | 'Medium' | 'Low';
  reorder_needed: boolean | null;
  created_at: string;
}

export interface ForecastLog extends ForecastLogSummary {
  request_payload: ForecastPayload;
  response_payload: ForecastResponse;
}
