import requests
import sys

BASE_URL = "http://127.0.0.1:8000"
all_passed = True

def print_result(name, condition, error_msg=""):
    global all_passed
    if condition:
        print(f"PASS - {name}")
    else:
        print(f"FAIL - {name}")
        if error_msg:
            print(f"       Details: {error_msg}")
        all_passed = False

def test_root():
    try:
        res = requests.get(f"{BASE_URL}/")
        data = res.json()
        condition = (res.status_code == 200 and
                     data.get("service") == "DemandFlow AI ML Service" and
                     data.get("status") == "running")
        print_result("GET /", condition)
    except Exception as e:
        print_result("GET /", False, str(e))

def test_health():
    try:
        res = requests.get(f"{BASE_URL}/health")
        data = res.json()
        condition = (res.status_code == 200 and
                     data.get("status") == "ok" and
                     data.get("model_loaded") is True)
        print_result("GET /health", condition)
    except Exception as e:
        print_result("GET /health", False, str(e))

def test_model_info():
    try:
        res = requests.get(f"{BASE_URL}/model-info")
        data = res.json()
        condition = (res.status_code == 200 and
                     data.get("model_name") == "RossmannEnhancedFutureAwareLSTM" and
                     data.get("model_version") == "v2" and
                     data.get("store_id") == 1 and
                     data.get("mape") == 7.32 and
                     data.get("supports_business_insights") is True)
        print_result("GET /model-info", condition)
    except Exception as e:
        print_result("GET /model-info", False, str(e))

def test_forecast_without_stock():
    try:
        payload = {
            "forecast_days": 7,
            "forecast_open_days_only": True,
            "promo_dates": [],
            "school_holiday_dates": []
        }
        res = requests.post(f"{BASE_URL}/api/v1/forecast/store-1", json=payload)
        data = res.json()
        bi = data.get("business_insights", {})
        
        condition = (res.status_code == 200 and
                     data.get("forecast_days") == 7 and
                     len(data.get("forecast", [])) == 7 and
                     "business_insights" in data and
                     "Add current_stock" in bi.get("recommendation", "") and
                     bi.get("current_stock") is None and
                     bi.get("projected_stock_after_7_days") is None)
        print_result("forecast without stock", condition)
    except Exception as e:
        print_result("forecast without stock", False, str(e))

def test_forecast_with_stock():
    try:
        payload = {
            "forecast_days": 7,
            "forecast_open_days_only": True,
            "promo_dates": [],
            "school_holiday_dates": [],
            "current_stock": 18000,
            "unit_price": 12.5,
            "reorder_lead_time_days": 3,
            "safety_stock_percentage": 0.15
        }
        res = requests.post(f"{BASE_URL}/api/v1/forecast/store-1", json=payload)
        data = res.json()
        bi = data.get("business_insights", {})
        
        condition = (res.status_code == 200 and
                     bi.get("expected_revenue") is not None and
                     bi.get("projected_stock_after_7_days") is not None and
                     bi.get("stockout_risk") is not None and
                     isinstance(bi.get("reorder_needed"), bool) and
                     bi.get("recommended_reorder_quantity") is not None and
                     bi.get("recommendation") != "")
        print_result("forecast with stock and price", condition)
    except Exception as e:
        print_result("forecast with stock and price", False, str(e))

def test_invalid_forecast():
    try:
        payload = {
            "forecast_days": 14,
            "forecast_open_days_only": True,
            "promo_dates": [],
            "school_holiday_dates": []
        }
        res = requests.post(f"{BASE_URL}/api/v1/forecast/store-1", json=payload)
        data = res.json()
        
        condition = (res.status_code == 400 and
                     ("supports only 7-day forecasts" in data.get("detail", "").lower() or
                      "supports 7-day forecasts only" in data.get("detail", "").lower()))
        print_result("invalid 14-day forecast validation", condition)
    except Exception as e:
        print_result("invalid 14-day forecast validation", False, str(e))

if __name__ == "__main__":
    test_root()
    test_health()
    test_model_info()
    test_forecast_without_stock()
    test_forecast_with_stock()
    test_invalid_forecast()
    
    print("")
    if all_passed:
        print("All FastAPI tests passed.")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)
