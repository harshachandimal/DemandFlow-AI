# DemandFlow AI

## Phase 4.1: Laravel Backend Integration

This phase integrates the FastAPI Machine Learning service with a new Laravel backend. The Laravel application serves as an API gateway for future frontend clients and manages request validation and response logging.

### Prerequisites
- Python 3.10+
- PHP 8.2+
- Composer

### Running the Services

**Note**: The FastAPI ML service must be running before the Laravel backend attempts to call its endpoints.

1. **Start the FastAPI ML Service**
   Open a terminal and run the ML service on port 8001 (to avoid conflict with Laravel on 8000).
   ```bash
   cd ml-service
   python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
   ```

2. **Start the Laravel Backend**
   Open a separate terminal, ensure `.env` points to `DEMANDFLOW_ML_SERVICE_URL=http://127.0.0.1:8001`, and start the Laravel development server.
   ```bash
   cd backend
   php artisan serve --host=127.0.0.1 --port=8000
   ```

### API Endpoints (Laravel)

- `GET /api/ml/health`: Returns the health status of the ML service.
- `GET /api/ml/model-info`: Returns the details of the champion ML model currently loaded.
- `POST /api/ml/forecast/store-1`: Generates a demand forecast and business insights.

#### Example Request Body (`POST /api/ml/forecast/store-1`)
```json
{
  "forecast_days": 7,
  "forecast_open_days_only": true,
  "promo_dates": [],
  "school_holiday_dates": [],
  "current_stock": 18000,
  "unit_price": 12.5,
  "reorder_lead_time_days": 3,
  "safety_stock_percentage": 0.15
}
```

#### Example Response Summary
```json
{
  "forecast": [
    {
      "date": "2015-08-01",
      "predicted_sales": 5231
    }
  ],
  "business_insights": {
    "total_predicted_sales": 36617,
    "average_predicted_sales": 5231,
    "stockout_risk": "Low",
    "reorder_recommendation": {
      "reorder_needed": false,
      "recommended_reorder_date": null,
      "suggested_order_quantity": 0
    },
    "projected_revenue": 457712.5
  }
}
```

### Architecture Overview
In DemandFlow AI, **Laravel acts as the main application backend**. It is responsible for handling all incoming HTTP requests from the frontend, validating payloads, maintaining session state, and saving data to the database. 

**FastAPI acts as the specialized ML inference service**. It remains lightweight and is invoked internally by Laravel solely for demand forecasting using PyTorch models.

### Troubleshooting

- **Port Conflict**: If Laravel returns a FastAPI error format like `{"detail":"Not Found"}`, it means FastAPI is bound to port 8000. Ensure you terminate any stray Python processes and restart FastAPI on port `8001`.
- **ML Service Unavailable (503)**: If Laravel responds with an error stating the ML Service is unavailable, ensure your `uvicorn` instance is actively running on port `8001`. 
- **Config Cache Issues**: If you update `.env` variables and Laravel isn't picking them up, flush the caches using:
  ```bash
  php artisan config:clear
  php artisan cache:clear
  ```