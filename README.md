# DemandFlow AI

AI-powered demand forecasting platform for retail. Combines a PyTorch LSTM model, a FastAPI ML inference service, a Laravel API gateway, and a React dashboard.

---

## Architecture

```
React Frontend (Vite + Tailwind)
        │
        ▼  HTTP (port 5173 → 8000)
Laravel Backend (API Gateway)
        │
        ▼  HTTP (port 8000 → 8001)
FastAPI ML Service
        │
        ▼
PyTorch LSTM Model (Rossmann Store #1)
```

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10+ |
| PHP | 8.2+ |
| Composer | latest |
| Node.js | 18+ |
| npm | 9+ |

---

## Running the Services

> **Order matters**: start FastAPI first, then Laravel, then React.

### 1. Start the FastAPI ML Service (port 8001)

```bash
cd ml-service
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
```

### 2. Start the Laravel Backend (port 8000)

Ensure `.env` has: `DEMANDFLOW_ML_SERVICE_URL=http://127.0.0.1:8001`

```bash
cd backend
php artisan serve --host=127.0.0.1 --port=8000
```

### 3. Start the React Frontend (port 5173)

```bash
cd frontend
npm install   # only needed once
npm run dev
```

---

## URLs

| Service | URL |
|---|---|
| React Dashboard | http://127.0.0.1:5173/dashboard |
| Laravel API | http://127.0.0.1:8000/api |
| FastAPI Docs | http://127.0.0.1:8001/docs |

---

## Laravel API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/ml/health` | ML service health check |
| GET | `/api/ml/model-info` | Champion model metadata & metrics |
| POST | `/api/ml/forecast/store-1` | Generate 7-day demand forecast |
| POST | `/api/ml/forecast/scenarios` | Compare multiple forecasting scenarios side-by-side |
| GET | `/api/ml/forecast-logs` | Get paginated forecast history logs |
| GET | `/api/ml/forecast-logs/{id}` | Get specific forecast log details |
| DELETE | `/api/ml/forecast-logs/{id}` | Delete a forecast log |

### Example Forecast Request

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

### Example Forecast Response

```json
{
  "forecast": [
    { "date": "2015-08-01", "predicted_sales": 5231, "demand_level": "High" }
  ],
  "business_insights": {
    "total_predicted_sales": 36617,
    "average_predicted_sales": 5231,
    "stockout_risk": "low",
    "projected_revenue": 457712.5,
    "reorder_needed": false,
    "recommended_reorder_date": null,
    "recommended_reorder_quantity": 0
  }
}
```

---

## Dashboard Features (Phase 5.1 - 5.4)

- **System Status** — Laravel & ML service connection health
- **Champion Model Info** — model name, version, MAE / RMSE / MAPE
- **Forecast Form** — configurable stock, pricing, promo/holiday dates
- **KPI Cards** — total sales, avg daily, highest & lowest demand day
- **Business Insights** — expected revenue, projected stock, stockout risk, reorder recommendation
- **Forecast Chart** — responsive Recharts bar + line combo chart
- **Forecast Table** — day-by-day table with demand level badges
- **Forecast History Dashboard** — browse, view details, and manage past forecasts using the `/history` route
- **What-if Scenario Planner** — compare up to 5 parallel forecasts side-by-side, visualizing data on a combined line chart, and utilizing an automated 'Best Scenario' heuristic on the `/scenarios` route.

*Note: The Scenario Planner reuses the trained PyTorch LSTM model behind the scenes, parallelizing multiple predictive runs to compare hypothetical business decisions, rather than training separate ML models.*

---

## Project Structure

```
demandflow-ai/
├── backend/          # Laravel 11 (API gateway)
├── ml-service/       # FastAPI + PyTorch LSTM
└── frontend/         # React 18 + Vite 8 + Tailwind 4
    └── src/
        ├── api/          # axiosClient + forecastApi
        ├── components/   # layout + forecast components (.tsx)
        ├── pages/        # DashboardPage (.tsx)
        └── types/        # TypeScript interfaces
        └── utils/        # formatters
```

---

## Troubleshooting

- **Port conflict**: Ensure FastAPI is on `8001` and Laravel on `8000`. Check for stray `uvicorn` processes.
- **ML Service unavailable (503)**: Restart `uvicorn` on port `8001`.
- **CORS errors in browser**: Verify `backend/config/cors.php` allows `http://127.0.0.1:5173` and `http://localhost:5173`.
- **Config cache stale**:
  ```bash
  cd backend
  php artisan config:clear
  php artisan cache:clear
  ```