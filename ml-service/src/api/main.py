from fastapi import FastAPI, HTTPException
from .schemas import ForecastRequest, ForecastResponse
from .model_loader import load_model_and_scalers
from .forecast_service import generate_forecast

app = FastAPI(title="DemandFlow AI ML Service")

# Global variables to hold model and scalers
model = None
past_scaler = None
future_scaler = None
target_scaler = None
model_loaded = False

@app.on_event("startup")
def startup_event():
    global model, past_scaler, future_scaler, target_scaler, model_loaded
    try:
        model, past_scaler, future_scaler, target_scaler = load_model_and_scalers()
        model_loaded = True
        print("Model and scalers loaded successfully.")
    except Exception as e:
        print(f"Failed to load model and scalers: {e}")

@app.get("/")
def read_root():
    return {
        "service": "DemandFlow AI ML Service",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

@app.get("/model-info")
def model_info():
    return {
        "model_name": "RossmannEnhancedFutureAwareLSTM",
        "model_version": "v2",
        "store_id": 1,
        "mae": 332.43,
        "rmse": 439.03,
        "mape": 7.32,
        "supports_business_insights": True
    }

@app.post("/api/v1/forecast/store-1", response_model=ForecastResponse)
def forecast_store_1(request: ForecastRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    try:
        response = generate_forecast(request, model, past_scaler, future_scaler, target_scaler)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
