<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

use App\Http\Controllers\Api\AuthController;
use App\Http\Controllers\Api\ForecastController;
use App\Http\Controllers\Api\ForecastLogController;

Route::prefix('auth')->group(function () {
    Route::post('/register', [AuthController::class, 'register']);
    Route::post('/login', [AuthController::class, 'login']);

    Route::middleware('auth:sanctum')->group(function () {
        Route::get('/me', [AuthController::class, 'me']);
        Route::post('/logout', [AuthController::class, 'logout']);
    });
});

Route::prefix('ml')->group(function () {
    Route::get('/health', [ForecastController::class, 'health']);
    
    Route::middleware('auth:sanctum')->group(function () {
        Route::get('/model-info', [ForecastController::class, 'modelInfo']);
        Route::post('/forecast/store-1', [ForecastController::class, 'forecastStoreOne']);
        Route::post('/forecast/scenarios', [ForecastController::class, 'compareScenarios']);
        
        // Forecast History Logs
        Route::get('/forecast-logs', [ForecastLogController::class, 'index']);
        Route::get('/forecast-logs/{forecastLog}', [ForecastLogController::class, 'show']);
        Route::delete('/forecast-logs/{forecastLog}', [ForecastLogController::class, 'destroy']);
    });
});
