<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

use App\Http\Controllers\Api\ForecastController;
use App\Http\Controllers\Api\ForecastLogController;

Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

Route::prefix('ml')->group(function () {
    Route::get('/health', [ForecastController::class, 'health']);
    Route::get('/model-info', [ForecastController::class, 'modelInfo']);
    Route::post('/forecast/store-1', [ForecastController::class, 'forecastStoreOne']);
    
    // Forecast History Logs
    Route::get('/forecast-logs', [ForecastLogController::class, 'index']);
    Route::get('/forecast-logs/{forecastLog}', [ForecastLogController::class, 'show']);
    Route::delete('/forecast-logs/{forecastLog}', [ForecastLogController::class, 'destroy']);
});
