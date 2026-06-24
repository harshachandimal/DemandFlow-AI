<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

use App\Http\Controllers\Api\ForecastController;

Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

Route::prefix('ml')->group(function () {
    Route::get('/health', [ForecastController::class, 'health']);
    Route::get('/model-info', [ForecastController::class, 'modelInfo']);
    Route::post('/forecast/store-1', [ForecastController::class, 'forecastStoreOne']);
});
