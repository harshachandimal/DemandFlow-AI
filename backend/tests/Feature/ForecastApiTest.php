<?php

namespace Tests\Feature;

use App\Models\ForecastLog;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\Http;
use Tests\TestCase;

class ForecastApiTest extends TestCase
{
    use RefreshDatabase;

    public function test_health_endpoint_returns_success()
    {
        Http::fake([
            '*health' => Http::response(['status' => 'ok', 'version' => '1.0.0'], 200)
        ]);

        $response = $this->getJson('/api/ml/health');

        $response->assertStatus(200)
                 ->assertJson(['status' => 'ok', 'version' => '1.0.0']);
    }

    public function test_model_info_endpoint_returns_success()
    {
        Http::fake([
            '*model-info' => Http::response(['model_name' => 'Champion Rossmann LSTM v2'], 200)
        ]);

        $response = $this->getJson('/api/ml/model-info');

        $response->assertStatus(200)
                 ->assertJson(['model_name' => 'Champion Rossmann LSTM v2']);
    }

    public function test_forecast_store_one_successful_request()
    {
        Http::fake([
            '*api/v1/forecast/store-1' => Http::response([
                'forecast' => [],
                'business_insights' => [
                    'total_predicted_sales' => 50000,
                    'average_predicted_sales' => 7000,
                    'stockout_risk' => 'Low',
                    'reorder_recommendation' => [
                        'reorder_needed' => false
                    ]
                ]
            ], 200)
        ]);

        $payload = [
            'forecast_days' => 7,
            'forecast_open_days_only' => true,
            'current_stock' => 18000,
            'unit_price' => 12.5,
            'reorder_lead_time_days' => 3,
            'safety_stock_percentage' => 0.15
        ];

        $response = $this->postJson('/api/ml/forecast/store-1', $payload);

        $response->assertStatus(200)
                 ->assertJsonPath('business_insights.total_predicted_sales', 50000);

        $this->assertDatabaseHas('forecast_logs', [
            'store_id' => 1,
            'total_predicted_sales' => 50000,
            'stockout_risk' => 'Low',
            'reorder_needed' => false,
        ]);
    }

    public function test_forecast_store_one_validation_error()
    {
        // Missing required 'forecast_days'
        $payload = [
            'forecast_open_days_only' => true,
        ];

        $response = $this->postJson('/api/ml/forecast/store-1', $payload);

        $response->assertStatus(422)
                 ->assertJsonValidationErrors(['forecast_days']);
    }
    
    public function test_forecast_store_one_ml_service_unavailable()
    {
        Http::fake([
            '*api/v1/forecast/store-1' => Http::response(['detail' => 'Service Unavailable'], 503)
        ]);

        $payload = [
            'forecast_days' => 7,
            'forecast_open_days_only' => true,
        ];

        $response = $this->postJson('/api/ml/forecast/store-1', $payload);

        $response->assertStatus(503);
    }
}
