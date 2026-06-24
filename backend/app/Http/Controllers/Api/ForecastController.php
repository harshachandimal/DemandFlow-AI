<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Requests\StoreForecastRequest;
use App\Services\DemandForecastService;
use App\Models\ForecastLog;
use Exception;
use Illuminate\Http\JsonResponse;

class ForecastController extends Controller
{
    protected DemandForecastService $forecastService;

    public function __construct(DemandForecastService $forecastService)
    {
        $this->forecastService = $forecastService;
    }

    /**
     * Check ML Service Health.
     */
    public function health(): JsonResponse
    {
        try {
            $data = $this->forecastService->health();
            return response()->json($data);
        } catch (Exception $e) {
            return response()->json(['error' => $e->getMessage()], 503);
        }
    }

    /**
     * Get ML Model Info.
     */
    public function modelInfo(): JsonResponse
    {
        try {
            $data = $this->forecastService->modelInfo();
            return response()->json($data);
        } catch (Exception $e) {
            return response()->json(['error' => $e->getMessage()], 503);
        }
    }

    /**
     * Make a forecast for Store 1.
     */
    public function forecastStoreOne(StoreForecastRequest $request): JsonResponse
    {
        try {
            $payload = $request->validated();
            
            // Call the ML service
            $data = $this->forecastService->forecastStoreOne($payload);
            
            // Log the request and response in the database
            ForecastLog::create([
                'store_id' => 1,
                'request_payload' => $payload,
                'response_payload' => $data,
                'total_predicted_sales' => $data['business_insights']['total_predicted_sales'] ?? null,
                'average_predicted_sales' => $data['business_insights']['average_predicted_sales'] ?? null,
                'stockout_risk' => $data['business_insights']['stockout_risk'] ?? null,
                'reorder_needed' => $data['business_insights']['reorder_needed'] ?? null,
            ]);

            return response()->json($data);
        } catch (Exception $e) {
            $status = $e->getCode();
            if (!is_numeric($status) || $status < 100 || $status >= 600) {
                $status = 500;
            }
            if ($status === 500 && str_contains($e->getMessage(), 'ML Service is unavailable')) {
                $status = 503;
            }
            return response()->json(['error' => $e->getMessage()], $status);
        }
    }

    /**
     * Compare multiple forecasting scenarios.
     */
    public function compareScenarios(\App\Http\Requests\ScenarioComparisonRequest $request): JsonResponse
    {
        try {
            $payload = $request->validated();
            $scenariosData = [];

            foreach ($payload['scenarios'] as $scenario) {
                $name = $scenario['name'];
                $data = $this->forecastService->forecastStoreOne($scenario);
                
                $scenariosData[] = [
                    'scenario_name' => $name,
                    'forecast' => $data['forecast'] ?? [],
                    'business_insights' => $data['business_insights'] ?? []
                ];
            }

            $best = null;
            $bestScore = -INF;
            $bestReason = "";

            foreach ($scenariosData as $data) {
                $insights = $data['business_insights'];
                $score = 0;

                if (isset($insights['current_stock'])) {
                    $riskScore = ['low' => 100, 'medium' => 50, 'high' => 0];
                    $risk = strtolower($insights['stockout_risk'] ?? 'high');
                    $score += $riskScore[$risk] ?? 0;

                    if (!($insights['reorder_needed'] ?? true)) {
                        $score += 200;
                    }

                    $score += ($insights['projected_stock_after_7_days'] ?? 0) * 0.01;
                    $score -= ($insights['recommended_reorder_quantity'] ?? 0) * 0.05;

                    if ($score > $bestScore) {
                        $bestScore = $score;
                        $best = $data;
                        $bestReason = "Lowest stock-out risk, higher projected stock, or lower reorder need.";
                    }
                } else {
                    $score = $insights['total_predicted_sales'] ?? 0;
                    if ($score > $bestScore) {
                        $bestScore = $score;
                        $best = $data;
                        $bestReason = "Highest predicted sales. Inventory risk cannot be compared without current_stock.";
                    }
                }
            }

            return response()->json([
                'scenario_count' => count($scenariosData),
                'scenarios' => $scenariosData,
                'best_scenario' => [
                    'scenario_name' => $best['scenario_name'] ?? '',
                    'reason' => $bestReason
                ]
            ]);

        } catch (Exception $e) {
            return response()->json(['error' => $e->getMessage()], 500);
        }
    }
}
