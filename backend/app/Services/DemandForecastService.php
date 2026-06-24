<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Exception;

class DemandForecastService
{
    protected string $baseUrl;

    public function __construct()
    {
        $this->baseUrl = config('services.demandflow_ml.url');
    }

    /**
     * Call the ML service health endpoint.
     *
     * @return array
     * @throws Exception
     */
    public function health(): array
    {
        try {
            $response = Http::timeout(30)->get("{$this->baseUrl}/health");

            if (!$response->successful()) {
                throw new Exception('ML Service is unavailable or returned an error.');
            }

            return $response->json();
        } catch (Exception $e) {
            throw new Exception('Error connecting to ML service: ' . $e->getMessage(), 503);
        }
    }

    /**
     * Call the ML service model-info endpoint.
     *
     * @return array
     * @throws Exception
     */
    public function modelInfo(): array
    {
        try {
            $response = Http::timeout(30)->get("{$this->baseUrl}/model-info");

            if (!$response->successful()) {
                throw new Exception('ML Service is unavailable or returned an error.');
            }

            return $response->json();
        } catch (Exception $e) {
            throw new Exception('Error connecting to ML service: ' . $e->getMessage(), 503);
        }
    }

    /**
     * Call the ML service forecast/store-1 endpoint.
     *
     * @param array $payload
     * @return array
     * @throws Exception
     */
    public function forecastStoreOne(array $payload): array
    {
        try {
            $response = Http::timeout(30)->post("{$this->baseUrl}/api/v1/forecast/store-1", $payload);

            if ($response->status() === 422 || $response->status() === 400) {
                throw new Exception('ML Service validation error: ' . $response->body(), $response->status());
            }

            if (!$response->successful()) {
                throw new Exception('ML Service is unavailable or returned an error: ' . $response->body(), $response->status() ?: 503);
            }

            return $response->json();
        } catch (Exception $e) {
            $code = $e->getCode() ?: 503;
            throw new Exception($e->getMessage(), $code);
        }
    }
}
