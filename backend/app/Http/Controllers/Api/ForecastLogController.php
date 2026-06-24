<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\ForecastLog;
use Illuminate\Http\Request;

class ForecastLogController extends Controller
{
    /**
     * Display a listing of forecast logs (summary fields).
     */
    public function index(Request $request)
    {
        $limit = $request->query('limit', 20);
        
        $logs = ForecastLog::select([
                'id',
                'store_id',
                'total_predicted_sales',
                'average_predicted_sales',
                'stockout_risk',
                'reorder_needed',
                'created_at'
            ])
            ->where('user_id', auth()->id())
            ->orderBy('created_at', 'desc')
            ->limit((int) $limit)
            ->get();
            
        return response()->json($logs);
    }

    /**
     * Display the specified forecast log (full details).
     */
    public function show(ForecastLog $forecastLog)
    {
        if ($forecastLog->user_id !== auth()->id()) {
            return response()->json(['message' => 'Forbidden'], 403);
        }

        // Select all fields for the detail view
        return response()->json($forecastLog);
    }

    /**
     * Remove the specified forecast log from storage.
     */
    public function destroy(ForecastLog $forecastLog)
    {
        if ($forecastLog->user_id !== auth()->id()) {
            return response()->json(['message' => 'Forbidden'], 403);
        }

        $forecastLog->delete();
        
        return response()->json([
            'message' => 'Forecast log deleted successfully'
        ]);
    }
}
