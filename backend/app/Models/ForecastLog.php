<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ForecastLog extends Model
{
    protected $fillable = [
        'store_id',
        'request_payload',
        'response_payload',
        'total_predicted_sales',
        'average_predicted_sales',
        'stockout_risk',
        'reorder_needed',
    ];

    protected $casts = [
        'request_payload' => 'array',
        'response_payload' => 'array',
        'reorder_needed' => 'boolean',
    ];
}
