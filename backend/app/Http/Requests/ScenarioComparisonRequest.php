<?php

namespace App\Http\Requests;

use Illuminate\Contracts\Validation\ValidationRule;
use Illuminate\Foundation\Http\FormRequest;

class ScenarioComparisonRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'scenarios' => 'required|array|min:1|max:5',
            'scenarios.*.name' => 'required|string|max:100',
            'scenarios.*.forecast_days' => 'required|integer|in:7',
            'scenarios.*.forecast_open_days_only' => 'sometimes|boolean',
            'scenarios.*.promo_dates' => 'sometimes|array',
            'scenarios.*.promo_dates.*' => 'date_format:Y-m-d',
            'scenarios.*.school_holiday_dates' => 'sometimes|array',
            'scenarios.*.school_holiday_dates.*' => 'date_format:Y-m-d',
            'scenarios.*.current_stock' => 'nullable|integer|min:0',
            'scenarios.*.unit_price' => 'nullable|numeric|min:0',
            'scenarios.*.reorder_lead_time_days' => 'sometimes|integer|min:1|max:30',
            'scenarios.*.safety_stock_percentage' => 'sometimes|numeric|min:0|max:1',
        ];
    }
}
