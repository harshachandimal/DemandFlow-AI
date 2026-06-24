<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreForecastRequest extends FormRequest
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
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'forecast_days' => ['required', 'integer', 'in:7'],
            'forecast_open_days_only' => ['sometimes', 'boolean'],
            'promo_dates' => ['sometimes', 'array'],
            'promo_dates.*' => ['date_format:Y-m-d'],
            'school_holiday_dates' => ['sometimes', 'array'],
            'school_holiday_dates.*' => ['date_format:Y-m-d'],
            'current_stock' => ['nullable', 'integer', 'min:0'],
            'unit_price' => ['nullable', 'numeric', 'min:0'],
            'reorder_lead_time_days' => ['sometimes', 'integer', 'min:1', 'max:30'],
            'safety_stock_percentage' => ['sometimes', 'numeric', 'min:0', 'max:1'],
        ];
    }
}
