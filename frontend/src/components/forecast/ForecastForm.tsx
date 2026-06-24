import React, { useState } from 'react';
import { Send, RefreshCw } from 'lucide-react';
import { ForecastPayload } from '../../types';

interface FormState {
  forecast_days: number | string;
  forecast_open_days_only: boolean;
  current_stock: number | string;
  unit_price: number | string;
  reorder_lead_time_days: number | string;
  safety_stock_percentage: number | string;
  promo_dates_raw: string;
  school_holiday_dates_raw: string;
}

const DEFAULT: FormState = {
  forecast_days: 7,
  forecast_open_days_only: true,
  current_stock: 18000,
  unit_price: 12.5,
  reorder_lead_time_days: 3,
  safety_stock_percentage: 0.15,
  promo_dates_raw: '',
  school_holiday_dates_raw: '',
};

const parseDates = (raw: string): string[] =>
  raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

interface ForecastFormProps {
  onSubmit: (payload: ForecastPayload) => void;
  loading: boolean;
}

export default function ForecastForm({ onSubmit, loading }: ForecastFormProps) {
  const [form, setForm] = useState<FormState>(DEFAULT);

  const set = (key: keyof FormState, val: string | number | boolean) => setForm((f) => ({ ...f, [key]: val }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      forecast_days: Number(form.forecast_days),
      forecast_open_days_only: form.forecast_open_days_only,
      current_stock: Number(form.current_stock),
      unit_price: Number(form.unit_price),
      reorder_lead_time_days: Number(form.reorder_lead_time_days),
      safety_stock_percentage: Number(form.safety_stock_percentage),
      promo_dates: parseDates(form.promo_dates_raw),
      school_holiday_dates: parseDates(form.school_holiday_dates_raw),
    });
  };

  const handleReset = () => setForm(DEFAULT);

  interface FieldProps {
    id: string;
    label: string;
    type?: string;
    step?: string;
    placeholder?: string;
    value: string | number;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  }

  const F: React.FC<FieldProps> = ({ id, label, type = 'number', step, placeholder, value, onChange }) => (
    <div>
      <label htmlFor={id} className="form-label">{label}</label>
      <input
        id={id} type={type} step={step} placeholder={placeholder}
        value={value} onChange={onChange}
        className="form-input"
      />
    </div>
  );

  return (
    <div className="glass-card" style={{ padding: '1.5rem' }}>
      <div style={{ marginBottom: '1.25rem' }}>
        <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--color-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Forecast Configuration
        </div>
        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#e2e8f0', marginTop: '0.15rem' }}>
          7-Day Demand Forecast · Store #1
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
          <F id="current_stock" label="Current Stock (units)" step="1"
            value={form.current_stock} onChange={(e) => set('current_stock', e.target.value)} />
          <F id="unit_price" label="Unit Price (USD)" step="0.01"
            value={form.unit_price} onChange={(e) => set('unit_price', e.target.value)} />
          <F id="reorder_lead_time_days" label="Reorder Lead Time (days)" step="1"
            value={form.reorder_lead_time_days} onChange={(e) => set('reorder_lead_time_days', e.target.value)} />
          <F id="safety_stock_percentage" label="Safety Stock %" step="0.01"
            value={form.safety_stock_percentage} onChange={(e) => set('safety_stock_percentage', e.target.value)} />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <F id="promo_dates" label="Promo Dates (comma-separated, YYYY-MM-DD)" type="text"
            placeholder="e.g. 2015-08-01, 2015-08-03"
            value={form.promo_dates_raw} onChange={(e) => set('promo_dates_raw', e.target.value)} />
          <F id="school_holiday_dates" label="School Holiday Dates (comma-separated)" type="text"
            placeholder="e.g. 2015-08-02"
            value={form.school_holiday_dates_raw} onChange={(e) => set('school_holiday_dates_raw', e.target.value)} />
        </div>

        {/* Open-days toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
          <input
            id="open_days_only" type="checkbox"
            checked={form.forecast_open_days_only}
            onChange={(e) => set('forecast_open_days_only', e.target.checked)}
            style={{ width: '16px', height: '16px', accentColor: '#06b6d4' }}
          />
          <label htmlFor="open_days_only" style={{ fontSize: '0.85rem', color: '#94a3b8', cursor: 'pointer' }}>
            Forecast open days only (exclude Sundays)
          </label>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button type="submit" disabled={loading} className="btn-primary" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
            {loading ? <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Send size={16} />}
            {loading ? 'Forecasting…' : 'Run Forecast'}
          </button>
          <button
            type="button" onClick={handleReset}
            style={{
              padding: '0.65rem 1rem', borderRadius: '0.625rem',
              background: 'rgba(100,116,139,0.15)',
              border: '1px solid rgba(100,116,139,0.25)',
              color: '#94a3b8', fontSize: '0.875rem', cursor: 'pointer',
              transition: 'background 0.2s',
            }}
          >
            Reset
          </button>
        </div>
      </form>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
