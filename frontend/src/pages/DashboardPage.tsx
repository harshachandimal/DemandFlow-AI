import { useState, useEffect, useCallback } from 'react';
import { getMlHealth, getModelInfo, forecastStoreOne } from '../api/forecastApi';
import StatusCard from '../components/forecast/StatusCard';
import ModelInfoCard from '../components/forecast/ModelInfoCard';
import ForecastForm from '../components/forecast/ForecastForm';
import DemandSummaryCards from '../components/forecast/DemandSummaryCards';
import BusinessInsightsCard from '../components/forecast/BusinessInsightsCard';
import ForecastChart from '../components/forecast/ForecastChart';
import ForecastTable from '../components/forecast/ForecastTable';
import { HealthStatus, ModelInfo, ForecastPayload, ForecastResult, BusinessInsights } from '../types';

export default function DashboardPage() {
  const [health, setHealth]         = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo]   = useState<ModelInfo | null>(null);
  const [healthLoading, setHL]      = useState(true);
  const [modelLoading, setML]       = useState(true);
  const [healthError, setHE]        = useState<string | null>(null);

  const [forecast, setForecast]         = useState<ForecastResult[] | null>(null);
  const [insights, setInsights]         = useState<BusinessInsights | null>(null);
  const [forecastLoading, setFL]        = useState(false);
  const [forecastError, setForecastErr] = useState<string | null>(null);

  // Load health + model info on mount
  useEffect(() => {
    getMlHealth()
      .then(setHealth)
      .catch((e: Error) => setHE(e.message))
      .finally(() => setHL(false));

    getModelInfo()
      .then(setModelInfo)
      .catch(() => {/* silent — status card shows the error */})
      .finally(() => setML(false));
  }, []);

  const handleForecast = useCallback(async (payload: ForecastPayload) => {
    setFL(true);
    setForecastErr(null);
    setForecast(null);
    setInsights(null);
    try {
      const data = await forecastStoreOne(payload);
      setForecast(data.forecast || []);
      setInsights(data.business_insights || null);
    } catch (e: any) {
      let msg = e.message;
      if (msg.includes('Network Error') || msg.includes('timeout')) {
        msg = "Backend service is unavailable. Please start Laravel.";
      } else if (msg.includes('500') || msg.toLowerCase().includes('fastapi') || msg.toLowerCase().includes('ml service')) {
        msg = "ML forecasting service is unavailable. Please start FastAPI.";
      }
      setForecastErr(msg);
    } finally {
      setFL(false);
    }
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

      {/* Page title */}
      <div style={{ paddingBottom: '0.5rem', borderBottom: '1px solid var(--color-border)' }}>
        <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#e2e8f0', letterSpacing: '-0.03em' }}>
          Demand Forecast{' '}
          <span className="gradient-text">Dashboard</span>
        </h1>
        <p style={{ marginTop: '0.35rem', fontSize: '0.875rem', color: '#64748b' }}>
          Real-time ML-powered sales forecasting · Store #1 · Rossmann dataset
        </p>
      </div>

      {/* Status + Model Info row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
        <StatusCard health={health} loading={healthLoading} error={healthError} />
        <ModelInfoCard modelInfo={modelInfo} loading={modelLoading} />
      </div>

      {/* Forecast form */}
      <ForecastForm onSubmit={handleForecast} loading={forecastLoading} />

      {/* Forecast error */}
      {forecastError && (
        <div
          style={{
            padding: '1rem 1.25rem',
            background: 'rgba(239,68,68,0.08)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: '0.75rem',
            color: '#f87171', fontSize: '0.875rem',
          }}
        >
          ⚠️ Forecast failed: {forecastError}
        </div>
      )}

      {/* Forecast results */}
      {forecast && insights && (
        <>
          <DemandSummaryCards insights={insights} forecast={forecast} />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <ForecastChart forecast={forecast} />
            <BusinessInsightsCard insights={insights} />
          </div>

          <ForecastTable forecast={forecast} />
        </>
      )}

      {/* Empty state — no forecast yet */}
      {!forecast && !forecastLoading && !forecastError && (
        <div
          style={{
            textAlign: 'center', padding: '3rem',
            color: '#334155', fontSize: '0.9rem',
            border: '1px dashed rgba(99,179,237,0.1)',
            borderRadius: '1rem',
          }}
        >
          Configure the form above and click <strong style={{ color: '#475569' }}>Run Forecast</strong> to generate predictions.
        </div>
      )}
    </div>
  );
}
