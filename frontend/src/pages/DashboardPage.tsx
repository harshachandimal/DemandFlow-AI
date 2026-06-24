import { useState, useEffect, useCallback } from 'react';
import { getModelInfo, forecastStoreOne } from '../api/forecastApi';
import ServiceStatusBanner from '../components/layout/ServiceStatusBanner';
import ModelInfoCard from '../components/forecast/ModelInfoCard';
import ForecastReportCard from '../components/reports/ForecastReportCard';
import ForecastForm from '../components/forecast/ForecastForm';
import DemandSummaryCards from '../components/forecast/DemandSummaryCards';
import BusinessInsightsCard from '../components/forecast/BusinessInsightsCard';
import ForecastChart from '../components/forecast/ForecastChart';
import ForecastTable from '../components/forecast/ForecastTable';
import { ModelInfo, ForecastPayload, ForecastResult, BusinessInsights } from '../types';
import ErrorState from '../components/ui/ErrorState';
import EmptyState from '../components/ui/EmptyState';
import { Activity } from 'lucide-react';

export default function DashboardPage() {
  const [modelInfo, setModelInfo]   = useState<ModelInfo | null>(null);
  const [modelLoading, setML]       = useState(true);

  const [forecast, setForecast]         = useState<ForecastResult[] | null>(null);
  const [insights, setInsights]         = useState<BusinessInsights | null>(null);
  const [forecastLoading, setFL]        = useState(false);
  const [forecastError, setForecastErr] = useState<string | null>(null);

  // Load model info on mount
  useEffect(() => {

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

      {/* Status Banner */}
      <ServiceStatusBanner />

      {/* Model Info row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
        <ModelInfoCard modelInfo={modelInfo} loading={modelLoading} />
      </div>

      {/* Forecast form */}
      <ForecastForm onSubmit={handleForecast} loading={forecastLoading} />

      {/* Forecast error */}
      {forecastError && (
        <ErrorState message={forecastError} title="Forecast Failed" />
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

          <ForecastReportCard data={{ forecast, business_insights: insights }} modelInfo={modelInfo} />
        </>
      )}

      {/* Empty state — no forecast yet */}
      {!forecast && !forecastLoading && !forecastError && (
        <EmptyState 
          icon={<Activity size={48} />}
          title="No Forecast Generated"
          description="Configure the form above and click Run Forecast to generate predictions."
        />
      )}
    </div>
  );
}
