import { useState } from 'react';
import { Plus, Play } from 'lucide-react';
import { ScenarioConfig, ScenarioComparisonResponse } from '../types';
import { compareScenarios } from '../api/scenarioApi';
import ScenarioFormCard from '../components/scenarios/ScenarioFormCard';
import ScenarioComparisonTable from '../components/scenarios/ScenarioComparisonTable';
import ScenarioForecastChart from '../components/scenarios/ScenarioForecastChart';
import BestScenarioCard from '../components/scenarios/BestScenarioCard';
import ScenarioReportCard from '../components/reports/ScenarioReportCard';
import ServiceStatusBanner from '../components/layout/ServiceStatusBanner';
import ErrorState from '../components/ui/ErrorState';
import EmptyState from '../components/ui/EmptyState';
import Button from '../components/ui/Button';
import { GitCompare } from 'lucide-react';

export default function ScenarioPlannerPage() {
  const [scenarios, setScenarios] = useState<ScenarioConfig[]>([
    { name: 'Normal Forecast', forecast_days: 7, forecast_open_days_only: true, current_stock: 18000, promo_dates: [] },
    { name: 'Promo Campaign', forecast_days: 7, forecast_open_days_only: true, current_stock: 18000, promo_dates: ['2015-08-03', '2015-08-04'] }
  ]);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ScenarioComparisonResponse | null>(null);

  const handleAdd = () => {
    if (scenarios.length >= 5) return;
    setScenarios([...scenarios, { name: `Scenario ${scenarios.length + 1}`, forecast_days: 7, forecast_open_days_only: true }]);
  };

  const handleRemove = (index: number) => {
    if (scenarios.length <= 1) return;
    const next = [...scenarios];
    next.splice(index, 1);
    setScenarios(next);
  };

  const handleChange = (index: number, updated: ScenarioConfig) => {
    const next = [...scenarios];
    next[index] = updated;
    setScenarios(next);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await compareScenarios(scenarios);
      setResults(data);
    } catch (e: any) {
      if (e.response && e.response.status === 422) {
        setError("Validation failed: " + JSON.stringify(e.response.data.errors));
      } else {
        setError("Unable to compare scenarios. Please check that Laravel and FastAPI are running.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '0', minHeight: 'calc(100vh - 64px)' }}>
      <div className="fade-up">
        <ServiceStatusBanner />
        
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.02em', marginBottom: '0.5rem' }}>
          What-if Scenario Planner
        </h1>
        <p style={{ color: '#94a3b8', marginBottom: '2rem' }}>
          Compare up to 5 forecasting scenarios side-by-side to understand the impact of promotions, pricing, and stock levels.
        </p>

        {error && (
          <div className="mb-8">
            <ErrorState message={error} title="Comparison Failed" />
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
          {scenarios.map((s, i) => (
            <ScenarioFormCard key={i} index={i} scenario={s} onChange={handleChange} onRemove={handleRemove} />
          ))}
          {scenarios.length < 5 && (
            <div 
              onClick={handleAdd}
              className="glass-card" 
              style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', minHeight: '200px', border: '1px dashed rgba(255,255,255,0.2)' }}
            >
              <Plus size={32} color="#94a3b8" style={{ marginBottom: '1rem' }} />
              <div style={{ color: '#94a3b8', fontWeight: 600 }}>Add Scenario</div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '3rem' }}>
          <Button onClick={handleSubmit} loading={loading} icon={<Play size={20} />}>
            Compare Scenarios
          </Button>
        </div>

        {!results && !loading && !error && (
          <EmptyState 
            icon={<GitCompare size={48} />}
            title="No Comparison Yet"
            description="Add multiple scenarios above and click Compare Scenarios to see the impact of your decisions."
          />
        )}

        {results && (
          <div className="fade-up" style={{ animationDelay: '0.2s' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1.5rem' }}>Comparison Results</h2>
            <BestScenarioCard best={results.best_scenario} />
            <ScenarioComparisonTable results={results.scenarios} />
            <ScenarioForecastChart results={results.scenarios} />
            <ScenarioReportCard data={results} />
          </div>
        )}
      </div>
    </main>
  );
}
