import { useState, useEffect } from 'react';
import { Activity, AlertTriangle, CheckCircle2, ServerCrash } from 'lucide-react';
import { getMlHealth, getModelInfo } from '../../api/forecastApi';
import { HealthStatus, ModelInfo } from '../../types';

export default function ServiceStatusBanner() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    
    Promise.all([
      getMlHealth().catch(() => { throw new Error('ML Service Unavailable') }),
      getModelInfo().catch(() => null) // Model info isn't critical for basic health
    ])
    .then(([healthData, modelData]) => {
      if (!mounted) return;
      setHealth(healthData);
      setModelInfo(modelData);
      setError(null);
    })
    .catch(e => {
      if (!mounted) return;
      let msg = e.message;
      if (msg.includes('Network Error') || msg.includes('timeout')) {
        msg = "Backend service is unavailable. Please start the Laravel server.";
      } else if (msg.includes('ML Service')) {
        msg = "ML forecasting service is unavailable. Please start the FastAPI service.";
      }
      setError(msg);
    });

    return () => { mounted = false; };
  }, []);

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-center gap-3 text-red-400 text-sm mb-6">
        <ServerCrash size={18} />
        <span className="font-medium">{error}</span>
      </div>
    );
  }

  if (health?.status !== 'healthy') {
    return (
      <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 flex items-center gap-3 text-amber-400 text-sm mb-6">
        <AlertTriangle size={18} />
        <span className="font-medium">Connecting to services...</span>
      </div>
    );
  }

  // Once healthy, we don't strictly need to show a banner taking up space, 
  // but let's show a subtle healthy state as requested in "Add service status banner".
  // Actually, a permanent banner might be too much. Let's make it very subtle or only show briefly.
  // Wait, task says "It should show: Backend/ML status healthy, Model loaded"
  return (
    <div className="bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-3 flex flex-wrap items-center justify-between gap-4 text-emerald-400/90 text-sm mb-6">
      <div className="flex items-center gap-3">
        <CheckCircle2 size={18} className="text-emerald-500" />
        <span className="font-medium">Services Healthy</span>
      </div>
      {modelInfo && (
        <div className="flex items-center gap-2 text-xs text-emerald-400/70 border-l border-emerald-500/20 pl-4">
          <Activity size={14} />
          <span>Model: {modelInfo.model_version}</span>
        </div>
      )}
    </div>
  );
}
