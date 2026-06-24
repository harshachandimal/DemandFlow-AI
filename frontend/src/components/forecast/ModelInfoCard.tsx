
import { Brain, BadgeCheck } from 'lucide-react';
import { fmtNumber } from '../../utils/formatters';
import { ModelInfo } from '../../types';

interface MetricRowProps {
  label: string;
  value?: number | null;
  unit?: string;
}

const MetricRow: React.FC<MetricRowProps> = ({ label, value, unit = '' }) => (
  <div
    style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '0.5rem 0', borderBottom: '1px solid rgba(99,179,237,0.07)',
    }}
  >
    <span style={{ fontSize: '0.8rem', color: '#64748b' }}>{label}</span>
    <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#e2e8f0' }}>
      {value !== undefined && value !== null ? `${fmtNumber(value, 2)}${unit}` : '—'}
    </span>
  </div>
);

interface ModelInfoCardProps {
  modelInfo: ModelInfo | null;
  loading: boolean;
}

export default function ModelInfoCard({ modelInfo, loading }: ModelInfoCardProps) {
  if (loading) {
    return (
      <div className="glass-card" style={{ padding: '1.25rem' }}>
        <div style={{ height: '140px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ color: '#475569', fontSize: '0.875rem' }}>Loading model info…</div>
        </div>
      </div>
    );
  }

  if (!modelInfo) return null;

  const metrics = modelInfo.metrics || {};
  const supportsInsights = modelInfo.supports_business_insights;

  return (
    <div className="glass-card" style={{ padding: '1.25rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <div style={{ flex: 1, minWidth: 0, paddingRight: '0.5rem' }}>
          <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--color-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            Champion Model
          </div>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#e2e8f0', marginTop: '0.15rem', wordBreak: 'break-word' }}>
            {modelInfo.model_name || 'N/A'}
          </div>
        </div>
        <div
          style={{
            padding: '0.3rem 0.6rem', borderRadius: '9999px',
            background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.2)',
            fontSize: '0.7rem', fontWeight: 700, color: '#06b6d4',
            flexShrink: 0
          }}
        >
          {modelInfo.model_version?.startsWith('v') ? modelInfo.model_version : `v${modelInfo.model_version || '1.0'}`}
        </div>
      </div>

      {/* Store badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '1rem' }}>
        <Brain size={14} color="#64748b" />
        <span style={{ fontSize: '0.78rem', color: '#64748b' }}>
          Store #{modelInfo.store_id || 1} · PyTorch LSTM
        </span>
        {supportsInsights && (
          <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#10b981', fontSize: '0.72rem', fontWeight: 600 }}>
            <BadgeCheck size={13} />Business Insights ✓
          </span>
        )}
      </div>

      {/* Metrics */}
      <div>
        <MetricRow label="MAE  (Mean Abs. Error)" value={metrics.mae} />
        <MetricRow label="RMSE (Root Mean Sq. Error)" value={metrics.rmse} />
        <MetricRow label="MAPE (Mean Abs. % Error)" value={metrics.mape} unit="%" />
      </div>
    </div>
  );
}
