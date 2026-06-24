
import { fmtCurrency, fmtNumber } from '../../utils/formatters';
import { AlertTriangle, CheckCircle, ShoppingCart, DollarSign, Package, Layers } from 'lucide-react';
import { BusinessInsights } from '../../types';

const riskConfig: Record<string, { color: string; bg: string; border: string; icon: React.ElementType }> = {
  Low:    { color: '#10b981', bg: 'rgba(16,185,129,0.08)',  border: 'rgba(16,185,129,0.2)',  icon: CheckCircle },
  Medium: { color: '#f59e0b', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.2)', icon: AlertTriangle },
  High:   { color: '#ef4444', bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.2)',  icon: AlertTriangle },
};

interface RowProps {
  icon: React.ElementType;
  label: string;
  value: string;
  accent?: string;
}

const Row: React.FC<RowProps> = ({ icon: Icon, label, value, accent }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.55rem 0', borderBottom: '1px solid rgba(99,179,237,0.06)' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <Icon size={14} color="#475569" />
      <span style={{ fontSize: '0.8rem', color: '#64748b' }}>{label}</span>
    </div>
    <span style={{ fontSize: '0.875rem', fontWeight: 700, color: accent || '#e2e8f0' }}>{value}</span>
  </div>
);

interface BusinessInsightsCardProps {
  insights: BusinessInsights | null;
}

export default function BusinessInsightsCard({ insights }: BusinessInsightsCardProps) {
  if (!insights) return null;

  const rawRisk = insights.stockout_risk || 'Low';
  const risk = rawRisk.charAt(0).toUpperCase() + rawRisk.slice(1);
  const cfg = riskConfig[risk] || riskConfig.Low;
  const RiskIcon = cfg.icon;

  return (
    <div className="glass-card fade-up" style={{ padding: '1.5rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div>
          <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--color-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            Business Intelligence
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#e2e8f0', marginTop: '0.15rem' }}>
            Inventory & Revenue Insights
          </div>
        </div>
        {/* Risk badge */}
        <div
          style={{
            display: 'flex', alignItems: 'center', gap: '0.4rem',
            padding: '0.4rem 0.875rem', borderRadius: '9999px',
            background: cfg.bg, border: `1px solid ${cfg.border}`,
            color: cfg.color, fontSize: '0.8rem', fontWeight: 700,
          }}
        >
          <RiskIcon size={14} />
          {risk} Stockout Risk
        </div>
      </div>

      {/* Grid: Revenue / Stock */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        <div style={{ padding: '1rem', borderRadius: '0.75rem', background: 'rgba(6,182,212,0.06)', border: '1px solid rgba(6,182,212,0.12)' }}>
          <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Expected Revenue</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#06b6d4', marginTop: '0.35rem' }}>
            {fmtCurrency(insights.projected_revenue ?? insights.expected_revenue)}
          </div>
        </div>
        <div style={{ padding: '1rem', borderRadius: '0.75rem', background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.12)' }}>
          <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Projected Stock (after 7d)</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#10b981', marginTop: '0.35rem' }}>
            {fmtNumber(insights.projected_stock_after_7_days ?? insights.projected_stock_after_forecast)} units
          </div>
        </div>
      </div>

      {/* Detail rows */}
      <div>
        <Row icon={Package}     label="Current Stock"         value={`${fmtNumber(insights.current_stock)} units`} />
        <Row icon={Layers}      label="Reorder Needed"        value={insights.reorder_needed ? 'Yes' : 'No'}
          accent={insights.reorder_needed ? '#f59e0b' : '#10b981'} />
        <Row icon={ShoppingCart} label="Suggested Reorder Qty" value={`${fmtNumber(insights.recommended_reorder_quantity ?? 0)} units`} />
        {insights.recommended_reorder_date && (
          <Row icon={DollarSign} label="Reorder Date" value={insights.recommended_reorder_date} />
        )}
      </div>

      {/* Recommendation callout */}
      {insights.recommendation && (
        <div
          style={{
            marginTop: '1rem', padding: '0.875rem',
            background: `${cfg.bg}`, border: `1px solid ${cfg.border}`,
            borderRadius: '0.75rem',
            fontSize: '0.82rem', color: cfg.color, lineHeight: 1.6,
          }}
        >
          💡 {insights.recommendation}
        </div>
      )}
    </div>
  );
}
