
import { TrendingUp, TrendingDown, BarChart2, Activity } from 'lucide-react';
import { fmtNumber, fmtDate } from '../../utils/formatters';
import { BusinessInsights, ForecastResult } from '../../types';

interface CardProps {
  icon: React.ElementType;
  iconColor: string;
  label: string;
  value: string;
  sub?: string;
}

const Card: React.FC<CardProps> = ({ icon: Icon, iconColor, label, value, sub }) => (
  <div
    className="glass-card fade-up"
    style={{ padding: '1.25rem', position: 'relative', overflow: 'hidden' }}
  >
    {/* Decorative glow */}
    <div
      aria-hidden="true"
      style={{
        position: 'absolute', top: '-20px', right: '-20px',
        width: '80px', height: '80px', borderRadius: '50%',
        background: iconColor, opacity: 0.08, filter: 'blur(20px)',
      }}
    />
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem' }}>
      <div
        style={{
          width: '32px', height: '32px', borderRadius: '8px',
          background: `${iconColor}18`,
          border: `1px solid ${iconColor}30`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        <Icon size={16} color={iconColor} />
      </div>
      <span style={{ fontSize: '0.72rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
        {label}
      </span>
    </div>
    <div className="kpi-value" style={{ color: '#e2e8f0' }}>{value}</div>
    {sub && <div style={{ marginTop: '0.35rem', fontSize: '0.75rem', color: '#475569' }}>{sub}</div>}
  </div>
);

interface DemandSummaryCardsProps {
  insights: BusinessInsights | null;
  forecast: ForecastResult[] | null;
}

export default function DemandSummaryCards({ insights, forecast }: DemandSummaryCardsProps) {
  if (!insights || !forecast) return null;

  const sorted = [...forecast].sort((a, b) => b.predicted_sales - a.predicted_sales);
  const highest = sorted[0];
  const lowest = sorted[sorted.length - 1];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '1rem' }}>
      <Card
        icon={BarChart2} iconColor="#06b6d4"
        label="Total Predicted Sales"
        value={fmtNumber(insights.total_predicted_sales)}
        sub="units over 7 days"
      />
      <Card
        icon={Activity} iconColor="#10b981"
        label="Avg Daily Sales"
        value={fmtNumber(insights.average_predicted_sales)}
        sub="units per day"
      />
      <Card
        icon={TrendingUp} iconColor="#f59e0b"
        label="Highest Demand Day"
        value={fmtNumber(highest?.predicted_sales)}
        sub={fmtDate(highest?.date)}
      />
      <Card
        icon={TrendingDown} iconColor="#8b5cf6"
        label="Lowest Demand Day"
        value={fmtNumber(lowest?.predicted_sales)}
        sub={fmtDate(lowest?.date)}
      />
    </div>
  );
}
