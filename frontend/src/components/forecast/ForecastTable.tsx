
import { fmtNumber, fmtWeekday } from '../../utils/formatters';
import { ForecastResult } from '../../types';

const demandColor: Record<string, { color: string; bg: string }> = {
  High:   { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
  Medium: { color: '#06b6d4', bg: 'rgba(6,182,212,0.1)' },
  Low:    { color: '#8b5cf6', bg: 'rgba(139,92,246,0.1)' },
};

const Badge: React.FC<{ text: string }> = ({ text }) => {
  const cfg = demandColor[text] || demandColor.Low;
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '0.2rem 0.55rem',
        borderRadius: '9999px',
        background: cfg.bg,
        color: cfg.color,
        fontSize: '0.72rem',
        fontWeight: 700,
      }}
    >
      {text}
    </span>
  );
};

const BoolCell: React.FC<{ val?: boolean }> = ({ val }) => (
  <span style={{ color: val ? '#10b981' : '#475569', fontWeight: 600, fontSize: '0.82rem' }}>
    {val ? '✓' : '—'}
  </span>
);

interface ForecastTableProps {
  forecast: ForecastResult[] | null;
}

export default function ForecastTable({ forecast }: ForecastTableProps) {
  if (!forecast?.length) return null;

  return (
    <div className="glass-card fade-up" style={{ padding: '1.5rem' }}>
      <div style={{ marginBottom: '1.25rem' }}>
        <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--color-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Forecast Breakdown
        </div>
        <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#e2e8f0', marginTop: '0.15rem' }}>
          Day-by-Day Prediction Table
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr>
              {['Date', 'Weekday', 'Promo', 'School Holiday', 'Demand Level', 'Predicted Sales'].map((h) => (
                <th
                  key={h}
                  style={{
                    padding: '0.65rem 0.875rem',
                    textAlign: 'left', fontWeight: 600,
                    fontSize: '0.72rem', color: '#64748b',
                    textTransform: 'uppercase', letterSpacing: '0.06em',
                    borderBottom: '1px solid rgba(99,179,237,0.1)',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {forecast.map((row, i) => (
              <tr
                key={row.date}
                style={{
                  background: i % 2 === 0 ? 'transparent' : 'rgba(26,34,53,0.3)',
                  transition: 'background 0.15s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(6,182,212,0.05)')}
                onMouseLeave={(e) => (e.currentTarget.style.background = i % 2 === 0 ? 'transparent' : 'rgba(26,34,53,0.3)')}
              >
                <td style={{ padding: '0.7rem 0.875rem', color: '#e2e8f0', fontWeight: 600 }}>{row.date}</td>
                <td style={{ padding: '0.7rem 0.875rem', color: '#94a3b8' }}>{fmtWeekday(row.date)}</td>
                <td style={{ padding: '0.7rem 0.875rem' }}><BoolCell val={row.is_promo ?? row.promo} /></td>
                <td style={{ padding: '0.7rem 0.875rem' }}><BoolCell val={row.is_school_holiday ?? row.school_holiday} /></td>
                <td style={{ padding: '0.7rem 0.875rem' }}><Badge text={row.demand_level || 'Medium'} /></td>
                <td style={{ padding: '0.7rem 0.875rem', color: '#06b6d4', fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>
                  {fmtNumber(row.predicted_sales)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
