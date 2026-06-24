
import {
  ResponsiveContainer, ComposedChart, Bar, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, Area
} from 'recharts';
import { fmtNumber, fmtDate } from '../../utils/formatters';
import { ForecastResult } from '../../types';

const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: 'rgba(17,24,39,0.95)', border: '1px solid rgba(6,182,212,0.25)',
        borderRadius: '0.625rem', padding: '0.75rem 1rem', fontSize: '0.82rem',
      }}
    >
      <div style={{ color: '#94a3b8', marginBottom: '0.35rem', fontWeight: 600 }}>{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 700 }}>
          {p.name}: {fmtNumber(p.value as number)} units
        </div>
      ))}
    </div>
  );
};

interface ForecastChartProps {
  forecast: ForecastResult[] | null;
}

export default function ForecastChart({ forecast }: ForecastChartProps) {
  if (!forecast?.length) return null;

  const data = forecast.map((d) => ({
    date: fmtDate(d.date),
    'Predicted Sales': d.predicted_sales,
  }));

  return (
    <div className="glass-card fade-up" style={{ padding: '1.5rem' }}>
      <div style={{ marginBottom: '1.25rem' }}>
        <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--color-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Forecast Visualisation
        </div>
        <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#e2e8f0', marginTop: '0.15rem' }}>
          7-Day Predicted Sales Chart
        </div>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
          <defs>
            <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.85} />
              <stop offset="100%" stopColor="#10b981" stopOpacity={0.55} />
            </linearGradient>
            <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.06)" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: 'rgba(99,179,237,0.12)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false} tickLine={false}
            tickFormatter={(v) => fmtNumber(v)}
            width={60}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(6,182,212,0.05)' }} />
          <Legend
            wrapperStyle={{ fontSize: '0.78rem', color: '#64748b', paddingTop: '12px' }}
          />
          <Area dataKey="Predicted Sales" fill="url(#areaGrad)" stroke="none" />
          <Bar dataKey="Predicted Sales" fill="url(#barGrad)" radius={[5, 5, 0, 0]} maxBarSize={48} />
          <Line
            dataKey="Predicted Sales" stroke="#06b6d4"
            strokeWidth={2.5} dot={{ r: 4, fill: '#06b6d4', strokeWidth: 0 }}
            activeDot={{ r: 6, fill: '#fff', stroke: '#06b6d4', strokeWidth: 2 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
