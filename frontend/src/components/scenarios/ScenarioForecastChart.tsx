import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { ScenarioResult } from '../../types';

export default function ScenarioForecastChart({ results }: { results: ScenarioResult[] }) {
  if (!results.length) return null;

  // We need to merge all scenario forecasts by date into one data array
  // Format: { date: '2015-08-01', "Scenario 1": 5000, "Scenario 2": 6000 }
  const dataMap = new Map<string, any>();

  results.forEach(res => {
    res.forecast.forEach(f => {
      const existing = dataMap.get(f.date) || { date: f.date };
      existing[res.scenario_name] = f.predicted_sales;
      dataMap.set(f.date, existing);
    });
  });

  const chartData = Array.from(dataMap.values()).sort((a, b) => a.date.localeCompare(b.date));
  const colors = ['#06b6d4', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

  return (
    <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '2rem', height: '400px' }}>
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '1.5rem' }}>Forecast Trajectory Comparison</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis 
            dataKey="date" 
            stroke="#64748b" 
            fontSize={12} 
            tickFormatter={(val) => val.slice(5)} 
          />
          <YAxis 
            stroke="#64748b" 
            fontSize={12} 
            tickFormatter={(val) => (val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val)} 
          />
          <Tooltip 
            contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '0.5rem', color: '#fff' }}
            itemStyle={{ fontSize: '0.85rem' }}
          />
          <Legend wrapperStyle={{ fontSize: '0.85rem', color: '#cbd5e1' }} />
          {results.map((r, i) => (
            <Line 
              key={r.scenario_name}
              type="monotone" 
              dataKey={r.scenario_name} 
              stroke={colors[i % colors.length]} 
              strokeWidth={3}
              dot={{ r: 4, strokeWidth: 2 }}
              activeDot={{ r: 6 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
