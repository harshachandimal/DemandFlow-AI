import { ScenarioResult } from '../../types';
import { fmtNumber, fmtCurrency } from '../../utils/formatters';

export default function ScenarioComparisonTable({ results }: { results: ScenarioResult[] }) {
  if (!results.length) return null;

  return (
    <div className="glass-card" style={{ overflowX: 'auto', marginBottom: '2rem' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', minWidth: '800px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <th style={{ padding: '1rem', color: '#94a3b8', fontWeight: 600, fontSize: '0.85rem' }}>Metric</th>
            {results.map((r, i) => (
              <th key={i} style={{ padding: '1rem', color: '#e2e8f0', fontWeight: 600, fontSize: '0.9rem' }}>{r.scenario_name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <td style={{ padding: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>Total Sales</td>
            {results.map((r, i) => <td key={i} style={{ padding: '1rem', color: '#cbd5e1', fontWeight: 600 }}>{fmtNumber(r.business_insights.total_predicted_sales)}</td>)}
          </tr>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <td style={{ padding: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>Projected Revenue</td>
            {results.map((r, i) => <td key={i} style={{ padding: '1rem', color: '#10b981', fontWeight: 600 }}>{fmtCurrency(r.business_insights.projected_revenue ?? r.business_insights.expected_revenue)}</td>)}
          </tr>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <td style={{ padding: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>Stockout Risk</td>
            {results.map((r, i) => {
              const risk = r.business_insights.stockout_risk.toLowerCase();
              const color = risk === 'high' ? '#ef4444' : risk === 'medium' ? '#f59e0b' : '#10b981';
              return <td key={i} style={{ padding: '1rem', color, fontWeight: 700, textTransform: 'capitalize' }}>{risk}</td>
            })}
          </tr>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <td style={{ padding: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>Reorder Needed?</td>
            {results.map((r, i) => <td key={i} style={{ padding: '1rem', color: '#cbd5e1' }}>{r.business_insights.reorder_needed ? 'Yes' : 'No'}</td>)}
          </tr>
          <tr>
            <td style={{ padding: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>Reorder Qty</td>
            {results.map((r, i) => <td key={i} style={{ padding: '1rem', color: '#cbd5e1' }}>{fmtNumber(r.business_insights.recommended_reorder_quantity ?? 0)}</td>)}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
