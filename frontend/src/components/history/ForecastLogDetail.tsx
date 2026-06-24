import { ArrowLeft, Package, DollarSign } from 'lucide-react';
import { ForecastLog } from '../../types';
import { fmtNumber } from '../../utils/formatters';
import ForecastTable from '../forecast/ForecastTable';

interface DetailProps {
  log: ForecastLog;
  onBack: () => void;
}

export default function ForecastLogDetail({ log, onBack }: DetailProps) {
  const req = log.request_payload;
  const res = log.response_payload;
  const insights = res.business_insights;

  return (
    <div className="fade-up">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
        <button onClick={onBack} className="btn" style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem' }}>
          <ArrowLeft size={20} />
        </button>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#fff' }}>Forecast Details</h2>
          <div style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
            Generated on {new Date(log.created_at).toLocaleString()}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Package size={18} color="#06b6d4" /> Input Parameters
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Days to Forecast:</span>
              <span style={{ color: '#cbd5e1', fontWeight: 600 }}>{req.forecast_days}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Current Stock:</span>
              <span style={{ color: '#cbd5e1', fontWeight: 600 }}>{fmtNumber(req.current_stock)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Unit Price:</span>
              <span style={{ color: '#cbd5e1', fontWeight: 600 }}>${req.unit_price}</span>
            </div>
          </div>
        </div>

        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <DollarSign size={18} color="#10b981" /> Business Insights
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Stockout Risk:</span>
              <span style={{ color: '#cbd5e1', fontWeight: 600, textTransform: 'capitalize' }}>{insights.stockout_risk}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Total Sales:</span>
              <span style={{ color: '#cbd5e1', fontWeight: 600 }}>{fmtNumber(insights.total_predicted_sales)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Reorder Recommendation:</span>
              <span style={{ color: insights.reorder_needed ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                {insights.reorder_needed ? `Order ${fmtNumber(insights.recommended_reorder_quantity)}` : 'Sufficient'}
              </span>
            </div>
          </div>
          {insights.recommendation && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(255,255,255,0.05)', borderRadius: '0.5rem', fontSize: '0.85rem', color: '#cbd5e1' }}>
              {insights.recommendation}
            </div>
          )}
        </div>
      </div>

      <h3 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#fff', marginBottom: '1rem' }}>Daily Forecast</h3>
      <ForecastTable forecast={res.forecast} />
    </div>
  );
}
