import { Eye, Trash2 } from 'lucide-react';
import { ForecastLogSummary } from '../../types';
import { fmtNumber } from '../../utils/formatters';

interface TableProps {
  logs: ForecastLogSummary[];
  onView: (id: number) => void;
  onDelete: (id: number) => void;
}

export default function ForecastHistoryTable({ logs, onView, onDelete }: TableProps) {
  const getRiskBadge = (risk: string) => {
    const riskLower = risk.toLowerCase();
    if (riskLower === 'high') {
      return <span style={{ background: 'rgba(239,68,68,0.2)', color: '#ef4444', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontSize: '0.75rem', fontWeight: 600 }}>High Risk</span>;
    }
    if (riskLower === 'medium') {
      return <span style={{ background: 'rgba(245,158,11,0.2)', color: '#f59e0b', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontSize: '0.75rem', fontWeight: 600 }}>Med Risk</span>;
    }
    return <span style={{ background: 'rgba(16,185,129,0.2)', color: '#10b981', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontSize: '0.75rem', fontWeight: 600 }}>Low Risk</span>;
  };

  return (
    <div className="glass-card" style={{ overflow: 'hidden' }}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', minWidth: '800px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <th style={{ padding: '1rem', color: '#94a3b8', fontWeight: 600, fontSize: '0.85rem' }}>Date Created</th>
              <th style={{ padding: '1rem', color: '#94a3b8', fontWeight: 600, fontSize: '0.85rem' }}>Store ID</th>
              <th style={{ padding: '1rem', color: '#94a3b8', fontWeight: 600, fontSize: '0.85rem' }}>Total Sales</th>
              <th style={{ padding: '1rem', color: '#94a3b8', fontWeight: 600, fontSize: '0.85rem' }}>Avg/Day</th>
              <th style={{ padding: '1rem', color: '#94a3b8', fontWeight: 600, fontSize: '0.85rem' }}>Stockout Risk</th>
              <th style={{ padding: '1rem', color: '#94a3b8', fontWeight: 600, fontSize: '0.85rem' }}>Reorder?</th>
              <th style={{ padding: '1rem', color: '#94a3b8', fontWeight: 600, fontSize: '0.85rem', textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => (
              <tr key={log.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', transition: 'background 0.2s' }} className="hover:bg-slate-800/30">
                <td style={{ padding: '1rem', color: '#e2e8f0', fontSize: '0.9rem' }}>
                  {new Date(log.created_at).toLocaleString()}
                </td>
                <td style={{ padding: '1rem', color: '#cbd5e1', fontSize: '0.9rem' }}>#{log.store_id}</td>
                <td style={{ padding: '1rem', color: '#cbd5e1', fontSize: '0.9rem', fontWeight: 600 }}>{fmtNumber(log.total_predicted_sales)}</td>
                <td style={{ padding: '1rem', color: '#cbd5e1', fontSize: '0.9rem' }}>{fmtNumber(log.average_predicted_sales)}</td>
                <td style={{ padding: '1rem' }}>{getRiskBadge(log.stockout_risk)}</td>
                <td style={{ padding: '1rem', color: log.reorder_needed ? '#f59e0b' : '#10b981', fontSize: '0.9rem', fontWeight: 600 }}>
                  {log.reorder_needed ? 'Yes' : 'No'}
                </td>
                <td style={{ padding: '1rem', textAlign: 'right', display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
                  <button 
                    onClick={() => onView(log.id)}
                    className="btn"
                    style={{ background: 'rgba(6,182,212,0.1)', color: '#06b6d4', padding: '0.4rem 0.75rem' }}>
                    <Eye size={16} /> View
                  </button>
                  <button 
                    onClick={() => onDelete(log.id)}
                    className="btn"
                    style={{ background: 'rgba(239,68,68,0.1)', color: '#ef4444', padding: '0.4rem 0.75rem' }}>
                    <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
