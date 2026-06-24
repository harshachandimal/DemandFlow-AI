import { Database } from 'lucide-react';

export default function HistoryEmptyState() {
  return (
    <div className="glass-card" style={{ padding: '4rem 2rem', textAlign: 'center' }}>
      <div
        style={{
          width: '64px', height: '64px', borderRadius: '50%',
          background: 'rgba(51,65,85,0.4)', border: '1px solid rgba(255,255,255,0.05)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 1.5rem',
        }}
      >
        <Database size={32} color="#64748b" />
      </div>
      <h3 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '0.5rem' }}>
        No Forecast History
      </h3>
      <p style={{ color: '#94a3b8', maxWidth: '400px', margin: '0 auto' }}>
        You haven't run any forecasts yet. Go to the dashboard and submit a new forecast to start building your history.
      </p>
    </div>
  );
}
