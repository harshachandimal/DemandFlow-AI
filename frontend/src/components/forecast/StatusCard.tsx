
import { CheckCircle, XCircle, Loader } from 'lucide-react';
import { HealthStatus } from '../../types';

interface ItemProps {
  label: string;
  ok: boolean;
  loading?: boolean;
}

const Item: React.FC<ItemProps> = ({ label, ok, loading }) => (
  <div
    style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0.85rem 1rem',
      background: 'rgba(26,34,53,0.6)',
      borderRadius: '0.75rem',
      border: `1px solid ${ok ? 'rgba(16,185,129,0.2)' : loading ? 'rgba(99,179,237,0.15)' : 'rgba(239,68,68,0.2)'}`,
    }}
  >
    <span style={{ fontSize: '0.875rem', color: '#94a3b8', fontWeight: 500 }}>{label}</span>
    {loading ? (
      <Loader size={18} color="#06b6d4" style={{ animation: 'spin 1s linear infinite' }} />
    ) : ok ? (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#10b981' }}>
        <CheckCircle size={18} />
        <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>Online</span>
      </div>
    ) : (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#ef4444' }}>
        <XCircle size={18} />
        <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>Offline</span>
      </div>
    )}
  </div>
);

interface StatusCardProps {
  health: HealthStatus | null;
  loading: boolean;
  error: string | null;
}

export default function StatusCard({ health, loading, error }: StatusCardProps) {
  const laravelOk = !error && !!health;
  const mlOk = health?.status === 'ok' || health?.model_loaded === true;

  return (
    <div className="glass-card" style={{ padding: '1.25rem' }}>
      <div style={{ marginBottom: '1rem' }}>
        <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--color-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Service Status
        </div>
        <div style={{ fontSize: '1rem', fontWeight: 700, color: '#e2e8f0', marginTop: '0.15rem' }}>
          System Health
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <Item label="Laravel API Gateway" ok={laravelOk} loading={loading} />
        <Item label="ML Inference Engine" ok={mlOk} loading={loading} />
      </div>

      {error && (
        <div
          style={{
            marginTop: '0.75rem', padding: '0.65rem 0.875rem',
            background: 'rgba(239,68,68,0.08)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: '0.5rem',
            fontSize: '0.78rem', color: '#f87171', lineHeight: 1.5,
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
