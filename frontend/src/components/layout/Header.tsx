import { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';

export default function Header() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const timeStr = time.toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });

  return (
    <header
      style={{
        borderBottom: '1px solid var(--color-border)',
        background: 'rgba(11,15,26,0.85)',
        backdropFilter: 'blur(12px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}
    >
      <div
        style={{
          maxWidth: '1400px', margin: '0 auto',
          padding: '0 1.5rem',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          height: '64px',
        }}
      >
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div
            style={{
              width: '36px', height: '36px', borderRadius: '10px',
              background: 'linear-gradient(135deg, #06b6d4, #10b981)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 16px rgba(6,182,212,0.4)',
            }}
          >
            <Activity size={20} color="#fff" />
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.1rem', color: '#e2e8f0', letterSpacing: '-0.02em' }}>
              DemandFlow <span className="gradient-text">AI</span>
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)', letterSpacing: '0.08em' }}>
              DEMAND FORECASTING PLATFORM
            </div>
          </div>
        </div>

        {/* Right area: live clock + Store badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>Live Clock</div>
            <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#06b6d4', fontVariantNumeric: 'tabular-nums' }}>
              {timeStr}
            </div>
          </div>
          <div
            style={{
              padding: '0.3rem 0.75rem', borderRadius: '9999px',
              background: 'rgba(6,182,212,0.12)',
              border: '1px solid rgba(6,182,212,0.25)',
              fontSize: '0.75rem', fontWeight: 600, color: '#06b6d4',
            }}
          >
            Store #1 · Rossmann
          </div>
        </div>
      </div>
    </header>
  );
}
