import React from 'react';
import Header from './Header';

interface AppShellProps {
  children: React.ReactNode;
}

export default function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen" style={{ background: 'var(--color-bg)' }}>
      {/* Decorative ambient blobs */}
      <div
        aria-hidden="true"
        style={{
          position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
          background:
            'radial-gradient(ellipse 60% 40% at 10% 0%, rgba(6,182,212,0.07) 0%, transparent 70%), ' +
            'radial-gradient(ellipse 50% 40% at 90% 100%, rgba(16,185,129,0.06) 0%, transparent 70%)',
        }}
      />
      <div style={{ position: 'relative', zIndex: 1 }}>
        <Header />
        <main
          style={{
            maxWidth: '1400px',
            margin: '0 auto',
            padding: '2rem 1.5rem 4rem',
          }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
