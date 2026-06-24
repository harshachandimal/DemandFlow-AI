import { Award } from 'lucide-react';
import { ScenarioComparisonResponse } from '../../types';

export default function BestScenarioCard({ best }: { best: ScenarioComparisonResponse['best_scenario'] }) {
  if (!best) return null;

  return (
    <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '2rem', border: '1px solid rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.05)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <div style={{ padding: '1rem', background: 'rgba(16,185,129,0.2)', borderRadius: '50%' }}>
          <Award size={32} color="#10b981" />
        </div>
        <div>
          <h3 style={{ fontSize: '0.9rem', color: '#cbd5e1', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Recommended Scenario</h3>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#10b981' }}>{best.scenario_name}</div>
          <div style={{ fontSize: '0.9rem', color: '#94a3b8', marginTop: '0.25rem' }}>{best.reason}</div>
        </div>
      </div>
    </div>
  );
}
