import { Trash2 } from 'lucide-react';
import { ScenarioConfig } from '../../types';

interface Props {
  index: number;
  scenario: ScenarioConfig;
  onChange: (index: number, updated: ScenarioConfig) => void;
  onRemove: (index: number) => void;
}

export default function ScenarioFormCard({ index, scenario, onChange, onRemove }: Props) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = e.target;
    if (name === 'promo_dates') {
      const dates = value ? value.split(',').map(d => d.trim()) : [];
      onChange(index, { ...scenario, promo_dates: dates });
      return;
    }
    
    let val: any = value;
    if (type === 'number') val = value === '' ? undefined : Number(value);
    onChange(index, { ...scenario, [name]: val });
  };

  return (
    <div className="glass-card" style={{ padding: '1.25rem', position: 'relative' }}>
      <button 
        onClick={() => onRemove(index)}
        className="btn"
        style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'rgba(239,68,68,0.1)', color: '#ef4444', padding: '0.4rem' }}>
        <Trash2 size={16} />
      </button>
      
      <div style={{ marginBottom: '1rem', paddingRight: '2rem' }}>
        <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.25rem' }}>Scenario Name</label>
        <input 
          name="name" value={scenario.name} onChange={handleChange} 
          className="form-input" style={{ padding: '0.4rem 0.6rem', fontSize: '0.9rem' }} required
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
        <div>
          <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.25rem' }}>Current Stock</label>
          <input 
            type="number" name="current_stock" value={scenario.current_stock ?? ''} onChange={handleChange} 
            className="form-input" style={{ padding: '0.4rem 0.6rem', fontSize: '0.9rem' }}
          />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.25rem' }}>Unit Price ($)</label>
          <input 
            type="number" name="unit_price" value={scenario.unit_price ?? ''} onChange={handleChange} step="0.01"
            className="form-input" style={{ padding: '0.4rem 0.6rem', fontSize: '0.9rem' }}
          />
        </div>
      </div>

      <div>
        <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.25rem' }}>Promo Dates (YYYY-MM-DD, comma sep)</label>
        <input 
          name="promo_dates" value={scenario.promo_dates?.join(', ') || ''} onChange={handleChange} 
          className="form-input" style={{ padding: '0.4rem 0.6rem', fontSize: '0.9rem' }} placeholder="e.g. 2015-08-01, 2015-08-02"
        />
      </div>
    </div>
  );
}
