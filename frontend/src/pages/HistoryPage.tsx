import { useState, useEffect } from 'react';
import { getForecastLogs, getForecastLog, deleteForecastLog } from '../api/historyApi';
import { ForecastLogSummary, ForecastLog } from '../types';
import ForecastHistoryTable from '../components/history/ForecastHistoryTable';
import ForecastLogDetail from '../components/history/ForecastLogDetail';
import HistoryEmptyState from '../components/history/HistoryEmptyState';
import { Loader, AlertTriangle } from 'lucide-react';

export default function HistoryPage() {
  const [logs, setLogs] = useState<ForecastLogSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedLog, setSelectedLog] = useState<ForecastLog | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const fetchLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getForecastLogs();
      setLogs(data);
    } catch (e: any) {
      setError("Unable to load forecast history. Please check that Laravel is running.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const handleView = async (id: number) => {
    setLoadingDetail(true);
    setError(null);
    try {
      const data = await getForecastLog(id);
      setSelectedLog(data);
    } catch (e: any) {
      setError("Unable to load forecast details.");
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm("Are you sure you want to delete this forecast log?")) return;
    
    try {
      await deleteForecastLog(id);
      setLogs(logs.filter(log => log.id !== id));
      if (selectedLog?.id === id) {
        setSelectedLog(null);
      }
    } catch (e: any) {
      setError("Could not delete this forecast log. Please try again.");
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <Loader className="animate-spin" size={48} color="#06b6d4" style={{ marginBottom: '1rem' }} />
        <div style={{ color: '#94a3b8', fontSize: '1.1rem', fontWeight: 500 }}>Loading forecast history...</div>
      </div>
    );
  }

  return (
    <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '2rem 1.5rem', minHeight: 'calc(100vh - 64px)' }}>
      {error && (
        <div className="fade-up" style={{
          padding: '1rem 1.5rem', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: '0.75rem', color: '#fca5a5', marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '0.75rem'
        }}>
          <AlertTriangle size={20} />
          <div style={{ fontWeight: 500 }}>{error}</div>
        </div>
      )}

      {selectedLog ? (
        <ForecastLogDetail log={selectedLog} onBack={() => setSelectedLog(null)} />
      ) : (
        <div className="fade-up">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.02em' }}>
              Forecast History
            </h1>
          </div>

          {logs.length === 0 ? (
            <HistoryEmptyState />
          ) : (
            <ForecastHistoryTable logs={logs} onView={handleView} onDelete={handleDelete} />
          )}
        </div>
      )}

      {loadingDetail && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(15,23,42,0.8)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50
        }}>
          <div className="glass-card" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Loader className="animate-spin" size={32} color="#06b6d4" style={{ marginBottom: '1rem' }} />
            <div style={{ color: '#e2e8f0', fontWeight: 500 }}>Loading details...</div>
          </div>
        </div>
      )}
    </main>
  );
}
