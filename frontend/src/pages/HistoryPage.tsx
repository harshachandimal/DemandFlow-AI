import { useState, useEffect } from 'react';
import { getForecastLogs, getForecastLog, deleteForecastLog } from '../api/historyApi';
import { ForecastLogSummary, ForecastLog } from '../types';
import ForecastHistoryTable from '../components/history/ForecastHistoryTable';
import ForecastLogDetail from '../components/history/ForecastLogDetail';
import ServiceStatusBanner from '../components/layout/ServiceStatusBanner';
import LoadingState from '../components/ui/LoadingState';
import ErrorState from '../components/ui/ErrorState';
import EmptyState from '../components/ui/EmptyState';
import { Database } from 'lucide-react';

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
    return <LoadingState message="Loading forecast history..." minHeight="60vh" />;
  }

  return (
    <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '0', minHeight: 'calc(100vh - 64px)' }}>
      <ServiceStatusBanner />
      
      {error && (
        <div className="mb-8">
          <ErrorState message={error} title="History Error" />
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
            <EmptyState 
              icon={<Database size={48} />}
              title="No Forecast History"
              description="You haven't run any forecasts yet. Go to the dashboard and submit a new forecast to start building your history."
            />
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
          <LoadingState message="Loading details..." minHeight="200px" />
        </div>
      )}
    </main>
  );
}
