import { ForecastResponse, ModelInfo } from '../../types';
import ReportActions from './ReportActions';
import { downloadCsvReport, downloadJsonReport, printReportSection, formatForecastReportData } from '../../utils/reportUtils';
import { fmtCurrency, fmtNumber } from '../../utils/formatters';

interface ForecastReportCardProps {
  reportId?: string;
  data: ForecastResponse;
  modelInfo?: ModelInfo | null;
}

export default function ForecastReportCard({ reportId = 'forecast-report', data, modelInfo }: ForecastReportCardProps) {
  const insights = data.business_insights;
  const forecast = data.forecast;
  const dateStr = new Date().toLocaleString();

  const handleExportCsv = () => {
    downloadCsvReport(forecast, `forecast-report-${Date.now()}.csv`);
  };

  const handleExportJson = () => {
    downloadJsonReport(formatForecastReportData(data), `forecast-report-${Date.now()}.json`);
  };

  const handlePrint = () => {
    printReportSection(reportId);
  };

  return (
    <div id={reportId} className="glass-card report-container" style={{ padding: '2rem', marginTop: '2rem', backgroundColor: 'var(--color-surface)' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--color-border)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#e2e8f0' }}>Executive Forecast Report</h2>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '0.25rem' }}>Generated on: {dateStr}</p>
        </div>
        <ReportActions onExportCsv={handleExportCsv} onExportJson={handleExportJson} onPrint={handlePrint} />
      </div>

      {/* Meta Info */}
      <div className="report-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
        <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '0.5rem' }}>
          <h3 style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#94a3b8', marginBottom: '0.5rem' }}>Model Details</h3>
          <p style={{ fontSize: '0.875rem' }}><strong>Name:</strong> {modelInfo?.model_name || 'RossmannEnhancedFutureAwareLSTM'}</p>
          <p style={{ fontSize: '0.875rem' }}><strong>Version:</strong> v{modelInfo?.model_version || '2.0'}</p>
          <p style={{ fontSize: '0.875rem' }}><strong>Store ID:</strong> {modelInfo?.store_id || 1}</p>
        </div>
        <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '0.5rem' }}>
          <h3 style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#94a3b8', marginBottom: '0.5rem' }}>Business Summary</h3>
          <p style={{ fontSize: '0.875rem' }}><strong>Total Predicted Sales:</strong> {fmtNumber(insights.total_predicted_sales)} units</p>
          <p style={{ fontSize: '0.875rem' }}><strong>Expected Revenue:</strong> {insights.expected_revenue ? fmtCurrency(insights.expected_revenue) : 'N/A'}</p>
          <p style={{ fontSize: '0.875rem' }}>
            <strong>Stockout Risk:</strong>{' '}
            <span style={{ color: insights.stockout_risk.toLowerCase() === 'high' ? '#ef4444' : insights.stockout_risk.toLowerCase() === 'medium' ? '#f59e0b' : '#10b981' }}>
              {insights.stockout_risk.toUpperCase()}
            </span>
          </p>
        </div>
      </div>

      {/* Recommendation */}
      <div style={{ padding: '1rem', background: 'rgba(6, 182, 212, 0.05)', borderLeft: '4px solid #06b6d4', borderRadius: '0.25rem', marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '0.875rem', fontWeight: 700, color: '#06b6d4', marginBottom: '0.25rem' }}>Recommendation</h3>
        <p style={{ fontSize: '0.875rem', color: '#e2e8f0' }}>
          {insights.recommendation || (insights.reorder_needed 
            ? `Reorder immediately. Recommended quantity: ${insights.recommended_reorder_quantity} units.` 
            : 'Stock levels are sufficient for this period.')}
        </p>
      </div>

      {/* Table */}
      <div>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem' }}>Forecast Breakdown</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--color-border)', color: '#94a3b8' }}>
                <th style={{ padding: '0.75rem 0.5rem' }}>Date</th>
                <th style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>Predicted Sales</th>
                <th style={{ padding: '0.75rem 0.5rem', textAlign: 'center' }}>Promo</th>
                <th style={{ padding: '0.75rem 0.5rem', textAlign: 'center' }}>Holiday</th>
              </tr>
            </thead>
            <tbody>
              {forecast.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '0.75rem 0.5rem' }}>{row.date}</td>
                  <td style={{ padding: '0.75rem 0.5rem', textAlign: 'right', fontWeight: 600 }}>{fmtNumber(row.predicted_sales)}</td>
                  <td style={{ padding: '0.75rem 0.5rem', textAlign: 'center' }}>{row.is_promo || row.promo ? '✅' : '—'}</td>
                  <td style={{ padding: '0.75rem 0.5rem', textAlign: 'center' }}>{row.is_school_holiday || row.school_holiday ? '✅' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
