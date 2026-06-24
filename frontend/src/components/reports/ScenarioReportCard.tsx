import { ScenarioComparisonResponse } from '../../types';
import ReportActions from './ReportActions';
import { downloadCsvReport, downloadJsonReport, printReportSection, formatScenarioReportData } from '../../utils/reportUtils';
import { fmtCurrency, fmtNumber } from '../../utils/formatters';

interface ScenarioReportCardProps {
  reportId?: string;
  data: ScenarioComparisonResponse;
}

export default function ScenarioReportCard({ reportId = 'scenario-report', data }: ScenarioReportCardProps) {
  const dateStr = new Date().toLocaleString();
  const scenarios = data.scenarios || [];

  const handleExportCsv = () => {
    // Flatten scenario data for CSV
    const rows = scenarios.map(s => ({
      Scenario: s.scenario_name,
      'Total Sales': s.business_insights.total_predicted_sales,
      'Revenue': s.business_insights.expected_revenue || '',
      'Stockout Risk': s.business_insights.stockout_risk,
      'Reorder Needed': s.business_insights.reorder_needed ? 'Yes' : 'No',
      'Recommended Qty': s.business_insights.recommended_reorder_quantity || 0,
    }));
    downloadCsvReport(rows, `scenario-report-${Date.now()}.csv`);
  };

  const handleExportJson = () => {
    downloadJsonReport(formatScenarioReportData(data), `scenario-report-${Date.now()}.json`);
  };

  const handlePrint = () => {
    printReportSection(reportId);
  };

  return (
    <div id={reportId} className="glass-card report-container" style={{ padding: '2rem', marginTop: '2rem', backgroundColor: 'var(--color-surface)' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--color-border)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#e2e8f0' }}>Scenario Comparison Report</h2>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '0.25rem' }}>Generated on: {dateStr} · {data.scenario_count} scenarios</p>
        </div>
        <ReportActions onExportCsv={handleExportCsv} onExportJson={handleExportJson} onPrint={handlePrint} />
      </div>

      {/* Best Scenario Highlight */}
      {data.best_scenario && (
        <div style={{ padding: '1.5rem', background: 'rgba(16, 185, 129, 0.05)', borderLeft: '4px solid #10b981', borderRadius: '0.5rem', marginBottom: '2rem' }}>
          <h3 style={{ fontSize: '0.875rem', fontWeight: 700, color: '#10b981', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Recommended Strategy</h3>
          <p style={{ fontSize: '1.125rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '0.25rem' }}>{data.best_scenario.scenario_name}</p>
          <p style={{ fontSize: '0.875rem', color: '#cbd5e1' }}>{data.best_scenario.reason}</p>
        </div>
      )}

      {/* Comparison Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        {scenarios.map((s, i) => (
          <div key={i} style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '0.5rem', border: s.scenario_name === data.best_scenario?.scenario_name ? '1px solid #10b981' : '1px solid transparent' }}>
            <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '1rem' }}>{s.scenario_name}</h4>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Sales:</span>
                <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{fmtNumber(s.business_insights.total_predicted_sales)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Revenue:</span>
                <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{s.business_insights.expected_revenue ? fmtCurrency(s.business_insights.expected_revenue) : 'N/A'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Stockout Risk:</span>
                <span style={{ fontSize: '0.875rem', fontWeight: 600, color: s.business_insights.stockout_risk.toLowerCase() === 'high' ? '#ef4444' : s.business_insights.stockout_risk.toLowerCase() === 'medium' ? '#f59e0b' : '#10b981' }}>
                  {s.business_insights.stockout_risk.toUpperCase()}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Reorder:</span>
                <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{s.business_insights.reorder_needed ? 'Yes' : 'No'}</span>
              </div>
              {s.business_insights.reorder_needed && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Rec. Qty:</span>
                  <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{fmtNumber(s.business_insights.recommended_reorder_quantity || 0)}</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
