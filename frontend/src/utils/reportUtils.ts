import { ForecastResponse, ScenarioComparisonResponse } from '../types';

/**
 * Transforms forecast response into a clean structure for reporting.
 */
export const formatForecastReportData = (forecastResponse: ForecastResponse) => {
  return {
    report_type: 'Executive Forecast Report',
    generated_at: new Date().toISOString(),
    insights: forecastResponse.business_insights,
    forecast_data: forecastResponse.forecast,
  };
};

/**
 * Transforms scenario comparison response into a clean structure for reporting.
 */
export const formatScenarioReportData = (scenarioResponse: ScenarioComparisonResponse) => {
  return {
    report_type: 'Scenario Comparison Report',
    generated_at: new Date().toISOString(),
    best_scenario: scenarioResponse.best_scenario,
    scenarios: scenarioResponse.scenarios.map(s => ({
      scenario_name: s.scenario_name,
      business_insights: s.business_insights,
      forecast_data: s.forecast
    }))
  };
};

/**
 * Triggers a download of JSON data.
 */
export const downloadJsonReport = (data: any, filename: string) => {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Converts an array of objects to CSV format and triggers a download.
 */
export const downloadCsvReport = (rows: any[], filename: string) => {
  if (!rows || !rows.length) return;
  
  const headers = Object.keys(rows[0]);
  const csvContent = [
    headers.join(','),
    ...rows.map(row => 
      headers.map(header => {
        let val = row[header];
        if (val === null || val === undefined) val = '';
        const strVal = String(val);
        // Escape quotes and wrap in quotes if there are commas
        if (strVal.includes(',') || strVal.includes('"')) {
          return `"${strVal.replace(/"/g, '""')}"`;
        }
        return strVal;
      }).join(',')
    )
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Opens browser print dialog specifically targeting an element by ID.
 * To make this work best, CSS @media print should hide other elements.
 */
export const printReportSection = (elementId: string) => {
  // We handle the specific element highlighting via CSS @media print classes
  // The simplest reliable cross-browser way without iframes is to toggle a class on body
  // and trigger window.print()
  document.body.classList.add('printing-report');
  
  const target = document.getElementById(elementId);
  if (target) {
    target.classList.add('print-target');
  }

  // A small delay ensures styles are applied before print dialog opens
  setTimeout(() => {
    window.print();

    document.body.classList.remove('printing-report');
    if (target) {
      target.classList.remove('print-target');
    }
  }, 100);
};
