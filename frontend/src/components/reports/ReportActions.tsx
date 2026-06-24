import { FileText, Download, Printer } from 'lucide-react';

interface ReportActionsProps {
  onExportCsv?: () => void;
  onExportJson?: () => void;
  onPrint?: () => void;
}

export default function ReportActions({ onExportCsv, onExportJson, onPrint }: ReportActionsProps) {
  const btnStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
    padding: '0.4rem 0.8rem',
    borderRadius: '0.5rem',
    fontSize: '0.8rem',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
  };

  return (
    <div className="report-actions hide-on-print" style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
      {onExportCsv && (
        <button
          onClick={onExportCsv}
          style={{ ...btnStyle, backgroundColor: 'rgba(16, 185, 129, 0.1)', color: '#10b981', border: '1px solid rgba(16, 185, 129, 0.2)' }}
        >
          <FileText size={14} /> CSV
        </button>
      )}
      
      {onExportJson && (
        <button
          onClick={onExportJson}
          style={{ ...btnStyle, backgroundColor: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6', border: '1px solid rgba(59, 130, 246, 0.2)' }}
        >
          <Download size={14} /> JSON
        </button>
      )}

      {onPrint && (
        <button
          onClick={onPrint}
          style={{ ...btnStyle, backgroundColor: 'rgba(148, 163, 184, 0.1)', color: '#cbd5e1', border: '1px solid rgba(148, 163, 184, 0.2)' }}
        >
          <Printer size={14} /> Print / PDF
        </button>
      )}
    </div>
  );
}
