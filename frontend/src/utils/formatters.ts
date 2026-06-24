/** Format a number with thousands separators */
export const fmtNumber = (n?: number | string | null, decimals = 0): string =>
  Number(n ?? 0).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

/** Format currency (USD default) */
export const fmtCurrency = (n?: number | string | null, currency = 'USD'): string =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(Number(n ?? 0));

/** Format a date string to "Mon, 14 Jul" */
export const fmtDate = (dateStr?: string | null): string => {
  if (!dateStr) return '—';
  const d = new Date(dateStr + 'T00:00:00');
  return d.toLocaleDateString('en-US', { weekday: 'short', day: 'numeric', month: 'short' });
};

/** Short weekday from date string */
export const fmtWeekday = (dateStr?: string | null): string => {
  if (!dateStr) return '—';
  return new Date(dateStr + 'T00:00:00').toLocaleDateString('en-US', { weekday: 'long' });
};

/** Percentage formatter */
export const fmtPct = (n?: number | string | null, decimals = 1): string =>
  `${Number(n ?? 0).toFixed(decimals)}%`;
