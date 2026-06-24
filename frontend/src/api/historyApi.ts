import axiosClient from './axiosClient';
import { ForecastLogSummary, ForecastLog } from '../types';

/** GET /api/ml/forecast-logs */
export const getForecastLogs = (): Promise<ForecastLogSummary[]> =>
  axiosClient.get<ForecastLogSummary[]>('/ml/forecast-logs').then((r) => r.data);

/** GET /api/ml/forecast-logs/{id} */
export const getForecastLog = (id: number): Promise<ForecastLog> =>
  axiosClient.get<ForecastLog>(`/ml/forecast-logs/${id}`).then((r) => r.data);

/** DELETE /api/ml/forecast-logs/{id} */
export const deleteForecastLog = (id: number): Promise<{ message: string }> =>
  axiosClient.delete<{ message: string }>(`/ml/forecast-logs/${id}`).then((r) => r.data);
