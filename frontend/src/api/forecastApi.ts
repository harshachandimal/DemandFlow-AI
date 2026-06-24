import axiosClient from './axiosClient';
import { HealthStatus, ModelInfo, ForecastPayload, ForecastResponse } from '../types';

/** GET /api/ml/health */
export const getMlHealth = (): Promise<HealthStatus> =>
  axiosClient.get<HealthStatus>('/ml/health').then((r) => r.data);

/** GET /api/ml/model-info */
export const getModelInfo = (): Promise<ModelInfo> =>
  axiosClient.get<ModelInfo>('/ml/model-info').then((r) => r.data);

/**
 * POST /api/ml/forecast/store-1
 * @param payload - Forecast request body
 */
export const forecastStoreOne = (payload: ForecastPayload): Promise<ForecastResponse> =>
  axiosClient.post<ForecastResponse>('/ml/forecast/store-1', payload).then((r) => r.data);
