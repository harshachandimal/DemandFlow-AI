import axiosClient from './axiosClient';
import { ScenarioConfig, ScenarioComparisonResponse } from '../types';

/** POST /api/ml/forecast/scenarios */
export const compareScenarios = (scenarios: ScenarioConfig[]): Promise<ScenarioComparisonResponse> =>
  axiosClient.post<ScenarioComparisonResponse>('/ml/forecast/scenarios', { scenarios }).then((r) => r.data);
