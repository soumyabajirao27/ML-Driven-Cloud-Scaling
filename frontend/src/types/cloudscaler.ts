export type ResourceStatus = 'UNDERUTILIZED' | 'OPTIMAL' | 'OVERLOADED';

export type PriorityLevel = 'HIGH' | 'MEDIUM' | 'LOW';

export interface CloudResource {
  id: string;
  name: string;
  provider: string;
  region: string;
  capacity: number;
  cpu: number; // Percentage
  memory: number; // Percentage
  storage: number; // Percentage
  network: number; // Percentage
  monthlyCost: number; // USD
  status: ResourceStatus;
  lastUpdated: string;
  tags?: Record<string, string>;
}

export interface MetricPoint {
  id?: number;
  resource_id?: number;
  resource_name?: string;
  timestamp: string;
  rawTimestamp?: string;
  cpu: number;
  memory: number;
  storage?: number;
  network?: number;
  source?: string;
  predictedCpu?: number;
  upperBoundCpu?: number;
  lowerBoundCpu?: number;
}

export interface BorgReplayResult {
  dataset_source: string;
  rows_selected: number;
  rows_imported: number;
  rows_skipped: number;
  duplicates_ignored: number;
  timestamp_range: {
    start: string | null;
    end: string | null;
  };
  resources_created: number;
  resources_updated: number;
  success: boolean;
  message: string;
}

export interface BorgDatasetSummary {
  dataset_name: string;
  original_file: string;
  original_row_count: number;
  original_column_count: number;
  imported_metric_count: number;
  imported_resource_count: number;
  mock_metric_count: number;
  mock_resource_count: number;
  earliest_timestamp: string | null;
  latest_timestamp: string | null;
  telemetry_source: string;
}

export interface PredictionProvenance {
  model_id: string;
  model_type: string;
  prediction_target: string;
  input_data_source: string;
  is_fallback: boolean;
  sufficient_history: boolean;
  sample_count_used: number;
  model_version: string;
}

export interface PredictionSummary {
  resourceId: string;
  currentCpu: number;
  predictedCpu: number;
  confidence: number; // e.g. 0.94 -> 94%
  horizonHours: number;
  modelVersion: string;
  status: ResourceStatus;
  suggestedAction: string;
  provenance?: PredictionProvenance;
  metrics: MetricPoint[];
}

export interface CostEstimate {
  baseline_cost: number;
  recommended_cost: number;
  simulated_cost_difference: number;
  simulated_cost_percentage_change: number;
  unit: string;
}

export interface EnergyEstimate {
  baseline_power_w: number;
  recommended_power_w: number;
  baseline_energy_wh: number;
  recommended_energy_wh: number;
  simulated_energy_percentage_change: number;
  unit: string;
  power_model: string;
}

export interface SystemEnergySummary {
  total_resources: number;
  baseline_power_w: number;
  recommended_power_w: number;
  baseline_energy_wh: number;
  recommended_energy_wh: number;
  energy_savings_wh: number;
  energy_savings_kwh: number;
  energy_savings_percentage: number;
  evaluation_interval_sec: number;
  power_model: string;
  is_simulated: boolean;
}

export interface ScalingRecommendation {
  id: string;
  resourceId: string;
  resourceName: string;
  currentStatus: ResourceStatus;
  cpuUtilization: number;
  memoryUtilization: number;
  recommendation: string;
  reason: string;
  priority: PriorityLevel;
  impact: {
    performance: 'INCREASE' | 'MAINTAIN' | 'OPTIMIZE';
    availability: 'INCREASE' | 'STABLE';
    cost: 'INCREASE' | 'DECREASE' | 'NEUTRAL';
  };
  estimatedCostChange?: number; // +/- dollar amount
  actionType?: 'SCALE_UP' | 'SCALE_DOWN' | 'MAINTAIN' | 'DEFERRED_COOLDOWN' | 'NO_ACTION_INSUFFICIENT_DATA';
  policyId?: string;
  safeguardStatus?: string;
  cooldownRemainingSteps?: number;
  allocatedCapacity?: number;
  costEstimate?: CostEstimate;
  energyEstimate?: EnergyEstimate;
  isSimulated?: boolean;
}

export interface CostOptimizationItem {
  resourceId: string;
  resourceName: string;
  currentUtilization: {
    cpu: number;
    memory: number;
  };
  currentCost: number;
  recommendedSize: string;
  estimatedCost: number;
  potentialSavings: number;
  recommendation: string;
  confidence: 'High' | 'Medium' | 'Low';
}

export interface FeatureContribution {
  feature: string;
  weight: number; // percentage or impact ratio
  value: string | number;
  impact: 'High Risk' | 'Normal' | 'Under-capacity' | 'Neutral';
}

export interface ReasoningStep {
  step: number;
  stage: string;
  label: string;
  description: string;
  value: string;
}

export interface ExplainableAIDecision {
  resourceId: string;
  resourceName: string;
  classifiedStatus: ResourceStatus;
  recommendation: string;
  confidenceScore: number;
  primaryDrivers: FeatureContribution[];
  reasoningPath: ReasoningStep[];
}

export interface ModelInfo {
  model_id: string;
  model_type: string;
  model_version: string;
  target: string;
  expected_features_count: number;
  status: string;
  is_available: boolean;
}

export interface ModelAnalyticsData {
  modelVersion: string;
  activeStatus: string;
  predictionAccuracy: number;
  mae: number;
  rmse: number;
  trainingSamples: number;
  lastTrained: string;
  pipelineStages: {
    id: number;
    title: string;
    description: string;
    iconName: string;
  }[];
}

export interface SystemEvent {
  id: string;
  timestamp: string;
  type: 'ANALYSIS' | 'PREDICTION' | 'UTILIZATION' | 'RECOMMENDATION';
  title: string;
  description: string;
  severity: 'INFO' | 'WARNING' | 'CRITICAL' | 'SUCCESS';
  resourceId?: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: {
    email: string;
    name: string;
    role: string;
  } | null;
  token: string | null;
}
