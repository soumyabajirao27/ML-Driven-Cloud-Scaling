import {
  CloudResource,
  MetricPoint,
  PredictionSummary,
  ScalingRecommendation,
  CostOptimizationItem,
  ExplainableAIDecision,
  ModelAnalyticsData,
  SystemEvent
} from '../types/cloudscaler';

export const DEMO_RESOURCES: CloudResource[] = [
  {
    id: 'mock-app-vm-1',
    name: 'mock-app-vm-1',
    provider: 'Mock Cloud',
    region: 'us-central1',
    capacity: 1,
    cpu: 10,
    memory: 10,
    storage: 10,
    network: 5,
    monthlyCost: 3750, // ₹3,750/mo
    status: 'UNDERUTILIZED',
    lastUpdated: '2026-08-23T20:00:00Z',
    tags: { env: 'staging', app: 'background-worker' }
  },
  {
    id: 'mock-web-server-1',
    name: 'mock-web-server-1',
    provider: 'Mock Cloud',
    region: 'us-east-1',
    capacity: 2,
    cpu: 45,
    memory: 50,
    storage: 30,
    network: 20,
    monthlyCost: 10000, // ₹10,000/mo
    status: 'OPTIMAL',
    lastUpdated: '2026-08-23T20:00:00Z',
    tags: { env: 'production', tier: 'frontend' }
  },
  {
    id: 'mock-db-server-1',
    name: 'mock-db-server-1',
    provider: 'Mock Cloud',
    region: 'us-east-1',
    capacity: 4,
    cpu: 92,
    memory: 90,
    storage: 65,
    network: 35,
    monthlyCost: 25000, // ₹25,000/mo
    status: 'OVERLOADED',
    lastUpdated: '2026-08-23T20:00:00Z',
    tags: { env: 'production', tier: 'database', engine: 'postgresql' }
  }
];

export const DEMO_TOTALS = {
  totalResources: 3,
  totalMonthlyCost: 38750, // ₹3,750 + ₹10,000 + ₹25,000 = ₹38,750
  totalCapacity: 7, // 1 + 2 + 4
  potentialSavings: 2100, // Right-sizing mock-app-vm-1 from ₹3,750 to ₹1,650/mo (Savings = ₹2,100)
  systemHealth: 'Healthy (1 Action Required)'
};

// Generate 24 hours of time-series metric data for charts
const generateMetricHistory = (baseCpu: number, baseMem: number, trend: 'rising' | 'steady' | 'low'): MetricPoint[] => {
  const points: MetricPoint[] = [];
  const now = new Date();
  
  for (let i = 24; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 3600 * 1000);
    const timeStr = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    let cpuNoise = (Math.random() - 0.5) * 6;
    let memNoise = (Math.random() - 0.5) * 4;
    
    if (trend === 'rising') {
      const rise = (24 - i) * 1.5;
      cpuNoise += rise;
      memNoise += rise * 0.8;
    }
    
    const cpu = Math.min(100, Math.max(2, Math.round(baseCpu + cpuNoise)));
    const memory = Math.min(100, Math.max(2, Math.round(baseMem + memNoise)));
    
    let predictedCpu: number | undefined;
    let upperBoundCpu: number | undefined;
    let lowerBoundCpu: number | undefined;
    
    if (i <= 6) {
      predictedCpu = Math.min(100, Math.max(0, cpu + (trend === 'rising' ? 4 : 0)));
      upperBoundCpu = Math.min(100, predictedCpu + 7);
      lowerBoundCpu = Math.max(0, predictedCpu - 7);
    }

    points.push({
      timestamp: timeStr,
      cpu,
      memory,
      storage: trend === 'rising' ? 65 : 30,
      network: trend === 'rising' ? 35 : 15,
      predictedCpu,
      upperBoundCpu,
      lowerBoundCpu
    });
  }
  return points;
};

export const DEMO_PREDICTIONS: Record<string, PredictionSummary> = {
  'mock-db-server-1': {
    resourceId: 'mock-db-server-1',
    currentCpu: 92,
    predictedCpu: 97,
    confidence: 0.94,
    horizonHours: 24,
    modelVersion: 'RandomForestRegressor (random_forest_v1)',
    status: 'OVERLOADED',
    suggestedAction: 'High resource utilization detected. Increase resource capacity before demand causes performance degradation.',
    metrics: generateMetricHistory(75, 78, 'rising')
  },
  'mock-web-server-1': {
    resourceId: 'mock-web-server-1',
    currentCpu: 45,
    predictedCpu: 48,
    confidence: 0.91,
    horizonHours: 24,
    modelVersion: 'RandomForestRegressor (random_forest_v1)',
    status: 'OPTIMAL',
    suggestedAction: 'Resource utilization is currently within the recommended operating range. Maintain current provisioning.',
    metrics: generateMetricHistory(45, 50, 'steady')
  },
  'mock-app-vm-1': {
    resourceId: 'mock-app-vm-1',
    currentCpu: 10,
    predictedCpu: 12,
    confidence: 0.96,
    horizonHours: 24,
    modelVersion: 'RandomForestRegressor (random_forest_v1)',
    status: 'UNDERUTILIZED',
    suggestedAction: 'Resource is operating below the recommended utilization range. Consider right-sizing to reduce unnecessary cloud spend.',
    metrics: generateMetricHistory(10, 10, 'low')
  }
};

export const DEMO_SCALING_RECOMMENDATIONS: ScalingRecommendation[] = [
  {
    id: 'rec-1',
    resourceId: 'mock-db-server-1',
    resourceName: 'mock-db-server-1',
    currentStatus: 'OVERLOADED',
    cpuUtilization: 92,
    memoryUtilization: 90,
    recommendation: 'Scale Up Instance Capacity (Increase from 4 to 8 Units)',
    reason: 'Current utilization (92% CPU, 90% Memory) exceeds the configured safe operating threshold of 80% for over 3 consecutive hours.',
    priority: 'HIGH',
    impact: {
      performance: 'INCREASE',
      availability: 'INCREASE',
      cost: 'INCREASE'
    },
    estimatedCostChange: 12500 // +₹12,500
  },
  {
    id: 'rec-2',
    resourceId: 'mock-app-vm-1',
    resourceName: 'mock-app-vm-1',
    currentStatus: 'UNDERUTILIZED',
    cpuUtilization: 10,
    memoryUtilization: 10,
    recommendation: 'Downsize Capacity / Right-Size VM (Decrease from 1 Unit to Micro Tier)',
    reason: 'Average CPU utilization has remained under 15% for the past 7 days, indicating over-provisioned infrastructure.',
    priority: 'LOW',
    impact: {
      performance: 'OPTIMIZE',
      availability: 'STABLE',
      cost: 'DECREASE'
    },
    estimatedCostChange: -2100 // -₹2,100
  },
  {
    id: 'rec-3',
    resourceId: 'mock-web-server-1',
    resourceName: 'mock-web-server-1',
    currentStatus: 'OPTIMAL',
    cpuUtilization: 45,
    memoryUtilization: 50,
    recommendation: 'Maintain Provisioned Capacity (2 Units)',
    reason: 'Resource performance metrics show healthy utilization headroom within target SLO parameters (40-60%).',
    priority: 'MEDIUM',
    impact: {
      performance: 'MAINTAIN',
      availability: 'STABLE',
      cost: 'NEUTRAL'
    },
    estimatedCostChange: 0
  }
];

export const DEMO_COST_OPTIMIZATION: CostOptimizationItem[] = [
  {
    resourceId: 'mock-db-server-1',
    resourceName: 'mock-db-server-1',
    currentUtilization: { cpu: 92, memory: 90 },
    currentCost: 25000,
    recommendedSize: 'db.m5.2xlarge (High Capacity)',
    estimatedCost: 37500,
    potentialSavings: 0,
    recommendation: 'Scaling required for performance reliability',
    confidence: 'High'
  },
  {
    resourceId: 'mock-web-server-1',
    resourceName: 'mock-web-server-1',
    currentUtilization: { cpu: 45, memory: 50 },
    currentCost: 10000,
    recommendedSize: 'web.t3.medium (Current)',
    estimatedCost: 10000,
    potentialSavings: 0,
    recommendation: 'Right-sized — no cost action needed',
    confidence: 'High'
  },
  {
    resourceId: 'mock-app-vm-1',
    resourceName: 'mock-app-vm-1',
    currentUtilization: { cpu: 10, memory: 10 },
    currentCost: 3750,
    recommendedSize: 'app.t3.small (Right-sized)',
    estimatedCost: 1650,
    potentialSavings: 2100,
    recommendation: 'Right-size instance tier to eliminate idle capacity',
    confidence: 'High'
  }
];

export const DEMO_EXPLAINABLE_AI: Record<string, ExplainableAIDecision> = {
  'mock-db-server-1': {
    resourceId: 'mock-db-server-1',
    resourceName: 'mock-db-server-1',
    classifiedStatus: 'OVERLOADED',
    recommendation: 'Increase Capacity from 4 to 8 Units',
    confidenceScore: 0.94,
    primaryDrivers: [
      { feature: 'CPU Utilization (92%)', weight: 45, value: '92%', impact: 'High Risk' },
      { feature: 'Memory Utilization (90%)', weight: 35, value: '90%', impact: 'High Risk' },
      { feature: '24h Utilization Trend', weight: 12, value: '+14% / 3h', impact: 'High Risk' },
      { feature: 'Storage I/O Pressure', weight: 8, value: '65%', impact: 'Normal' }
    ],
    reasoningPath: [
      { step: 1, stage: 'Resource Metrics', label: 'Telemetry Collected', description: 'CPU 92%, Memory 90%, Network 35%', value: 'Sustained peak >85%' },
      { step: 2, stage: 'ML Demand Prediction', label: 'ML Forecast', description: 'Predicted CPU load in 6h is 97%', value: 'Critical Growth Trend' },
      { step: 3, stage: 'Status Classification', label: 'Threshold Rule Engine', description: 'Telemetry exceeded 80% threshold for >3h', value: 'OVERLOADED' },
      { step: 4, stage: 'AI Action Recommendation', label: 'FinOps & Auto-scaler', description: 'Generate capacity scale-up recommendation', value: 'Increase Capacity' }
    ]
  },
  'mock-app-vm-1': {
    resourceId: 'mock-app-vm-1',
    resourceName: 'mock-app-vm-1',
    classifiedStatus: 'UNDERUTILIZED',
    recommendation: 'Consider Right-Sizing Tier',
    confidenceScore: 0.96,
    primaryDrivers: [
      { feature: 'CPU Utilization (10%)', weight: 50, value: '10%', impact: 'Under-capacity' },
      { feature: 'Memory Utilization (10%)', weight: 40, value: '10%', impact: 'Under-capacity' },
      { feature: '7-Day Idle Percentage', weight: 10, value: '98% Idle', impact: 'Under-capacity' }
    ],
    reasoningPath: [
      { step: 1, stage: 'Resource Metrics', label: 'Telemetry Collected', description: 'CPU 10%, Memory 10%', value: 'Low Activity' },
      { step: 2, stage: 'ML Demand Prediction', label: 'ML Forecast', description: 'Predicted CPU in 24h is 12%', value: 'No Spikes Expected' },
      { step: 3, stage: 'Status Classification', label: 'Threshold Rule Engine', description: 'Telemetry consistently below 20%', value: 'UNDERUTILIZED' },
      { step: 4, stage: 'AI Action Recommendation', label: 'FinOps Optimization', description: 'Propose right-sizing from ₹3,750/mo to ₹1,650/mo', value: 'Right-Size VM' }
    ]
  },
  'mock-web-server-1': {
    resourceId: 'mock-web-server-1',
    resourceName: 'mock-web-server-1',
    classifiedStatus: 'OPTIMAL',
    recommendation: 'Maintain Current Provisioning',
    confidenceScore: 0.91,
    primaryDrivers: [
      { feature: 'CPU Utilization (45%)', weight: 40, value: '45%', impact: 'Normal' },
      { feature: 'Memory Utilization (50%)', weight: 40, value: '50%', impact: 'Normal' },
      { feature: 'Traffic Stability', weight: 20, value: 'Nominal', impact: 'Normal' }
    ],
    reasoningPath: [
      { step: 1, stage: 'Resource Metrics', label: 'Telemetry Collected', description: 'CPU 45%, Memory 50%', value: 'Target Headroom' },
      { step: 2, stage: 'ML Demand Prediction', label: 'ML Forecast', description: 'Predicted CPU in 24h is 48%', value: 'Stable Range' },
      { step: 3, stage: 'Status Classification', label: 'Threshold Rule Engine', description: 'Utilization within 30%-70% band', value: 'OPTIMAL' },
      { step: 4, stage: 'AI Action Recommendation', label: 'System Supervisor', description: 'Maintain active provisioning parameters', value: 'No Action Needed' }
    ]
  }
};

export const DEMO_MODEL_ANALYTICS: ModelAnalyticsData = {
  modelVersion: 'RandomForestRegressor (random_forest_v1)',
  activeStatus: 'Active & Healthy',
  predictionAccuracy: 90.7, // R²: 0.9071 from offline model comparison benchmark
  mae: 0.0006, // Mean Absolute Error: 0.000615 from Phase 4 benchmark
  rmse: 0.0019, // Root Mean Squared Error: 0.001899 from Phase 4 benchmark
  trainingSamples: 65398, // Verified row count in Google Borg temporal dataset
  lastTrained: 'Offline Phase 4 Benchmark / random_forest_v1',
  pipelineStages: [
    { id: 1, title: 'Historical Metrics', description: 'Stream CPU, memory, storage & network telemetry into time-series store', iconName: 'Database' },
    { id: 2, title: 'Feature Engineering', description: 'Extract sliding window statistics, exponential moving averages & Fourier seasonality', iconName: 'Cpu' },
    { id: 3, title: 'ML Model', description: 'RandomForest model trained on multi-variate cloud workloads', iconName: 'Brain' },
    { id: 4, title: 'Demand Prediction', description: 'Project next-step resource utilization with estimated deviation bounds', iconName: 'TrendingUp' },
    { id: 5, title: 'AI Optimization', description: 'Compare forecasted workload against SLA constraints and cost boundaries', iconName: 'Sparkles' },
    { id: 6, title: 'Scaling Recommendation', description: 'Deliver actionable, explainable right-sizing or capacity increase plan', iconName: 'CheckCircle2' }
  ]
};

export const DEMO_SYSTEM_EVENTS: SystemEvent[] = [
  {
    id: 'evt-1',
    timestamp: '10 mins ago',
    type: 'ANALYSIS',
    title: 'Resource Analysis Completed',
    description: 'AI analyzed mock-db-server-1 and classified status as OVERLOADED (CPU: 92%, Mem: 90%).',
    severity: 'CRITICAL',
    resourceId: 'mock-db-server-1'
  },
  {
    id: 'evt-2',
    timestamp: '25 mins ago',
    type: 'RECOMMENDATION',
    title: 'Scaling Recommendation Generated',
    description: 'High priority recommendation created: Scale up capacity for mock-db-server-1.',
    severity: 'WARNING',
    resourceId: 'mock-db-server-1'
  },
  {
    id: 'evt-3',
    timestamp: '1 hour ago',
    type: 'PREDICTION',
    title: 'Demand Forecast Updated',
    description: 'ML model predicted 97% peak CPU demand for mock-db-server-1 in next 6h.',
    severity: 'WARNING',
    resourceId: 'mock-db-server-1'
  },
  {
    id: 'evt-4',
    timestamp: '2 hours ago',
    type: 'RECOMMENDATION',
    title: 'FinOps Right-Sizing Opportunity',
    description: 'AI identified underutilized mock-app-vm-1 (₹2,100/mo potential cost savings).',
    severity: 'INFO',
    resourceId: 'mock-app-vm-1'
  },
  {
    id: 'evt-5',
    timestamp: '4 hours ago',
    type: 'UTILIZATION',
    title: 'Telemetry Sync Completed',
    description: '3 connected cloud resources synced successfully with 0 errors.',
    severity: 'SUCCESS'
  }
];
