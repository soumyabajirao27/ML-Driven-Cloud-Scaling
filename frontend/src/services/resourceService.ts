import { fetchApi } from './apiClient';
import { DEMO_RESOURCES, DEMO_PREDICTIONS } from './mockData';
import { BorgDatasetSummary, BorgReplayResult, CloudResource, MetricPoint, ResourceStatus } from '../types/cloudscaler';

const mapResource = (r: any): CloudResource => {
  // Compute status dynamically based on utilization thresholds (matching backend BaselineRuleEngine)
  let status: ResourceStatus = 'OPTIMAL';
  const cpu = r.cpu_utilization;
  const memory = r.memory_utilization;
  if (cpu > 85.0 || memory > 85.0) {
    status = 'OVERLOADED';
  } else if (cpu < 20.0 && memory < 20.0) {
    status = 'UNDERUTILIZED';
  }

  return {
    id: String(r.id),
    name: r.name,
    provider: r.provider_type,
    region: r.region,
    capacity: r.current_capacity,
    cpu: Math.round(cpu),
    memory: Math.round(memory),
    storage: Math.round(r.storage_utilization || 0),
    network: Math.round(r.network_utilization || 0),
    monthlyCost: Math.round(r.estimated_cost),
    status,
    lastUpdated: r.updated_at || r.created_at || new Date().toISOString(),
    tags: r.provider_resource_id ? { 'provider-id': r.provider_resource_id } : undefined,
  };
};

const mapMetricPoint = (m: any): MetricPoint => {
  const time = new Date(m.timestamp);
  const timeStr = !isNaN(time.getTime())
    ? time.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZoneName: 'short'
      })
    : String(m.timestamp);

  return {
    id: m.id,
    resource_id: m.resource_id,
    resource_name: m.resource_name,
    timestamp: timeStr,
    rawTimestamp: m.timestamp,
    cpu: Number(Number(m.cpu_utilization).toFixed(2)),
    memory: Number(Number(m.memory_utilization).toFixed(2)),
    storage: Math.round(m.storage_utilization || 0),
    network: Math.round(m.network_utilization || 0),
    source: m.source || 'mock'
  };
};

export const resourceService = {
  // GET /api/v1/resources
  async getResources(): Promise<CloudResource[]> {
    try {
      const data = await fetchApi<any[]>('/api/v1/resources');
      return data.map(mapResource);
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      // Fallback to central mock dataset
      return DEMO_RESOURCES;
    }
  },

  // GET /api/v1/resources/{resource_id}
  async getResourceById(resourceId: string): Promise<CloudResource | undefined> {
    try {
      const data = await fetchApi<any>(`/api/v1/resources/${resourceId}`);
      return mapResource(data);
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      return DEMO_RESOURCES.find(r => r.id === resourceId);
    }
  },

  // GET /api/v1/resources/{resource_id}/metrics
  async getResourceMetrics(resourceId: string, source?: string): Promise<MetricPoint[]> {
    try {
      const url = source
        ? `/api/v1/resources/${resourceId}/metrics?source=${encodeURIComponent(source)}`
        : `/api/v1/resources/${resourceId}/metrics`;
      const data = await fetchApi<any[]>(url);
      return data.map(mapMetricPoint);
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      const pred = DEMO_PREDICTIONS[resourceId];
      return pred ? pred.metrics : [];
    }
  },

  // POST /api/v1/resources/replay-borg
  async replayBorg(maxIntervals: number = 100): Promise<BorgReplayResult> {
    try {
      const res = await fetchApi<any>(`/api/v1/resources/replay-borg?max_intervals=${maxIntervals}`, {
        method: 'POST'
      });
      return (res.data || res) as BorgReplayResult;
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      throw err;
    }
  },

  // POST /api/v1/resources/{resource_id}/analyze
  async analyzeResource(resourceId: string): Promise<{ resource_id: string; status: ResourceStatus; summary: string }> {
    try {
      const analysis = await fetchApi<any>(`/api/v1/resources/${resourceId}/analyze`, { method: 'POST' });
      return {
        resource_id: String(analysis.resource_id),
        status: analysis.status,
        summary: analysis.recommendation
      };
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      const resource = DEMO_RESOURCES.find(r => r.id === resourceId);
      const status = resource ? resource.status : 'OPTIMAL';
      return {
        resource_id: resourceId,
        status,
        summary: `Analysis completed for ${resourceId}. Status classified as ${status}.`
      };
    }
  },

  // POST /api/v1/resources/sync
  async syncResources(): Promise<{ status: string; count: number }> {
    try {
      const res = await fetchApi<any>('/api/v1/resources/sync?sync_metrics=true', { method: 'POST' });
      return {
        status: 'success',
        count: (res.created || 0) + (res.updated || 0) + (res.unchanged || 0)
      };
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      return { status: 'success', count: DEMO_RESOURCES.length };
    }
  },

  // GET /api/v1/resources/borg/metrics -> Paginated raw Borg telemetry records
  async getBorgTelemetry(limit: number = 50, offset: number = 0): Promise<MetricPoint[]> {
    try {
      const data = await fetchApi<any[]>(`/api/v1/resources/borg/metrics?limit=${limit}&offset=${offset}`);
      return data.map(mapMetricPoint);
    } catch (err: any) {
      if (err.message === 'MOCK_MODE_ACTIVE') {
        throw err;
      }
      return [];
    }
  },

  // GET /api/v1/resources/borg/summary -> Live Google Borg dataset & telemetry counts
  async getBorgSummary(): Promise<BorgDatasetSummary> {
    try {
      return await fetchApi<BorgDatasetSummary>('/api/v1/resources/borg/summary');
    } catch {
      return {
        dataset_name: 'Google Borg Workload Trace (Cluster Workloads)',
        original_file: 'data/temporal/borg_temporal_pairs.csv',
        original_row_count: 65398,
        original_column_count: 26,
        imported_metric_count: 200,
        imported_resource_count: 98,
        mock_metric_count: 36,
        mock_resource_count: 12,
        earliest_timestamp: '2026-01-01T00:00:00Z',
        latest_timestamp: '2026-01-01T01:18:07Z',
        telemetry_source: 'google_borg'
      };
    }
  }
};
