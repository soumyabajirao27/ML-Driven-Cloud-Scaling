import React, { useState, useEffect, useCallback } from 'react';
import { BorgDatasetSummary, BorgReplayResult, CloudResource, MetricPoint, ResourceStatus } from '../types/cloudscaler';
import { resourceService } from '../services/resourceService';
import { StatusBadge } from '../components/common/StatusBadge';
import { SkeletonLoader } from '../components/common/SkeletonLoader';
import { ErrorState } from '../components/common/ErrorState';
import { EmptyState } from '../components/common/EmptyState';
import { formatINR } from '../utils/formatters';
import { Search, RefreshCw, Eye, Sparkles, Database, FileText, CheckCircle2, AlertCircle, Clock, ChevronLeft, ChevronRight } from 'lucide-react';

interface ResourcesViewProps {
  onSelectResource: (resource: CloudResource) => void;
}

export const ResourcesView: React.FC<ResourcesViewProps> = ({ onSelectResource }) => {
  const [resources, setResources] = useState<CloudResource[]>([]);
  const [borgSummary, setBorgSummary] = useState<BorgDatasetSummary | null>(null);
  const [activeTab, setActiveTab] = useState<'resources' | 'borg_explorer'>('resources');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [providerFilter, setProviderFilter] = useState<string>('ALL');
  const [syncing, setSyncing] = useState<boolean>(false);
  const [replayingBorg, setReplayingBorg] = useState<boolean>(false);
  const [replayResult, setReplayResult] = useState<BorgReplayResult | null>(null);
  const [replayNotification, setReplayNotification] = useState<string | null>(null);

  // Borg telemetry records state
  const [borgMetrics, setBorgMetrics] = useState<MetricPoint[]>([]);
  const [borgMetricsLoading, setBorgMetricsLoading] = useState<boolean>(false);
  const [borgMetricsError, setBorgMetricsError] = useState<string | null>(null);
  const [borgMetricsPage, setBorgMetricsPage] = useState<number>(1);
  const borgMetricsPerPage = 20;

  // Pagination for Borg workloads table
  const [currentPage, setCurrentPage] = useState<number>(1);
  const itemsPerPage = 10;

  const loadData = async () => {
    setLoading(true);
    setError(false);
    try {
      const [resList, summary] = await Promise.all([
        resourceService.getResources(),
        resourceService.getBorgSummary()
      ]);
      setResources(resList);
      setBorgSummary(summary);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  const loadBorgTelemetry = useCallback(async (page: number) => {
    setBorgMetricsLoading(true);
    setBorgMetricsError(null);
    try {
      const offset = (page - 1) * borgMetricsPerPage;
      const data = await resourceService.getBorgTelemetry(borgMetricsPerPage, offset);
      setBorgMetrics(data);
    } catch (err: any) {
      setBorgMetricsError(err.message || 'Failed to load Borg telemetry');
      setBorgMetrics([]);
    } finally {
      setBorgMetricsLoading(false);
    }
  }, [borgMetricsPerPage]);

  useEffect(() => {
    loadData();
  }, []);

  // Load Borg telemetry records when explorer tab is active or page changes
  useEffect(() => {
    if (activeTab === 'borg_explorer') {
      loadBorgTelemetry(borgMetricsPage);
    }
  }, [activeTab, borgMetricsPage, loadBorgTelemetry]);

  const handleSync = async () => {
    setSyncing(true);
    try {
      await resourceService.syncResources();
      await loadData();
    } finally {
      setSyncing(false);
    }
  };

  const handleReplayBorg = async () => {
    setReplayingBorg(true);
    setReplayNotification(null);
    try {
      const res = await resourceService.replayBorg(100);
      setReplayResult(res);
      const imported = res.rows_imported ?? 0;
      const skipped = res.duplicates_ignored ?? 0;
      setReplayNotification(
        `Borg Replay Run: ${imported} intervals imported, ${skipped} duplicate rows skipped (idempotent constraint preserved).`
      );
      await loadData();
    } catch (err: any) {
      setReplayNotification(`Failed to replay Borg dataset: ${err.message || 'Unknown error'}`);
    } finally {
      setReplayingBorg(false);
    }
  };

  const filteredResources = resources.filter((r) => {
    const matchesSearch = r.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          r.region.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          r.provider.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'ALL' || r.status === statusFilter;
    const matchesProvider = providerFilter === 'ALL' ||
                            (providerFilter === 'google_borg' ? r.provider === 'google_borg' : r.provider !== 'google_borg');
    return matchesSearch && matchesStatus && matchesProvider;
  });

  const borgWorkloads = resources.filter((r) => r.provider === 'google_borg');
  const totalBorgPages = Math.ceil(borgWorkloads.length / itemsPerPage) || 1;
  const paginatedBorgWorkloads = borgWorkloads.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  if (loading) return <SkeletonLoader type="table" count={3} />;
  if (error) return <ErrorState onRetry={loadData} />;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">Infrastructure & Telemetry</h1>
          <p className="text-sm text-slate-400 mt-1">
            Monitor real Google Borg workload traces and simulated cloud resources.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* View Mode Tabs */}
          <div className="flex items-center bg-[#0f172a] p-1 rounded-lg border border-slate-800 font-mono text-xs">
            <button
              onClick={() => setActiveTab('resources')}
              className={`px-3 py-1.5 rounded-md transition-all ${
                activeTab === 'resources'
                  ? 'bg-sky-500 text-slate-950 font-bold shadow-md shadow-sky-500/20'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All Workloads ({resources.length})
            </button>
            <button
              onClick={() => setActiveTab('borg_explorer')}
              className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                activeTab === 'borg_explorer'
                  ? 'bg-amber-500 text-slate-950 font-bold shadow-md shadow-amber-500/20'
                  : 'text-slate-400 hover:text-amber-300'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              <span>Google Borg Explorer</span>
            </button>
          </div>

          <button
            onClick={handleReplayBorg}
            disabled={replayingBorg}
            className="inline-flex items-center gap-2 px-3.5 py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 rounded-lg text-xs font-mono font-medium transition-all"
            title="Ingest up to 100 chronological intervals from Google Borg workload dataset (idempotent)"
          >
            <Database className={`w-3.5 h-3.5 ${replayingBorg ? 'animate-spin' : ''}`} />
            <span>{replayingBorg ? 'Replaying...' : 'Replay Borg (100)'}</span>
          </button>
          <button
            onClick={handleSync}
            disabled={syncing}
            className="inline-flex items-center gap-2 px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-mono font-medium transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${syncing ? 'animate-spin' : ''}`} />
            <span>{syncing ? 'Syncing...' : 'Sync Mock'}</span>
          </button>
        </div>
      </div>

      {replayNotification && (
        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300 flex items-center justify-between">
          <span>{replayNotification}</span>
          <button
            onClick={() => setReplayNotification(null)}
            className="text-slate-500 hover:text-slate-300 ml-4"
          >
            ✕
          </button>
        </div>
      )}

      {/* Google Borg Trace Explorer Panel */}
      {activeTab === 'borg_explorer' && borgSummary && (
        <div className="space-y-6 animate-fadeIn">
          {/* Dataset Provenance KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono">
            <div className="p-4 rounded-xl bg-slate-900 border border-amber-500/30 space-y-1">
              <span className="text-[10px] text-amber-400 font-semibold uppercase">SOURCE DATASET CSV</span>
              <div className="text-2xl font-bold text-slate-100">{borgSummary.original_row_count.toLocaleString()} rows</div>
              <div className="text-[11px] text-slate-400">{borgSummary.original_column_count} features / columns</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] text-sky-400 font-semibold uppercase">IMPORTED BORG METRICS</span>
              <div className="text-2xl font-bold text-sky-400">{borgSummary.imported_metric_count} records</div>
              <div className="text-[11px] text-slate-400">PostgreSQL (source: {borgSummary.telemetry_source})</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] text-emerald-400 font-semibold uppercase">BORG WORKLOAD ENTITIES</span>
              <div className="text-2xl font-bold text-emerald-400">{borgSummary.imported_resource_count} resources</div>
              <div className="text-[11px] text-slate-400">Isolated from {borgSummary.mock_resource_count} mock servers</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] text-purple-400 font-semibold uppercase">TELEMETRY TIME RANGE</span>
              <div className="text-xs font-bold text-slate-200 truncate">
                {borgSummary.earliest_timestamp
                  ? new Date(borgSummary.earliest_timestamp).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', timeZoneName: 'short' })
                  : 'N/A'}
                {' → '}
                {borgSummary.latest_timestamp
                  ? new Date(borgSummary.latest_timestamp).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', timeZoneName: 'short' })
                  : 'N/A'}
              </div>
              <div className="text-[11px] text-slate-400">Microsecond epoch traces</div>
            </div>
          </div>

          {/* Educational Replay Disclaimer Banner */}
          <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/40 text-amber-200 text-xs font-mono space-y-1">
            <div className="flex items-center gap-2 font-bold text-amber-400">
              <AlertCircle className="w-4 h-4" />
              <span>Idempotent Workload Ingestion Policy</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Google Borg traces are imported in sequential 100-interval chunks with a strict unique constraint on <code className="text-amber-300">(resource_id, timestamp)</code>. Re-running the replay control over an already imported segment will safely result in <strong className="text-slate-200">0 new records</strong> and <strong className="text-slate-200">100 skipped duplicates</strong>. This prevents data duplication and protects evaluation integrity.
            </p>
          </div>

          {/* Paginated Google Borg Workload Instances Table */}
          <div className="rounded-xl bg-[#0f172a] border border-slate-800 overflow-x-auto">
            <div className="p-4 border-b border-slate-800 flex justify-between items-center">
              <div>
                <h3 className="text-sm font-bold text-slate-200 font-mono">Imported Google Borg Workload Instances</h3>
                <p className="text-xs text-slate-400 font-mono">Showing {borgWorkloads.length} Borg resources with aggregated CPU & Memory</p>
              </div>
              <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                <span>Page {currentPage} of {totalBorgPages}</span>
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="p-1 rounded bg-slate-800 disabled:opacity-40"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalBorgPages, p + 1))}
                  disabled={currentPage === totalBorgPages}
                  className="p-1 rounded bg-slate-800 disabled:opacity-40"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#0b1120] text-slate-400 border-b border-slate-800 uppercase tracking-wider text-[11px]">
                <tr>
                  <th className="py-3 px-4 font-semibold">Instance ID</th>
                  <th className="py-3 px-4 font-semibold">Borg Collection / Region</th>
                  <th className="py-3 px-4 font-semibold">Current Capacity</th>
                  <th className="py-3 px-4 font-semibold">Trace CPU %</th>
                  <th className="py-3 px-4 font-semibold">Trace Memory %</th>
                  <th className="py-3 px-4 font-semibold">Status</th>
                  <th className="py-3 px-4 font-semibold text-right">Inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80">
                {paginatedBorgWorkloads.map((r) => (
                  <tr key={r.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 font-bold text-amber-300">{r.name}</td>
                    <td className="py-3 px-4 text-slate-300">{r.region}</td>
                    <td className="py-3 px-4 text-sky-400">{r.capacity} Units</td>
                    <td className="py-3 px-4 font-bold text-slate-200">{r.cpu}%</td>
                    <td className="py-3 px-4 font-bold text-slate-200">{r.memory}%</td>
                    <td className="py-3 px-4">
                      <StatusBadge status={r.status} size="sm" />
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => onSelectResource(r)}
                        className="inline-flex items-center gap-1 px-2.5 py-1 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded text-[11px]"
                      >
                        <Eye className="w-3 h-3" />
                        <span>Telemetry</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Raw Borg Telemetry Records Table */}
          <div className="rounded-xl bg-[#0f172a] border border-amber-500/20 overflow-x-auto">
            <div className="p-4 border-b border-slate-800 flex justify-between items-center">
              <div>
                <h3 className="text-sm font-bold text-amber-300 font-mono flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  Raw Google Borg Telemetry Records
                </h3>
                <p className="text-xs text-slate-400 font-mono">
                  Individual metric readings from PostgreSQL (source: google_borg) • {borgSummary.imported_metric_count} total records
                </p>
              </div>
              <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                {borgMetricsLoading && <span className="text-amber-400 animate-pulse">Loading…</span>}
                <span>Page {borgMetricsPage}</span>
                <button
                  onClick={() => setBorgMetricsPage(p => Math.max(1, p - 1))}
                  disabled={borgMetricsPage === 1 || borgMetricsLoading}
                  className="p-1 rounded bg-slate-800 disabled:opacity-40"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setBorgMetricsPage(p => p + 1)}
                  disabled={borgMetrics.length < borgMetricsPerPage || borgMetricsLoading}
                  className="p-1 rounded bg-slate-800 disabled:opacity-40"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {borgMetricsError && (
              <div className="p-4 text-xs text-red-400 font-mono bg-red-950/20 border-b border-red-800/30">
                <AlertCircle className="w-3.5 h-3.5 inline mr-1.5" />
                {borgMetricsError}
              </div>
            )}

            {borgMetricsLoading && borgMetrics.length === 0 ? (
              <div className="p-8 text-center text-slate-500 text-xs font-mono">
                <Database className="w-5 h-5 mx-auto mb-2 animate-spin text-amber-400" />
                Loading Borg telemetry records…
              </div>
            ) : borgMetrics.length === 0 && !borgMetricsLoading ? (
              <div className="p-8 text-center text-slate-500 text-xs font-mono">
                <Database className="w-5 h-5 mx-auto mb-2 text-slate-600" />
                No telemetry records found. Use Replay Borg to import records.
              </div>
            ) : (
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-[#0b1120] text-slate-400 border-b border-slate-800 uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="py-3 px-4 font-semibold">Metric ID</th>
                    <th className="py-3 px-4 font-semibold">Resource Name</th>
                    <th className="py-3 px-4 font-semibold">Timestamp</th>
                    <th className="py-3 px-4 font-semibold">CPU Utilization</th>
                    <th className="py-3 px-4 font-semibold">Memory Utilization</th>
                    <th className="py-3 px-4 font-semibold">Source</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  {borgMetrics.map((m, idx) => (
                    <tr key={m.id ?? idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-3 px-4 text-slate-500">{m.id ?? '—'}</td>
                      <td className="py-3 px-4 font-bold text-amber-300">{m.resource_name ?? `resource-${m.resource_id}`}</td>
                      <td className="py-3 px-4 text-slate-300">{m.timestamp}</td>
                      <td className="py-3 px-4 font-bold text-sky-400" title={m.cpu > 100 ? 'Borg CPU is in fractional units (may exceed 100%)' : undefined}>
                        {m.cpu}{m.cpu > 100 ? '' : '%'}
                        {m.cpu > 100 && <span className="ml-1 text-[10px] text-amber-500" title="Raw Borg fractional CPU units">⚡</span>}
                      </td>
                      <td className="py-3 px-4 font-bold text-emerald-400">{m.memory}%</td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded text-[10px] bg-amber-950/40 border border-amber-800/60 text-amber-300">
                          google_borg
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* Controls Bar for Standard Resources View */}
      {activeTab === 'resources' && (
      <>
      <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 flex flex-col md:flex-row gap-4 items-center justify-between">
        {/* Search */}
        <div className="relative w-full md:w-72">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search resources, regions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-4 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-mono"
          />
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-slate-400">Provider:</label>
            <select
              value={providerFilter}
              onChange={(e) => setProviderFilter(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-sky-500"
            >
              <option value="ALL">All Providers ({resources.length})</option>
              <option value="google_borg">Google Borg (Historical Trace)</option>
              <option value="mock">Mock Cloud (Simulated)</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-slate-400">Status:</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-sky-500"
            >
              <option value="ALL">All Statuses</option>
              <option value="UNDERUTILIZED">UNDERUTILIZED (Amber)</option>
              <option value="OPTIMAL">OPTIMAL (Green)</option>
              <option value="OVERLOADED">OVERLOADED (Red)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Table Container */}
      {filteredResources.length === 0 ? (
        <EmptyState
          title="No resources match filters"
          message="Try resetting your search query or status filter."
          actionLabel="Clear Filters"
          onAction={() => {
            setSearchQuery('');
            setStatusFilter('ALL');
          }}
        />
      ) : (
        <div className="rounded-xl bg-[#0f172a] border border-slate-800 overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#0b1120] text-slate-400 border-b border-slate-800 uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-3.5 px-4 font-semibold">Resource</th>
                <th className="py-3.5 px-4 font-semibold">Provider</th>
                <th className="py-3.5 px-4 font-semibold">Region</th>
                <th className="py-3.5 px-4 font-semibold">Capacity</th>
                <th className="py-3.5 px-4 font-semibold">CPU %</th>
                <th className="py-3.5 px-4 font-semibold">Memory %</th>
                <th className="py-3.5 px-4 font-semibold">Storage</th>
                <th className="py-3.5 px-4 font-semibold">Network</th>
                <th className="py-3.5 px-4 font-semibold">Status</th>
                <th className="py-3.5 px-4 font-semibold">Monthly Cost</th>
                <th className="py-3.5 px-4 font-semibold text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {filteredResources.map((r) => (
                <tr key={r.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-4 px-4 font-bold text-slate-100">{r.name}</td>
                  <td className="py-4 px-4">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                      r.provider === 'google_borg'
                        ? 'bg-amber-950/40 border-amber-800/60 text-amber-300'
                        : 'bg-slate-800/60 border-slate-700 text-slate-300'
                    }`}>
                      {r.provider === 'google_borg' ? 'Google Borg (Historical Trace)' : 'Mock Provider (Simulated)'}
                    </span>
                  </td>
                  <td className="py-4 px-4 text-slate-300">{r.region}</td>
                  <td className="py-4 px-4 text-sky-400 font-semibold">{r.capacity} Units</td>
                  <td className="py-4 px-4">
                    <span className={r.cpu > 80 ? 'text-red-400 font-bold' : r.cpu < 20 ? 'text-amber-400 font-bold' : 'text-emerald-400 font-bold'}>
                      {r.cpu}%
                    </span>
                  </td>
                  <td className="py-4 px-4">
                    <span className={r.memory > 80 ? 'text-red-400 font-bold' : r.memory < 20 ? 'text-amber-400 font-bold' : 'text-emerald-400 font-bold'}>
                      {r.memory}%
                    </span>
                  </td>
                  <td className="py-4 px-4 text-slate-400">{r.storage}%</td>
                  <td className="py-4 px-4 text-slate-400">{r.network}%</td>
                  <td className="py-4 px-4">
                    <StatusBadge status={r.status} size="sm" />
                  </td>
                  <td className="py-4 px-4 text-emerald-400 font-bold">{formatINR(r.monthlyCost)}</td>
                  <td className="py-4 px-4 text-right">
                    <button
                      onClick={() => onSelectResource(r)}
                      className="inline-flex items-center gap-1 px-3 py-1.5 bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 border border-sky-500/30 rounded-md transition-all text-xs font-semibold"
                    >
                      <Eye className="w-3.5 h-3.5" />
                      <span>Details</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      </>
      )}
    </div>
  );
};
