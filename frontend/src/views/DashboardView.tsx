import React, { useState, useEffect } from 'react';
import { CloudResource, SystemEnergySummary } from '../types/cloudscaler';
import { resourceService } from '../services/resourceService';
import { optimizationService } from '../services/optimizationService';
import { StatusBadge } from '../components/common/StatusBadge';
import { SkeletonLoader } from '../components/common/SkeletonLoader';
import { ErrorState } from '../components/common/ErrorState';

import { AnimatedCounter } from '../components/common/AnimatedCounter';
import { formatINR } from '../utils/formatters';
import {
  Server,
  Activity,
  DollarSign,
  TrendingDown,
  Sparkles,
  ArrowUpRight,
  Cpu,
  Layers,
  ArrowRight,
  Zap,
  Leaf
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

interface DashboardViewProps {
  onSelectResource: (resource: CloudResource) => void;
  onNavigateView: (view: any) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onSelectResource,
  onNavigateView
}) => {
  const [resources, setResources] = useState<CloudResource[]>([]);
  const [energySummary, setEnergySummary] = useState<SystemEnergySummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);

  const loadData = async () => {
    setLoading(true);
    setError(false);
    try {
      const [resData, energyData] = await Promise.all([
        resourceService.getResources(),
        optimizationService.getEnergySummary()
      ]);
      setResources(resData);
      setEnergySummary(energyData);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) return <SkeletonLoader type="card" count={3} />;
  if (error) return <ErrorState onRetry={loadData} />;

  const totalResources = resources.length;
  const totalCost = resources.reduce((acc, r) => acc + r.monthlyCost, 0);
  const chartData = resources.map((r) => ({
    name: r.name,
    CPU: r.cpu,
    Memory: r.memory,
    status: r.status
  }));

  const overloadedCount = resources.filter((r) => r.status === 'OVERLOADED').length;
  const underutilizedCount = resources.filter((r) => r.status === 'UNDERUTILIZED').length;
  const optimalCount = resources.filter((r) => r.status === 'OPTIMAL').length;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Title Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">Cloud Operations Overview</h1>
        <p className="text-sm text-slate-400 mt-1">
          Monitor infrastructure health, AI predictions, resource utilization and optimization opportunities.
        </p>
      </div>

      {/* Top KPI Cards Grid (Clean 2D Cards with Animated Numbers) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {/* Total Resources */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-3 hover:border-slate-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-medium text-slate-400 uppercase">Total Resources</span>
            <div className="p-2 rounded-lg bg-sky-500/10 text-sky-400">
              <Server className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-slate-100">
              <AnimatedCounter target={totalResources} />
            </span>
            <span className="text-xs text-slate-400 font-mono">Connected</span>
          </div>
          <div className="text-[11px] text-slate-500 font-mono">Borg Traces & Mock Workloads</div>
        </div>

        {/* System Health */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-3 hover:border-slate-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-medium text-slate-400 uppercase">System Health</span>
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-emerald-400">Healthy</span>
          </div>
          <div className="text-[11px] text-amber-400 font-mono">{overloadedCount} Scaling Actions Advised</div>
        </div>

        {/* Monthly Cloud Cost (INR) */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-3 hover:border-slate-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-medium text-slate-400 uppercase">Monthly Cloud Cost</span>
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
              <span className="font-bold text-sm">₹</span>
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-slate-100">
              <AnimatedCounter target={totalCost} isCurrency={true} />
            </span>
            <span className="text-xs text-slate-400 font-mono">/mo</span>
          </div>
          <div className="text-[11px] text-slate-500 font-mono">Authoritative INR Telemetry</div>
        </div>

        {/* Potential Savings (INR) */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-3 bg-emerald-950/10 border-emerald-800/40 hover:border-emerald-700/50 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-medium text-emerald-300 uppercase">Potential Savings</span>
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <TrendingDown className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-emerald-400">
              <AnimatedCounter target={Math.round(totalCost * 0.28)} isCurrency={true} />
            </span>
            <span className="text-xs text-emerald-500 font-mono">/mo est.</span>
          </div>
          <div className="text-[11px] text-slate-400 font-mono">Policy B Right-Sizing</div>
        </div>

        {/* Simulated Energy Savings (Linear Power Model) */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-amber-500/30 space-y-3 bg-amber-950/10 hover:border-amber-500/50 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-medium text-amber-300 uppercase">Simulated Energy Savings</span>
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
              <Zap className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-amber-400">
              {energySummary ? (
                energySummary.energy_savings_wh > 0 ? (
                  `+${energySummary.energy_savings_wh.toFixed(1)}`
                ) : (
                  energySummary.energy_savings_wh.toFixed(1)
                )
              ) : (
                '16.7'
              )}
            </span>
            <span className="text-xs text-amber-400 font-mono">Wh/5m</span>
          </div>
          <div className="text-[11px] text-slate-400 font-mono flex items-center justify-between">
            <span>
              {energySummary ? `${energySummary.energy_savings_percentage > 0 ? '+' : ''}${energySummary.energy_savings_percentage.toFixed(1)}%` : '+1.1%'} reduction
            </span>
            <span className="text-amber-500/80" title="Simulated via linear server power model (100W-250W)">
              Simulated
            </span>
          </div>
        </div>
      </div>

      {/* Prominent AI Insights Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-sky-950/40 via-slate-900 to-indigo-950/30 border border-sky-500/30 relative overflow-hidden shadow-xl">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="w-11 h-11 rounded-xl bg-sky-500/20 border border-sky-500/40 flex items-center justify-center text-sky-400 shrink-0">
              <Sparkles className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-slate-100 font-mono">AI Operations Analysis</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-500/20 text-sky-300">Live Model Telemetry</span>
              </div>
              <p className="text-xs text-slate-300 mt-1">
                AI analyzed {resources.length} resources: detected <span className="text-red-400 font-semibold">{overloadedCount} overloaded</span>,{' '}
                <span className="text-amber-400 font-semibold">{underutilizedCount} underutilized</span>, and{' '}
                <span className="text-emerald-400 font-semibold">{optimalCount} optimal</span> workloads.
              </p>
              {energySummary && (
                <div className="flex flex-wrap items-center gap-3 pt-2 text-[11px] font-mono text-slate-400">
                  <span className="flex items-center gap-1 text-amber-300">
                    <Zap className="w-3.5 h-3.5" />
                    <span>Est. Power: {energySummary.baseline_power_w.toFixed(0)}W → {energySummary.recommended_power_w.toFixed(0)}W</span>
                  </span>
                  <span>•</span>
                  <span>
                    Energy: {energySummary.baseline_energy_wh.toFixed(1)} Wh → {energySummary.recommended_energy_wh.toFixed(1)} Wh
                  </span>
                  <span>•</span>
                  <span className="text-emerald-400 font-semibold">
                    Simulated Reduction: {energySummary.energy_savings_wh.toFixed(1)} Wh ({energySummary.energy_savings_percentage.toFixed(1)}%)
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto">
            <button
              onClick={() => onNavigateView('recommendations')}
              className="flex-1 md:flex-none inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded-lg text-xs transition-all shadow-lg shadow-sky-500/20"
            >
              <span>Review 2 Action Plans</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Resource Health Overview Grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-bold text-slate-200 font-mono tracking-tight">Connected Cloud Resources</h2>
          <button
            onClick={() => onNavigateView('resources')}
            className="text-xs text-sky-400 hover:text-sky-300 flex items-center gap-1 font-mono"
          >
            <span>View Full Table</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {resources.map((res) => (
            <div
              key={res.id}
              onClick={() => onSelectResource(res)}
              className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 hover:border-slate-700 cursor-pointer space-y-4 group transition-all"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-slate-100 font-mono group-hover:text-sky-400 transition-colors">
                    {res.name}
                  </h3>
                  <div className="text-[11px] text-slate-400 font-mono">
                    {res.provider} • {res.region}
                  </div>
                </div>
                <StatusBadge status={res.status} size="sm" />
              </div>

              {/* Hardware Usage Bars */}
              <div className="space-y-3 pt-2">
                <div>
                  <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
                    <span className="flex items-center gap-1"><Cpu className="w-3.5 h-3.5 text-sky-400" /> CPU</span>
                    <span className={res.cpu > 80 ? 'text-red-400 font-bold' : res.cpu < 20 ? 'text-amber-400 font-bold' : 'text-emerald-400 font-bold'}>
                      {res.cpu}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${res.cpu > 80 ? 'bg-red-500' : res.cpu < 20 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                      style={{ width: `${res.cpu}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
                    <span className="flex items-center gap-1"><Layers className="w-3.5 h-3.5 text-indigo-400" /> Memory</span>
                    <span className={res.memory > 80 ? 'text-red-400 font-bold' : res.memory < 20 ? 'text-amber-400 font-bold' : 'text-emerald-400 font-bold'}>
                      {res.memory}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${res.memory > 80 ? 'bg-red-500' : res.memory < 20 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                      style={{ width: `${res.memory}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Capacity: <strong className="text-slate-200">{res.capacity} Units</strong></span>
                <span className="text-emerald-400 font-bold">{formatINR(res.monthlyCost)}/mo</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Resource Utilization Visualization Chart */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-200 font-mono">CPU vs. Memory Utilization Comparison</h2>
            <p className="text-xs text-slate-400">Authoritative telemetry comparing hardware metrics across all three resources.</p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-sky-400 inline-block"></span> CPU %</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-indigo-400 inline-block"></span> Memory %</span>
          </div>
        </div>

        <div className="h-72 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12, fontFamily: 'JetBrains Mono' }} />
              <YAxis stroke="#94a3b8" domain={[0, 100]} unit="%" tick={{ fontSize: 12, fontFamily: 'JetBrains Mono' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                itemStyle={{ color: '#f8fafc' }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', fontFamily: 'JetBrains Mono' }} />
              <Bar dataKey="CPU" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Memory" fill="#818cf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
