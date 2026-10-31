import React, { useState, useEffect } from 'react';
import { CostOptimizationItem } from '../types/cloudscaler';
import { optimizationService } from '../services/optimizationService';
import { DEMO_TOTALS } from '../services/mockData';
import { SkeletonLoader } from '../components/common/SkeletonLoader';
import { ErrorState } from '../components/common/ErrorState';
import { AnimatedCounter } from '../components/common/AnimatedCounter';
import { formatINR } from '../utils/formatters';
import { DollarSign, Layers, TrendingDown, Sparkles, CheckCircle2, ArrowRight, ShieldAlert } from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell
} from 'recharts';

interface CostOptimizationViewProps {
  onNavigateView: (view: any) => void;
}

export const CostOptimizationView: React.FC<CostOptimizationViewProps> = ({ onNavigateView }) => {
  const [data, setData] = useState<{ items: CostOptimizationItem[]; totals: typeof DEMO_TOTALS } | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await optimizationService.getCostOptimization();
        setData(res);
      } catch {
        setError(true);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return <SkeletonLoader type="card" count={3} />;
  if (error || !data) return <ErrorState />;

  const costBarData = data.items.map((item) => ({
    name: item.resourceName,
    Cost: item.currentCost,
  }));

  // Group dynamic pie distribution by provider / region from actual connected items
  const providerCostMap = data.items.reduce((acc, item) => {
    const key = item.resourceName.startsWith('borg-') ? 'Google Borg (historical replay)' : 'Mock Provider (simulated)';
    acc[key] = (acc[key] || 0) + item.currentCost;
    return acc;
  }, {} as Record<string, number>);

  const pieData = Object.entries(providerCostMap).map(([name, value]) => ({
    name,
    value: Math.max(value, 1)
  }));
  const COLORS = ['#38bdf8', '#818cf8', '#34d399', '#f472b6'];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">Cost Optimization (FinOps)</h1>
        <p className="text-sm text-slate-400 mt-1">
          AI-driven cloud spend reduction and resource right-sizing in Indian Rupees (₹).
        </p>
      </div>

      {/* KPI Cards (Clean 2D Cards) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2 hover:border-slate-700 transition-all">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>TOTAL MONTHLY COST</span>
            <span className="font-bold text-emerald-400">₹</span>
          </div>
          <div className="text-3xl font-bold font-mono text-slate-100">
            <AnimatedCounter target={data.totals.totalMonthlyCost} isCurrency={true} />
          </div>
          <div className="text-[11px] text-slate-500 font-mono">{data.totals.totalResources} Connected Workloads</div>
        </div>

        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2 hover:border-slate-700 transition-all">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>TOTAL CAPACITY</span>
            <Layers className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-slate-100">
            <AnimatedCounter target={data.totals.totalCapacity} suffix=" Units" />
          </div>
          <div className="text-[11px] text-slate-500 font-mono">Provisioned Compute Units</div>
        </div>

        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2 hover:border-slate-700 transition-all">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>AVG COST / RESOURCE</span>
            <span className="font-bold text-indigo-400">₹</span>
          </div>
          <div className="text-3xl font-bold font-mono text-slate-100">
            <AnimatedCounter target={Math.round(data.totals.totalMonthlyCost / data.totals.totalResources)} isCurrency={true} />
          </div>
          <div className="text-[11px] text-slate-500 font-mono">Normalized Average</div>
        </div>

        <div className="p-5 rounded-xl bg-emerald-950/20 border border-emerald-800/40 space-y-2 hover:border-emerald-700/50 transition-all">
          <div className="flex justify-between items-center text-xs font-mono text-emerald-300 font-semibold">
            <span>POTENTIAL SAVINGS</span>
            <TrendingDown className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-emerald-400">
            <AnimatedCounter target={data.totals.potentialSavings} isCurrency={true} suffix="/mo" />
          </div>
          <div className="text-[11px] text-emerald-500 font-mono">Estimated Right-Sizing ROI</div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Cost by Resource Bar Chart */}
        <div className="lg:col-span-2 p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-slate-200 font-mono">Monthly Cost Distribution by Resource (₹)</h2>
          <div className="h-64 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={costBarData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis stroke="#94a3b8" tickFormatter={(v) => `₹${(v/1000).toFixed(0)}k`} tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <Tooltip
                  formatter={(value: number) => [formatINR(value), 'Monthly Cost']}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  itemStyle={{ color: '#f8fafc' }}
                />
                <Bar dataKey="Cost" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Spend by Provider Donut Chart */}
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-200 font-mono">Spend by Region (₹)</h2>
            <p className="text-xs text-slate-400">Mock Cloud regional breakdown.</p>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={4}>
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number) => [formatINR(value), 'Spend']}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-1 text-xs font-mono text-slate-400 border-t border-slate-800 pt-3">
            <div className="flex justify-between">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-sky-400"></span> us-east-1 (₹35,000)</span>
              <span>90%</span>
            </div>
            <div className="flex justify-between">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span> us-central1 (₹3,750)</span>
              <span>10%</span>
            </div>
          </div>
        </div>
      </div>

      {/* AI Cost Recommendations Section */}
      <div className="space-y-4">
        <h2 className="text-base font-bold text-slate-200 font-mono">AI FinOps Right-Sizing Opportunities</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {data.items.map((item) => (
            <div
              key={item.resourceId}
              className={`p-5 rounded-xl bg-[#0f172a] border space-y-4 transition-all ${
                item.potentialSavings > 0 ? 'border-amber-500/40 bg-amber-950/10' : 'border-slate-800'
              }`}
            >
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-slate-100 font-mono text-sm">{item.resourceName}</h3>
                {item.potentialSavings > 0 ? (
                  <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono text-[10px] font-bold">
                    Right-Sizing Opportunity
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono text-[10px]">
                    Optimized
                  </span>
                )}
              </div>

              <div className="text-xs font-mono space-y-1 text-slate-300 bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                <div className="flex justify-between">
                  <span>Current Utilization:</span>
                  <span className="font-bold">CPU {item.currentUtilization.cpu}% / Mem {item.currentUtilization.memory}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Current Monthly Spend:</span>
                  <span className="font-bold text-slate-100">{formatINR(item.currentCost)}/mo</span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-200 font-mono">AI Recommendation:</div>
                <p className="text-xs text-slate-300 leading-relaxed font-mono">{item.recommendation}</p>
              </div>

              {item.potentialSavings > 0 && (
                <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-between text-xs font-mono">
                  <span className="text-emerald-300">Est. Savings:</span>
                  <span className="text-emerald-400 font-bold text-sm">{formatINR(item.potentialSavings)}/month</span>
                </div>
              )}

              <button
                onClick={() => onNavigateView('recommendations')}
                className="w-full flex items-center justify-center gap-2 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-mono transition-all"
              >
                <span>View Recommendation Details</span>
                <ArrowRight className="w-3.5 h-3.5 text-sky-400" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
