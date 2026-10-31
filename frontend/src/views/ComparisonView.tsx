import React, { useState, useEffect } from 'react';
import { CloudResource } from '../types/cloudscaler';
import { resourceService } from '../services/resourceService';
import { predictionService } from '../services/predictionService';
import { StatusBadge } from '../components/common/StatusBadge';
import { SkeletonLoader } from '../components/common/SkeletonLoader';
import { formatINR } from '../utils/formatters';
import { Columns, Cpu, Layers, HardDrive, Network, DollarSign, Server, Brain, Award, Info, ChevronRight } from 'lucide-react';
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

export const ComparisonView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'models' | 'resources'>('models');
  const [modelData, setModelData] = useState<any | null>(null);
  const [resources, setResources] = useState<CloudResource[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const load = async () => {
      try {
        const [res, comp] = await Promise.all([
          resourceService.getResources(),
          predictionService.getModelComparison()
        ]);
        setResources(res);
        setModelData(comp);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return <SkeletonLoader type="table" count={3} />;

  const chartData = modelData && modelData.models ? [
    {
      metric: 'MAE (×10⁻⁴)',
      'Random Forest': (modelData.models['Random Forest'].MAE * 10000).toFixed(2),
      'XGBoost': (modelData.models['XGBoost'].MAE * 10000).toFixed(2),
      'LSTM': (modelData.models['LSTM'].MAE * 10000).toFixed(2),
    },
    {
      metric: 'RMSE (×10⁻⁴)',
      'Random Forest': (modelData.models['Random Forest'].RMSE * 10000).toFixed(2),
      'XGBoost': (modelData.models['XGBoost'].RMSE * 10000).toFixed(2),
      'LSTM': (modelData.models['LSTM'].RMSE * 10000).toFixed(2),
    },
    {
      metric: 'R² Score (%)',
      'Random Forest': (modelData.models['Random Forest'].R2 * 100).toFixed(1),
      'XGBoost': (modelData.models['XGBoost'].R2 * 100).toFixed(1),
      'LSTM': (modelData.models['LSTM'].R2 * 100).toFixed(1),
    }
  ] : [];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header with Switcher */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">Evaluation & Comparison</h1>
          <p className="text-sm text-slate-400 mt-1">
            Empirical benchmark of predictive models (Phase 4) and live workload telemetry matrix.
          </p>
        </div>

        {/* Tab switch */}
        <div className="flex items-center bg-[#0f172a] p-1 rounded-xl border border-slate-800 font-mono text-xs">
          <button
            onClick={() => setActiveTab('models')}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'models'
                ? 'bg-sky-500 text-slate-950 font-bold shadow-md shadow-sky-500/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Brain className="w-3.5 h-3.5" />
            <span>ML Models Comparison</span>
          </button>
          <button
            onClick={() => setActiveTab('resources')}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'resources'
                ? 'bg-sky-500 text-slate-950 font-bold shadow-md shadow-sky-500/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Server className="w-3.5 h-3.5" />
            <span>Resource Matrix</span>
          </button>
        </div>
      </div>

      {activeTab === 'models' && modelData && (
        <div className="space-y-6">
          {/* Models KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 font-mono">
            {/* Random Forest */}
            <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                <span className="text-xs text-sky-400 font-bold uppercase tracking-wider">Tabular Ensemble</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">Phase 2B</span>
              </div>
              <h3 className="text-lg font-bold text-slate-100">Random Forest</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                200 Decision Trees operating on temporal tabular pairs (t → t+1) with 18 input features.
              </p>
              <div className="space-y-2 pt-2 text-xs border-t border-slate-800">
                <div className="flex justify-between">
                  <span className="text-slate-400">Offline Test MAE:</span>
                  <span className="font-bold text-sky-400">{modelData.models['Random Forest'].MAE}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Offline Test RMSE:</span>
                  <span className="font-bold text-sky-400">{modelData.models['Random Forest'].RMSE}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">R² Coefficient:</span>
                  <span className="font-bold text-emerald-400">{modelData.models['Random Forest'].R2}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Inference Latency:</span>
                  <span className="text-slate-300">{modelData.models['Random Forest'].infer_ms_per_sample} ms/sample</span>
                </div>
              </div>
            </div>

            {/* XGBoost */}
            <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                <span className="text-xs text-indigo-400 font-bold uppercase tracking-wider">Gradient Boosted</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">Phase 2B</span>
              </div>
              <h3 className="text-lg font-bold text-slate-100">XGBoost Regressor</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Gradient boosted decision trees (lr=0.05, max_depth=6) evaluated on identical 6,540 test pairs.
              </p>
              <div className="space-y-2 pt-2 text-xs border-t border-slate-800">
                <div className="flex justify-between">
                  <span className="text-slate-400">Offline Test MAE:</span>
                  <span className="font-bold text-indigo-400">{modelData.models['XGBoost'].MAE}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Offline Test RMSE:</span>
                  <span className="font-bold text-indigo-400">{modelData.models['XGBoost'].RMSE}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">R² Coefficient:</span>
                  <span className="font-bold text-emerald-400">{modelData.models['XGBoost'].R2}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Inference Latency:</span>
                  <span className="text-slate-300">{modelData.models['XGBoost'].infer_ms_per_sample} ms/sample</span>
                </div>
              </div>
            </div>

            {/* LSTM (Phase 3) */}
            <div className="p-6 rounded-2xl bg-[#0f172a] border border-amber-500/30 space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                <span className="text-xs text-amber-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Award className="w-3.5 h-3.5" /> Selected Primary
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-amber-950/40 border border-amber-800/40 text-amber-300">Phase 3</span>
              </div>
              <h3 className="text-lg font-bold text-slate-100">Stacked LSTM Neural Net</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Sequential recurrent network (64 → 32 units) over 3D sequence tensors ($5 \times 7$), achieving lowest RMSE.
              </p>
              <div className="space-y-2 pt-2 text-xs border-t border-slate-800">
                <div className="flex justify-between">
                  <span className="text-slate-400">Offline Test MAE:</span>
                  <span className="font-bold text-amber-300">{modelData.models['LSTM'].MAE}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Offline Test RMSE:</span>
                  <span className="font-bold text-amber-300">{modelData.models['LSTM'].RMSE}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">R² Coefficient:</span>
                  <span className="font-bold text-emerald-400">{modelData.models['LSTM'].R2}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Inference Latency:</span>
                  <span className="text-slate-300">{modelData.models['LSTM'].infer_ms_per_sample} ms/sample</span>
                </div>
              </div>
            </div>
          </div>

          {/* Comparison Bar Chart */}
          <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-slate-200 font-mono">Empirical Benchmark Metrics Comparison</h3>
            <p className="text-xs text-slate-400 font-mono">
              Lower MAE/RMSE is better. Higher R² is better. Evaluated on unscaled Borg CPU fractions (Phase 4).
            </p>
            <div className="h-72 w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="metric" stroke="#94a3b8" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                  <YAxis stroke="#94a3b8" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', fontFamily: 'JetBrains Mono' }} />
                  <Bar dataKey="Random Forest" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="XGBoost" fill="#818cf8" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="LSTM" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Evaluation Fairness Notice */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs font-mono text-slate-400 space-y-1">
            <div className="flex items-center gap-1.5 text-slate-300 font-bold">
              <Info className="w-4 h-4 text-sky-400" />
              <span>Scientific Evaluation Context & Fairness Assessment</span>
            </div>
            <p className="leading-relaxed">
              {modelData.fairness_assessment} Models predict <code className="text-sky-300">target_future_cpu_avg</code> on a chronological 80/10/10 train/validation/test split without future data leakage. Tabular models evaluated on 6,540 test pairs; LSTM on 1,560 sequence tensors.
            </p>
          </div>
        </div>
      )}

      {activeTab === 'resources' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {resources.map((res) => (
            <div
              key={res.id}
              className={`p-6 rounded-2xl bg-[#0f172a] border space-y-6 transition-all ${
                res.status === 'OVERLOADED'
                  ? 'border-red-500/40 bg-red-950/10'
                  : res.status === 'UNDERUTILIZED'
                  ? 'border-amber-500/30 bg-amber-950/10'
                  : 'border-slate-800'
              }`}
            >
              {/* Header */}
              <div className="border-b border-slate-800 pb-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-slate-400 uppercase">{res.provider}</span>
                  <StatusBadge status={res.status} size="sm" />
                </div>
                <h3 className="text-lg font-bold text-slate-100 font-mono">{res.name}</h3>
                <div className="text-xs text-slate-400 font-mono">Region: {res.region}</div>
              </div>

              {/* Metrics List */}
              <div className="space-y-4 font-mono text-xs">
                {/* CPU */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-400 flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5 text-sky-400" /> CPU Utilization</span>
                    <span className={`font-bold ${res.cpu > 80 ? 'text-red-400' : res.cpu < 20 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {res.cpu}%
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${res.cpu > 80 ? 'bg-red-500' : res.cpu < 20 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                      style={{ width: `${Math.min(100, res.cpu)}%` }}
                    />
                  </div>
                </div>

                {/* Memory */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-400 flex items-center gap-1.5"><Layers className="w-3.5 h-3.5 text-indigo-400" /> Memory Utilization</span>
                    <span className={`font-bold ${res.memory > 80 ? 'text-red-400' : res.memory < 20 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {res.memory}%
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${res.memory > 80 ? 'bg-red-500' : res.memory < 20 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                      style={{ width: `${Math.min(100, res.memory)}%` }}
                    />
                  </div>
                </div>

                {/* Storage */}
                <div className="flex justify-between p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 flex items-center gap-1.5"><HardDrive className="w-3.5 h-3.5 text-purple-400" /> Storage Usage</span>
                  <span className="font-bold text-slate-200">{res.storage}%</span>
                </div>

                {/* Network */}
                <div className="flex justify-between p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 flex items-center gap-1.5"><Network className="w-3.5 h-3.5 text-teal-400" /> Network Bandwidth</span>
                  <span className="font-bold text-slate-200">{res.network}%</span>
                </div>

                {/* Provisioned Capacity */}
                <div className="flex justify-between p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 flex items-center gap-1.5"><Server className="w-3.5 h-3.5 text-sky-400" /> Capacity</span>
                  <span className="font-bold text-sky-400">{res.capacity} Units</span>
                </div>

                {/* Monthly Cost (INR) */}
                <div className="flex justify-between p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 flex items-center gap-1.5"><span className="text-emerald-400 font-bold">₹</span> Monthly Cost</span>
                  <span className="font-bold text-emerald-400 text-sm">{formatINR(res.monthlyCost)}/mo</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
