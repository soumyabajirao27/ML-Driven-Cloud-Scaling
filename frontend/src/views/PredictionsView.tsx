import React, { useState, useEffect } from 'react';
import { CloudResource, PredictionSummary } from '../types/cloudscaler';
import { resourceService } from '../services/resourceService';
import { predictionService } from '../services/predictionService';
import { StatusBadge } from '../components/common/StatusBadge';
import { SkeletonLoader } from '../components/common/SkeletonLoader';
import { ErrorState } from '../components/common/ErrorState';

import { AnimatedCounter } from '../components/common/AnimatedCounter';
import { Brain, Cpu, TrendingUp, ShieldCheck, Sparkles, Clock, AlertTriangle } from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

export const PredictionsView: React.FC = () => {
  const [resources, setResources] = useState<CloudResource[]>([]);
  const [selectedResourceId, setSelectedResourceId] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    const init = async () => {
      try {
        const res = await resourceService.getResources();
        setResources(res);
        if (res.length > 0) {
          const targetId = selectedResourceId && res.some(r => r.id === selectedResourceId)
            ? selectedResourceId
            : res[0].id;
          
          if (targetId !== selectedResourceId) {
            setSelectedResourceId(targetId);
          }

          const pred = await predictionService.getPredictionForResource(targetId);
          setPrediction(pred);
        }
      } catch {
        setError(true);
      } finally {
        setLoading(false);
      }
    };
    init();
  }, [selectedResourceId]);

  if (loading) return <SkeletonLoader type="chart" count={1} />;
  if (error || !prediction) return <ErrorState />;

  const currentRes = resources.find((r) => r.id === selectedResourceId);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">AI Demand Predictions</h1>
          <p className="text-sm text-slate-400 mt-1">
            Forecast future resource demand using machine learning.
          </p>
        </div>

        {/* Resource Selector Buttons & Dropdown */}
        <div className="flex items-center gap-2 bg-[#0f172a] p-1.5 rounded-xl border border-slate-800 font-mono text-xs max-w-full overflow-x-auto">
          {resources.slice(0, 8).map((r) => (
            <button
              key={r.id}
              onClick={() => setSelectedResourceId(r.id)}
              className={`px-3 py-1.5 rounded-lg whitespace-nowrap font-medium transition-all ${
                selectedResourceId === r.id
                  ? 'bg-sky-500 text-slate-950 font-bold shadow-md shadow-sky-500/20'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              {r.name}
            </button>
          ))}
          {resources.length > 8 && (
            <select
              value={selectedResourceId}
              onChange={(e) => setSelectedResourceId(e.target.value)}
              className="bg-slate-900 border border-slate-800 text-slate-300 rounded px-2.5 py-1 text-xs focus:outline-none"
            >
              {resources.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} ({r.provider})
                </option>
              ))}
            </select>
          )}
        </div>
      </div>


      {/* Prediction Summary KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
          <span className="text-[11px] font-mono text-slate-400 uppercase">Current CPU</span>
          <div className="text-2xl font-bold font-mono text-slate-100">
            <AnimatedCounter target={prediction.currentCpu} suffix="%" />
          </div>
          <div className="text-[10px] text-slate-500 font-mono">Live Telemetry</div>
        </div>

        <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
          <span className="text-[11px] font-mono text-slate-400 uppercase">Predicted CPU (+1h)</span>
          <div className={`text-2xl font-bold font-mono ${prediction.predictedCpu > 80 ? 'text-red-400' : 'text-sky-400'}`}>
            <AnimatedCounter target={prediction.predictedCpu} suffix="%" />
          </div>
          <div className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
            <TrendingUp className="w-3 h-3 text-sky-400" />
            {prediction.provenance?.is_fallback ? 'Baseline Heuristic' : `${prediction.provenance?.model_type || 'ML'} Forecast`}
          </div>
        </div>

        <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
          <span className="text-[11px] font-mono text-slate-400 uppercase">Model Confidence</span>
          <div className="text-2xl font-bold font-mono text-emerald-400">
            <AnimatedCounter target={Math.round(prediction.confidence * 100)} suffix="%" />
          </div>
          <div className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
            <ShieldCheck className="w-3 h-3 text-emerald-400" />
            {prediction.provenance?.sufficient_history ? 'Evaluated Signal' : 'Limited Sample History'}
          </div>
        </div>

        <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
          <span className="text-[11px] font-mono text-slate-400 uppercase">Forecast Horizon</span>
          <div className="text-2xl font-bold font-mono text-indigo-400">{prediction.horizonHours} Step Ahead</div>
          <div className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
            <Clock className="w-3 h-3 text-indigo-400" /> Next Interval Target
          </div>
        </div>

        <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
          <span className="text-[11px] font-mono text-slate-400 uppercase">ML Model Engine</span>
          <div className="text-sm font-bold font-mono text-slate-200 mt-1 truncate" title={prediction.modelVersion}>
            {prediction.modelVersion}
          </div>
          <div className={`text-[10px] font-mono ${prediction.provenance?.is_fallback ? 'text-amber-400' : 'text-emerald-400'}`}>
            {prediction.provenance?.is_fallback ? '⚠ Baseline Fallback Active' : '● Verified Live Model'}
          </div>
        </div>
      </div>

      {/* Time-Series Forecast Chart with Confidence Band */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-base font-bold text-slate-200 font-mono">Time-Series Forecast & Confidence Interval</h2>
              {currentRes && <StatusBadge status={currentRes.status} size="sm" />}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Historical CPU telemetry vs. {prediction.modelVersion} predicted demand curve with estimated deviation bounds.
            </p>
          </div>
        </div>

        <div className="h-80 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={prediction.metrics} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="timestamp" stroke="#94a3b8" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
              <YAxis stroke="#94a3b8" domain={[0, 100]} unit="%" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                itemStyle={{ color: '#f8fafc' }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', fontFamily: 'JetBrains Mono' }} />
              <Area type="monotone" dataKey="upperBoundCpu" stroke="none" fill="#38bdf8" fillOpacity={0.15} name="Upper Bound" />
              <Area type="monotone" dataKey="lowerBoundCpu" stroke="none" fill="#090d16" fillOpacity={0.8} name="Lower Bound" />
              <Line type="monotone" dataKey="cpu" stroke="#38bdf8" strokeWidth={2.5} dot={false} name="Actual CPU %" />
              <Line type="monotone" dataKey="predictedCpu" stroke="#f43f5e" strokeWidth={2} strokeDasharray="4 4" dot={true} name="Predicted CPU %" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* AI Analysis Summary Banner */}
      <div className={`p-6 rounded-2xl border ${
        prediction.status === 'OVERLOADED'
          ? 'bg-red-950/20 border-red-800/40 text-red-200'
          : prediction.status === 'UNDERUTILIZED'
          ? 'bg-amber-950/20 border-amber-800/40 text-amber-200'
          : 'bg-emerald-950/20 border-emerald-800/40 text-emerald-200'
      }`}>
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 shrink-0">
            <Sparkles className="w-6 h-6 text-sky-400" />
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <h3 className="text-base font-bold font-mono">AI Demand Analysis ({selectedResourceId})</h3>
              <StatusBadge status={prediction.status} size="sm" />
            </div>
            <p className="text-xs leading-relaxed font-mono">
              {prediction.suggestedAction}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
