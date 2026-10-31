import React, { useState, useEffect } from 'react';
import { ScalingRecommendation } from '../types/cloudscaler';
import { optimizationService } from '../services/optimizationService';
import { StatusBadge } from '../components/common/StatusBadge';
import { SkeletonLoader } from '../components/common/SkeletonLoader';
import { ErrorState } from '../components/common/ErrorState';
import { formatINR } from '../utils/formatters';
import { Zap, AlertTriangle, ShieldCheck, ArrowRight, Sparkles, TrendingUp } from 'lucide-react';

interface RecommendationsViewProps {
  onNavigateView: (view: any, resourceId?: string) => void;
}

export const RecommendationsView: React.FC<RecommendationsViewProps> = ({ onNavigateView }) => {
  const [recommendations, setRecommendations] = useState<ScalingRecommendation[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await optimizationService.getScalingRecommendations();
        setRecommendations(res);
      } catch {
        setError(true);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return <SkeletonLoader type="card" count={3} />;
  if (error) return <ErrorState />;

  const highPriority = recommendations.filter((r) => r.priority === 'HIGH');
  const mediumPriority = recommendations.filter((r) => r.priority === 'MEDIUM');
  const lowPriority = recommendations.filter((r) => r.priority === 'LOW');

  const renderCard = (item: ScalingRecommendation) => (
    <div
      key={item.id}
      className={`p-6 rounded-2xl bg-[#0f172a] border space-y-4 transition-all ${
        item.priority === 'HIGH'
          ? 'border-red-500/40 bg-red-950/10'
          : item.priority === 'LOW'
          ? 'border-amber-500/30 bg-amber-950/10'
          : 'border-slate-800'
      }`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-3">
          <h3 className="font-bold text-slate-100 font-mono text-base">{item.resourceName}</h3>
          <StatusBadge status={item.currentStatus} size="sm" />
        </div>
        <StatusBadge status={item.priority} size="sm" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <span className="text-[11px] font-mono uppercase text-slate-400 font-semibold">Recommended Action:</span>
          <div className="text-sm font-bold font-mono text-sky-400 bg-slate-900/90 p-3 rounded-lg border border-slate-800">
            {item.recommendation}
          </div>
        </div>

        <div className="space-y-2">
          <span className="text-[11px] font-mono uppercase text-slate-400 font-semibold">AI Decision Rationale:</span>
          <p className="text-xs text-slate-300 font-mono leading-relaxed bg-slate-900/90 p-3 rounded-lg border border-slate-800">
            {item.reason}
          </p>
        </div>
      </div>

      {/* Impact Indicators */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono pt-2">
        <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-center">
          <div className="text-[10px] text-slate-400 uppercase">Performance Impact</div>
          <div className={`font-bold mt-0.5 ${item.impact.performance === 'INCREASE' ? 'text-emerald-400' : 'text-sky-400'}`}>
            {item.impact.performance === 'INCREASE' ? 'Performance ↑' : 'Optimized'}
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-center">
          <div className="text-[10px] text-slate-400 uppercase">Availability Impact</div>
          <div className="font-bold mt-0.5 text-emerald-400">
            {item.impact.availability === 'INCREASE' ? 'Availability ↑' : 'Stable'}
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-center">
          <div className="text-[10px] text-slate-400 uppercase">Estimated Cost Change</div>
          <div className={`font-bold mt-0.5 ${item.estimatedCostChange && item.estimatedCostChange > 0 ? 'text-amber-400' : item.estimatedCostChange && item.estimatedCostChange < 0 ? 'text-emerald-400' : 'text-slate-300'}`}>
            {item.estimatedCostChange && item.estimatedCostChange > 0
              ? `+${formatINR(item.estimatedCostChange)}/mo`
              : item.estimatedCostChange && item.estimatedCostChange < 0
              ? `-${formatINR(Math.abs(item.estimatedCostChange))}/mo`
              : '₹0 (Neutral)'}
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-center">
          <div className="text-[10px] text-slate-400 uppercase">Simulated Energy Impact</div>
          <div className={`font-bold mt-0.5 ${item.energyEstimate && item.energyEstimate.baseline_energy_wh > item.energyEstimate.recommended_energy_wh ? 'text-emerald-400' : item.energyEstimate && item.energyEstimate.baseline_energy_wh < item.energyEstimate.recommended_energy_wh ? 'text-amber-400' : 'text-slate-300'}`}>
            {item.energyEstimate
              ? item.energyEstimate.baseline_energy_wh > item.energyEstimate.recommended_energy_wh
                ? `-${(item.energyEstimate.baseline_energy_wh - item.energyEstimate.recommended_energy_wh).toFixed(2)} Wh (${Math.abs(item.energyEstimate.simulated_energy_percentage_change).toFixed(1)}% savings)`
                : item.energyEstimate.baseline_energy_wh < item.energyEstimate.recommended_energy_wh
                ? `+${(item.energyEstimate.recommended_energy_wh - item.energyEstimate.baseline_energy_wh).toFixed(2)} Wh (${item.energyEstimate.simulated_energy_percentage_change.toFixed(1)}% increase)`
                : '0.00 Wh (Neutral)'
              : 'Not available'}
          </div>
        </div>
      </div>

      {/* Safeguard & Policy Provenance Banner */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded-lg bg-slate-900/60 border border-slate-800 text-[11px] font-mono text-slate-400">
        <div className="flex items-center gap-2">
          <span className="text-sky-400 font-bold">{item.policyId || 'Policy B (Proactive ML)'}</span>
          <span>•</span>
          <span className={item.safeguardStatus?.includes('Cooldown') ? 'text-amber-400 font-semibold' : 'text-slate-300'}>
            {item.safeguardStatus || 'Normal'}
          </span>
        </div>
        <div className="flex items-center gap-2 text-slate-500">
          <span>Target Cap: <strong className="text-slate-300">{item.allocatedCapacity ?? 1.0} U</strong></span>
          <span>•</span>
          <span className="text-emerald-500/80 font-mono">Simulated Recommendation</span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-3 pt-2">
        <button
          onClick={() => onNavigateView('explainable')}
          className="flex-1 inline-flex items-center justify-center gap-2 py-2.5 px-4 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded-lg text-xs font-mono transition-all shadow-lg shadow-sky-500/20"
        >
          <Sparkles className="w-4 h-4" />
          <span>Explain AI Decision</span>
        </button>

        <button
          onClick={() => onNavigateView('predictions')}
          className="flex-1 inline-flex items-center justify-center gap-2 py-2.5 px-4 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-mono transition-all"
        >
          <TrendingUp className="w-4 h-4 text-sky-400" />
          <span>View Demand Prediction</span>
        </button>
      </div>
    </div>
  );

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">AI Recommendation Center</h1>
        <p className="text-sm text-slate-400 mt-1">
          Contextual scaling, capacity planning and right-sizing recommendations generated by machine learning.
        </p>
      </div>

      {/* High Priority Group */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-red-400 font-mono font-bold text-sm uppercase tracking-wider">
          <AlertTriangle className="w-4 h-4" />
          <span>High Priority Scaling Actions ({highPriority.length})</span>
        </div>
        <div className="space-y-4">{highPriority.map(renderCard)}</div>
      </div>

      {/* Low Priority Group */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-amber-400 font-mono font-bold text-sm uppercase tracking-wider">
          <span className="font-bold">₹</span>
          <span>Right-Sizing Opportunities ({lowPriority.length})</span>
        </div>
        <div className="space-y-4">{lowPriority.map(renderCard)}</div>
      </div>

      {/* Medium Priority / Optimal Group */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-emerald-400 font-mono font-bold text-sm uppercase tracking-wider">
          <ShieldCheck className="w-4 h-4" />
          <span>Optimal Provisioned Resources ({mediumPriority.length})</span>
        </div>
        <div className="space-y-4">{mediumPriority.map(renderCard)}</div>
      </div>
    </div>
  );
};
