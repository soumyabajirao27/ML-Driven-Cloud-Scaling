import React, { useState, useEffect } from 'react';
import { CloudResource, ExplainableAIDecision } from '../types/cloudscaler';
import { resourceService } from '../services/resourceService';
import { predictionService } from '../services/predictionService';
import { StatusBadge } from '../components/common/StatusBadge';
import { SkeletonLoader } from '../components/common/SkeletonLoader';
import { HelpCircle, Brain, ArrowRight, ShieldAlert, Sparkles, CheckCircle2 } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from 'recharts';

export const ExplainableAIView: React.FC = () => {
  const [resources, setResources] = useState<CloudResource[]>([]);
  const [selectedResourceId, setSelectedResourceId] = useState<string>('');
  const [decision, setDecision] = useState<ExplainableAIDecision | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const init = async () => {
      try {
        const resList = await resourceService.getResources();
        setResources(resList);
        if (resList.length > 0) {
          const targetId = selectedResourceId && resList.some(r => r.id === selectedResourceId)
            ? selectedResourceId
            : resList[0].id;
          if (targetId !== selectedResourceId) {
            setSelectedResourceId(targetId);
          }
          const res = await predictionService.getExplainableAI(targetId);
          setDecision(res);
        }
      } finally {
        setLoading(false);
      }
    };
    init();
  }, [selectedResourceId]);

  if (loading || !decision) return <SkeletonLoader type="card" count={3} />;

  const featureChartData = decision.primaryDrivers.map((d) => ({
    name: d.feature,
    ImpactWeight: d.weight,
    impact: d.impact
  }));

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">Explainable AI (XAI)</h1>
          <p className="text-sm text-slate-400 mt-1">
            Auditable feature attribution and step-by-step decision reasoning paths.
          </p>
        </div>

        {/* Resource selector */}
        <div className="flex items-center gap-2 bg-[#0f172a] p-1.5 rounded-xl border border-slate-800 font-mono text-xs max-w-full overflow-x-auto">
          {resources.slice(0, 8).map((r) => (
            <button
              key={r.id}
              onClick={() => setSelectedResourceId(r.id)}
              className={`px-3 py-1.5 rounded-lg transition-all whitespace-nowrap ${
                selectedResourceId === r.id
                  ? 'bg-sky-500 text-slate-950 font-bold shadow-md shadow-sky-500/20'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              {r.name || r.id}
            </button>
          ))}
          {resources.length > 8 && (
            <select
              value={selectedResourceId}
              onChange={(e) => setSelectedResourceId(e.target.value)}
              className="bg-slate-900 border border-slate-800 text-slate-300 rounded px-2 py-1 text-xs focus:outline-none"
            >
              {resources.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name || r.id} ({r.provider})
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Primary Question Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-sky-950/40 via-slate-900 to-indigo-950/30 border border-sky-500/30 space-y-4">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-sky-500/20 text-sky-400 border border-sky-500/40">
              <HelpCircle className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs font-mono text-sky-400 uppercase font-semibold">Core XAI Query</div>
              <h2 className="text-lg font-bold text-slate-100 font-mono mt-0.5">
                Why did CloudScaler recommend "{decision.recommendation}" for {decision.resourceName}?
              </h2>
            </div>
          </div>
          <StatusBadge status={decision.classifiedStatus} size="lg" />
        </div>
      </div>

      {/* Feature Contribution Breakdown & Reasoning Steps Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Contribution Visualization */}
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div>
            <h3 className="text-base font-bold text-slate-200 font-mono">Feature Contribution Weights</h3>
            <p className="text-xs text-slate-400">Relative impact of metrics driving the classification decision.</p>
          </div>

          <div className="h-64 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureChartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#94a3b8" unit="%" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis type="category" dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} width={120} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="ImpactWeight" fill="#38bdf8" radius={[0, 4, 4, 0]}>
                  {featureChartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.impact === 'High Risk' ? '#ef4444' : entry.impact === 'Under-capacity' ? '#f59e0b' : '#38bdf8'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-2 pt-2 border-t border-slate-800 font-mono text-xs">
            {decision.primaryDrivers.map((driver, i) => (
              <div key={i} className="flex justify-between items-center p-2 rounded-lg bg-slate-900/80 border border-slate-800">
                <span className="text-slate-300">{driver.feature}</span>
                <span className={`font-bold ${driver.impact === 'High Risk' ? 'text-red-400' : driver.impact === 'Under-capacity' ? 'text-amber-400' : 'text-sky-400'}`}>
                  {driver.value} ({driver.weight}% Weight)
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* 4-Step Decision Flowchart */}
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div>
            <h3 className="text-base font-bold text-slate-200 font-mono">4-Step Decision Workflow</h3>
            <p className="text-xs text-slate-400">Sequential reasoning path executed by the inference engine.</p>
          </div>

          <div className="space-y-4 pt-2">
            {decision.reasoningPath.map((step) => (
              <div key={step.step} className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-1 relative">
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="text-sky-400 font-bold">Step 0{step.step}: {step.stage}</span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-bold">{step.value}</span>
                </div>
                <div className="text-xs font-bold text-slate-200 font-mono mt-1">{step.label}</div>
                <p className="text-xs text-slate-400 font-mono">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
