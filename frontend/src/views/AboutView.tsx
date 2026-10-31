import React from 'react';
import { Cloud, Brain, Sparkles, Award, ShieldAlert, Database, Cpu, Layers, DollarSign, Activity } from 'lucide-react';

export const AboutView: React.FC = () => {
  return (
    <div className="space-y-8 animate-fadeIn max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">About CloudScale AI</h1>
        <p className="text-sm text-slate-400 mt-1">
          Machine Learning-Driven Auto Scaling and Cost Optimization Framework for Cloud Resources • University PBL Project
        </p>
      </div>

      <div className="p-8 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center text-white shadow-xl shadow-sky-500/20">
            <Cloud className="w-7 h-7" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-100 font-mono">CloudScale AI Research Prototype</h2>
            <p className="text-xs text-sky-400 font-mono">Evaluation-Ready PBL Framework • Rigorous ML Benchmark & FinOps Optimization</p>
          </div>
        </div>

        <p className="text-xs text-slate-300 font-mono leading-relaxed">
          CloudScale AI is a machine learning-driven cloud optimization research prototype. It couples chronological Google Borg cluster traces with multi-variate machine learning models to forecast future compute demand, prevent reactive threshold lag, and provide auditable, explainable autoscaling recommendations while estimating simulated FinOps and energy impacts.
        </p>

        {/* Technical Architecture Matrix */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs pt-4 border-t border-slate-800">
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <div className="text-amber-400 font-bold flex items-center gap-1.5">
              <Database className="w-4 h-4" /> Google Borg Telemetry
            </div>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Real historical trace dataset (65,398 pairs) ingested into PostgreSQL with sequential chunk replay, deterministic ordering, and strict uniqueness guarantees.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <div className="text-sky-400 font-bold flex items-center gap-1.5">
              <Brain className="w-4 h-4" /> ML Inference & Fallback
            </div>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              RandomForestRegressor (9 features) trained to predict next-step CPU demand. If history is limited (&lt;3 samples) or heterogeneous sources collide, it transparently falls back to BaselineRuleEngine.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <div className="text-emerald-400 font-bold flex items-center gap-1.5">
              <Sparkles className="w-4 h-4" /> Policy B Autoscaling
            </div>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Proactive scaling decisions with configurable hysteresis deadbands (18% - 87%), cooldown step dampening, linear server power modeling, and FinOps cost estimation.
            </p>
          </div>
        </div>

        {/* Empirical Findings & Scope Boundaries */}
        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3 font-mono text-xs">
          <div className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-sky-400" />
            <span>Methodological Rigor & Evaluation Transparency</span>
          </div>
          <ul className="space-y-1.5 text-[11px] text-slate-400 list-disc list-inside leading-relaxed">
            <li><strong className="text-slate-300">Offline Benchmark (Phase 4):</strong> Random Forest (R²=0.9071, MAE=0.0006), XGBoost (R²=0.865, MAE=0.0009), LSTM (R²=0.9838, MAE=0.0004). R² represents the coefficient of determination on chronological splits, not raw prediction accuracy.</li>
            <li><strong className="text-slate-300">Simulated Execution:</strong> All scaling actions, cost savings, and energy reductions represent simulated scenarios; physical cloud hardware is not mutated.</li>
            <li><strong className="text-slate-300">Power & Energy Formulation:</strong> Assumes a linear server power model (P(u) = 100 W + 150 W × u) integrated over the evaluation window into Watt-hours (Wh).</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
