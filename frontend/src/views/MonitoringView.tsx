import React, { useState, useEffect } from 'react';
import { DEMO_SYSTEM_EVENTS } from '../services/mockData';
import { StatusBadge } from '../components/common/StatusBadge';
import { Activity, AlertTriangle, ShieldCheck, Clock, Server, CheckCircle2, RefreshCw, Database } from 'lucide-react';
import { resourceService } from '../services/resourceService';
import { CloudResource } from '../types/cloudscaler';

export const MonitoringView: React.FC = () => {
  const [resources, setResources] = useState<CloudResource[]>([]);

  useEffect(() => {
    resourceService.getResources()
      .then((data) => setResources(data))
      .catch(() => {});
  }, []);

  const borgCount = resources.filter((r) => r.provider === 'google_borg').length;
  const mockCount = resources.filter((r) => r.provider !== 'google_borg').length;
  const totalCount = resources.length || 3;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight font-mono">System Observability & Monitoring</h1>
        <p className="text-sm text-slate-400 mt-1">
          Historical telemetry feeds, active alert logs, and system events.
        </p>
      </div>

      {/* Health Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>SYSTEM HEALTH</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400">OPERATIONAL</div>
          <div className="text-[11px] text-slate-500 font-mono">Telemetry Pipeline Online</div>
        </div>

        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>ACTIVE ALERTS</span>
            <AlertTriangle className="w-4 h-4 text-red-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-red-400">
            {resources.filter(r => r.status === 'OVERLOADED').length} Overloaded
          </div>
          <div className="text-[11px] text-slate-400 font-mono">Threshold SLA Warning</div>
        </div>

        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>CONNECTED INSTANCES</span>
            <Server className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-slate-100">{totalCount} Resources</div>
          <div className="text-[11px] text-slate-400 font-mono">
            {borgCount > 0 ? `${borgCount} Borg Trace, ${mockCount} Mock` : 'Mock Cloud Infrastructure'}
          </div>
        </div>

        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>TELEMETRY SOURCE</span>
            <Database className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-lg font-bold font-mono text-slate-200">
            {borgCount > 0 ? 'Google Borg Trace' : 'Mock Deterministic'}
          </div>
          <div className="text-[11px] text-amber-400 font-mono">
            {borgCount > 0 ? 'Historical Batch Ingested' : 'Local Sandbox Mode'}
          </div>
        </div>
      </div>

      {/* Activity Log & Telemetry Event Stream */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-base font-bold text-slate-200 font-mono">Activity Timeline & System Events</h2>
            <p className="text-xs text-slate-400 mt-0.5">Audit log of analysis passes, ML predictions, and scaling recommendations.</p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 text-slate-400 border border-slate-800">
            Authoritative Event Audit
          </span>
        </div>

        <div className="relative pl-6 space-y-6 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
          {DEMO_SYSTEM_EVENTS.map((evt) => (
            <div key={evt.id} className="relative flex items-start gap-4 group">
              {/* Event node */}
              <div className={`absolute -left-6 top-1.5 w-3 h-3 rounded-full border-2 border-[#0f172a] ${
                evt.severity === 'CRITICAL' ? 'bg-red-500' : evt.severity === 'WARNING' ? 'bg-amber-500' : evt.severity === 'SUCCESS' ? 'bg-emerald-500' : 'bg-sky-500'
              }`} />

              <div className="flex-1 p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-1 hover:border-slate-700 transition-all">
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="font-bold text-slate-100">{evt.title}</span>
                  <span className="text-slate-500 text-[11px]">{evt.timestamp}</span>
                </div>
                <p className="text-xs text-slate-300 font-mono leading-relaxed">{evt.description}</p>
                {evt.resourceId && (
                  <div className="pt-2 flex items-center gap-2">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-sky-400 border border-slate-700">
                      Target: {evt.resourceId}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
