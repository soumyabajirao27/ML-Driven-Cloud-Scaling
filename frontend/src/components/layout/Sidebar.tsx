import React, { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  Server,
  TrendingUp,
  DollarSign,
  Zap,
  Columns,
  Activity,
  Brain,
  HelpCircle,
  Settings,
  Info,
  Cloud,
  CheckCircle2,
  X
} from 'lucide-react';
import { predictionService } from '../../services/predictionService';
import { ModelInfo } from '../../types/cloudscaler';

export type NavItem =
  | 'dashboard'
  | 'resources'
  | 'predictions'
  | 'cost'
  | 'recommendations'
  | 'comparison'
  | 'monitoring'
  | 'analytics'
  | 'explainable'
  | 'settings'
  | 'about';

interface SidebarProps {
  activeNav: NavItem;
  setActiveNav: (nav: NavItem) => void;
  isOpenMobile: boolean;
  setIsOpenMobile: (open: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeNav,
  setActiveNav,
  isOpenMobile,
  setIsOpenMobile
}) => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    let isMounted = true;
    predictionService.getModelInfo().then((info) => {
      if (isMounted) setModelInfo(info);
    }).catch(() => {});
    return () => { isMounted = false; };
  }, []);

  const mainNavItems: { id: NavItem; label: string; icon: React.ElementType }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'resources', label: 'Resources', icon: Server },
    { id: 'predictions', label: 'Predictions', icon: TrendingUp },
    { id: 'cost', label: 'Cost Optimization', icon: DollarSign },
    { id: 'recommendations', label: 'Scaling Recommendations', icon: Zap },
    { id: 'comparison', label: 'Comparison', icon: Columns },
    { id: 'monitoring', label: 'System Monitoring', icon: Activity },
    { id: 'analytics', label: 'Model Analytics', icon: Brain },
    { id: 'explainable', label: 'Explainable AI', icon: HelpCircle },
  ];

  const handleNavClick = (id: NavItem) => {
    setActiveNav(id);
    setIsOpenMobile(false);
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpenMobile && (
        <div
          className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setIsOpenMobile(false)}
        />
      )}

      {/* Sidebar Drawer */}
      <aside
        className={`fixed top-0 left-0 bottom-0 w-64 bg-[#0b1120] border-r border-slate-800/80 z-50 flex flex-col justify-between transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpenMobile ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Top Branding Section */}
        <div>
          <div className="h-16 px-5 border-b border-slate-800/80 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-sky-500/20">
                <Cloud className="w-5 h-5" />
              </div>
              <div>
                <h1 className="font-bold text-slate-100 tracking-tight text-base leading-tight">
                  CloudScaler
                </h1>
                <span className="text-[10px] font-mono uppercase tracking-wider text-sky-400 font-semibold">
                  AI Cloud Ops
                </span>
              </div>
            </div>
            <button
              onClick={() => setIsOpenMobile(false)}
              className="lg:hidden text-slate-400 hover:text-white p-1"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1 overflow-y-auto max-h-[calc(100vh-280px)]">
            <div className="px-3 pt-2 pb-1 text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-500">
              Operations Center
            </div>
            {mainNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeNav === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30 font-semibold shadow-sm shadow-sky-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : 'text-slate-400'}`} />
                  <span className="truncate">{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Model Card & Auxiliary Nav */}
        <div className="p-3 space-y-3 border-t border-slate-800/80 bg-[#070b14]/50">
          <div className="space-y-1">
            <button
              onClick={() => handleNavClick('settings')}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                activeNav === 'settings'
                  ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
              }`}
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </button>

            <button
              onClick={() => handleNavClick('about')}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                activeNav === 'about'
                  ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
              }`}
            >
              <Info className="w-4 h-4" />
              <span>About CloudScaler</span>
            </button>
          </div>

          {/* Model Status Card */}
          <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800/80 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-medium">Model Status</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 font-semibold">
                {modelInfo ? 'RandomForest' : 'AI Engine'}
              </span>
            </div>
            <div className="text-xs font-semibold text-slate-200 font-mono">
              {modelInfo ? `${modelInfo.model_type} (${modelInfo.model_id})` : 'Model Status Checking...'}
            </div>
            <div className="flex items-center gap-2 text-[11px] text-emerald-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="font-mono text-[11px]">{modelInfo?.status || 'Active & Ready'}</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
