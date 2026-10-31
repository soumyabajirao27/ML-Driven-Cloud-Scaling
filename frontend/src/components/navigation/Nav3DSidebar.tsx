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
  X,
  Compass,
  Layers
} from 'lucide-react';
import { NavItem } from '../layout/Sidebar';
import { predictionService } from '../../services/predictionService';
import { ModelInfo } from '../../types/cloudscaler';

interface Nav3DSidebarProps {
  activeNav: NavItem;
  setActiveNav: (nav: NavItem) => void;
  isOpenMobile: boolean;
  setIsOpenMobile: (open: boolean) => void;
}

export const Nav3DSidebar: React.FC<Nav3DSidebarProps> = ({
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

  const mainNavItems: { id: NavItem; label: string; icon: React.ElementType; badge?: string }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'resources', label: 'Resources', icon: Server, badge: '3' },
    { id: 'predictions', label: 'Predictions', icon: TrendingUp, badge: 'ML' },
    { id: 'cost', label: 'Cost Optimization', icon: DollarSign, badge: '₹' },
    { id: 'recommendations', label: 'Scaling Recommendations', icon: Zap, badge: 'AI' },
    { id: 'comparison', label: 'Comparison', icon: Columns },
    { id: 'monitoring', label: 'System Monitoring', icon: Activity },
    { id: 'analytics', label: 'Model Analytics', icon: Brain },
    { id: 'explainable', label: 'Explainable AI', icon: HelpCircle, badge: 'XAI' },
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

      {/* 3D Navigation Sidebar Drawer */}
      <aside
        className={`fixed top-0 left-0 bottom-0 w-64 bg-gradient-to-b from-[#0b1120] via-[#090d16] to-[#070b14] border-r border-slate-800/80 z-50 flex flex-col justify-between transition-transform duration-300 ease-in-out lg:translate-x-0 ${
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
                <h1 className="font-bold text-slate-100 tracking-tight text-base leading-tight font-mono">
                  CloudScaler
                </h1>
                <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-sky-400 font-semibold">
                  <Compass className="w-3 h-3 text-sky-400 animate-spin" style={{ animationDuration: '8s' }} />
                  <span>3D Spatial Nav</span>
                </div>
              </div>
            </div>
            <button
              onClick={() => setIsOpenMobile(false)}
              className="lg:hidden text-slate-400 hover:text-white p-1"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* 3D Spatial Navigation Links */}
          <nav className="p-3 space-y-1.5 overflow-y-auto max-h-[calc(100vh-280px)] nav-3d-container">
            <div className="px-3 pt-2 pb-1 text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-500 flex items-center justify-between">
              <span>Operations Center</span>
              <span className="text-[10px] text-sky-400/80 bg-sky-500/10 px-1.5 py-0.5 rounded font-mono">3D Spatial</span>
            </div>

            {mainNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeNav === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-mono transition-all border ${
                    isActive
                      ? 'nav-3d-item-active text-sky-300 font-bold border-sky-500/50'
                      : 'nav-3d-item text-slate-400 hover:text-slate-100 bg-slate-900/60 border-slate-800/80'
                  }`}
                >
                  <div className="flex items-center gap-3 truncate">
                    <div className={`p-1.5 rounded-lg ${isActive ? 'bg-sky-500/20 text-sky-400' : 'bg-slate-800 text-slate-400'}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="truncate">{item.label}</span>
                  </div>

                  {item.badge && (
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded-md font-bold ${
                        isActive
                          ? 'bg-sky-500 text-slate-950 shadow-sm'
                          : 'bg-slate-800 text-slate-400 border border-slate-700'
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Model Card & Auxiliary Nav */}
        <div className="p-3 space-y-3 border-t border-slate-800/80 bg-[#070b14]/50">
          <div className="space-y-1 nav-3d-container">
            <button
              onClick={() => handleNavClick('settings')}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-mono transition-all border ${
                activeNav === 'settings'
                  ? 'nav-3d-item-active text-sky-300 font-bold border-sky-500/50'
                  : 'nav-3d-item text-slate-400 hover:text-slate-100 bg-slate-900/60 border-slate-800/80'
              }`}
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </button>

            <button
              onClick={() => handleNavClick('about')}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-mono transition-all border ${
                activeNav === 'about'
                  ? 'nav-3d-item-active text-sky-300 font-bold border-sky-500/50'
                  : 'nav-3d-item text-slate-400 hover:text-slate-100 bg-slate-900/60 border-slate-800/80'
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
