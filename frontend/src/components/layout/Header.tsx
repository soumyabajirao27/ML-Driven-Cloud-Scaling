import React, { useState, useEffect } from 'react';
import { Menu, Clock, Server, Bell, User, LogOut, Database, Wifi } from 'lucide-react';
import { isMockDataMode, setMockDataMode } from '../../services/apiClient';
import { resourceService } from '../../services/resourceService';
import { CloudResource } from '../../types/cloudscaler';

interface HeaderProps {
  setIsOpenMobile: (open: boolean) => void;
  onLogout: () => void;
  userEmail?: string;
}

export const Header: React.FC<HeaderProps> = ({ setIsOpenMobile, onLogout, userEmail }) => {
  const [utcTime, setUtcTime] = useState<string>('');
  const [isDemo, setIsDemo] = useState<boolean>(isMockDataMode());
  const [showNotifications, setShowNotifications] = useState<boolean>(false);
  const [showUserMenu, setShowUserMenu] = useState<boolean>(false);
  const [criticalResources, setCriticalResources] = useState<CloudResource[]>([]);

  useEffect(() => {
    let isMounted = true;
    resourceService.getResources().then((resources) => {
      if (isMounted) {
        const critical = resources.filter(r => r.status === 'OVERLOADED' || r.status === 'UNDERUTILIZED');
        setCriticalResources(critical.slice(0, 3));
      }
    }).catch(() => {});
    return () => { isMounted = false; };
  }, []);

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setUtcTime(now.toISOString().substring(11, 19) + ' UTC');
    };
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleDemo = () => {
    const next = !isDemo;
    setIsDemo(next);
    setMockDataMode(next);
  };

  return (
    <header className="h-16 bg-[#0b1120]/90 backdrop-blur border-b border-slate-800/80 sticky top-0 z-30 px-4 lg:px-8 flex items-center justify-between">
      {/* Left items: Mobile toggle + Breadcrumb/Status */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => setIsOpenMobile(true)}
          className="lg:hidden text-slate-400 hover:text-white p-2 rounded-lg bg-slate-900 border border-slate-800"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* System telemetry badges */}
        <div className="hidden sm:flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
            <Clock className="w-3.5 h-3.5 text-sky-400" />
            <span>{utcTime || '00:00:00 UTC'}</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
            <Server className="w-3.5 h-3.5 text-sky-400" />
            <span>3 Instances</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-xs font-mono text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>System Health: Healthy</span>
          </div>
        </div>
      </div>

      {/* Right items: Mode Switcher, Notifications, Profile */}
      <div className="flex items-center gap-3">
        {/* Backend API vs Demo Switcher */}
        <button
          onClick={handleToggleDemo}
          title={isDemo ? 'Using Centralized Demo Data Layer' : 'Connected to Live FastAPI Backend (http://localhost:8000)'}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
            isDemo
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-300 hover:bg-amber-500/20'
              : 'bg-sky-500/10 border-sky-500/30 text-sky-300 hover:bg-sky-500/20'
          }`}
        >
          {isDemo ? <Database className="w-3.5 h-3.5 text-amber-400" /> : <Wifi className="w-3.5 h-3.5 text-sky-400" />}
          <span className="hidden md:inline">{isDemo ? 'Demo Mode' : 'Live API (localhost:8000)'}</span>
        </button>

        {/* Notifications Button */}
        <div className="relative">
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            className="p-2 text-slate-400 hover:text-white rounded-lg bg-slate-900 border border-slate-800 relative"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-red-500"></span>
          </button>

          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 bg-[#0f172a] border border-slate-800 rounded-xl shadow-2xl z-50 p-4 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-semibold text-slate-200 uppercase font-mono">System Alerts</span>
                <span className="text-[10px] bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded font-mono">
                  {criticalResources.length > 0 ? `${criticalResources.length} Attention` : 'All Optimal'}
                </span>
              </div>
              <div className="space-y-2 text-xs">
                {criticalResources.length > 0 ? (
                  criticalResources.map((res) => (
                    <div
                      key={res.id}
                      className={`p-2.5 rounded-lg border ${
                        res.status === 'OVERLOADED'
                          ? 'bg-red-950/30 border-red-900/40 text-red-200'
                          : 'bg-amber-950/20 border-amber-900/40 text-amber-200'
                      }`}
                    >
                      <div className={`font-semibold ${res.status === 'OVERLOADED' ? 'text-red-400' : 'text-amber-400'}`}>
                        {res.name || res.id} {res.status === 'OVERLOADED' ? 'Overloaded' : 'Underutilized'}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        CPU {res.cpu}%, Memory {res.memory}%. Provider: {res.provider}.
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="p-2.5 rounded-lg bg-emerald-950/20 border border-emerald-900/40 text-emerald-300 text-[11px]">
                    All monitored workloads operating within safe operational margins.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* User Profile */}
        <div className="relative">
          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-medium text-slate-300 hover:text-white"
          >
            <div className="w-6 h-6 rounded-full bg-sky-500/20 border border-sky-500/40 flex items-center justify-center text-sky-400">
              <User className="w-3.5 h-3.5" />
            </div>
            <span className="hidden sm:inline font-mono">{userEmail || 'admin@cloudscaler.io'}</span>
          </button>

          {showUserMenu && (
            <div className="absolute right-0 mt-2 w-48 bg-[#0f172a] border border-slate-800 rounded-xl shadow-2xl z-50 p-2 space-y-1">
              <div className="px-3 py-2 border-b border-slate-800 text-xs">
                <div className="font-medium text-slate-200">Cloud Operations Lead</div>
                <div className="text-[10px] text-slate-400 truncate">{userEmail || 'admin@cloudscaler.io'}</div>
              </div>
              <button
                onClick={onLogout}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 rounded-lg text-left"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span>Sign Out</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
