import React from "react";
import {
  LayoutDashboard,
  Video,
  Truck,
  Cpu,
  History,
  Sliders,
  ShieldCheck,
} from "lucide-react";

export type NavTab =
  | "overview"
  | "monitoring"
  | "vehicles"
  | "model"
  | "events"
  | "settings";

interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  isBackendConnected: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  isBackendConnected,
}) => {
  const navItems = [
    { id: "overview" as NavTab, label: "Overview", icon: LayoutDashboard },
    { id: "monitoring" as NavTab, label: "Live Monitoring", icon: Video },
    { id: "vehicles" as NavTab, label: "Fleet Vehicles", icon: Truck },
    { id: "model" as NavTab, label: "Model Evaluation", icon: Cpu },
    { id: "events" as NavTab, label: "Event History", icon: History },
    { id: "settings" as NavTab, label: "Settings", icon: Sliders },
  ];

  return (
    <aside className="w-[220px] bg-white border-r border-fleet-border flex flex-col h-screen select-none shrink-0">
      {/* Brand Header */}
      <div className="h-16 px-4 flex items-center gap-2.5 border-b border-fleet-border">
        <div className="w-8 h-8 rounded bg-fleet-primary flex items-center justify-center text-white font-semibold">
          <ShieldCheck className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="text-sm font-bold tracking-tight text-fleet-text leading-tight">
            FLEETGUARD DMS
          </div>
          <div className="text-[11px] text-fleet-muted font-medium">
            Driver Safety AI
          </div>
        </div>
      </div>

      {/* Nav List */}
      <nav className="p-3 space-y-1 flex-1 overflow-y-auto">
        <div className="text-[10px] uppercase font-semibold text-fleet-muted px-2.5 py-1 tracking-wider">
          Operations
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`w-full flex items-center gap-2.5 px-3 py-2 text-[13px] font-medium rounded-md transition-colors text-left ${
                isActive
                  ? "bg-fleet-primary text-white shadow-sm"
                  : "text-fleet-text hover:bg-fleet-bg hover:text-fleet-primary"
              }`}
            >
              <Icon className={`w-4 h-4 shrink-0 ${isActive ? "text-white" : "text-fleet-muted"}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Operational Node Footer */}
      <div className="p-3 border-t border-fleet-border bg-[#FAFBFC]">
        <div className="text-[11px] font-semibold text-fleet-text mb-1">
          Local Edge Terminal
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-fleet-muted">
          <span
            className={`w-2 h-2 rounded-full ${
              isBackendConnected ? "bg-fleet-normal" : "bg-fleet-critical animate-pulse"
            }`}
          />
          <span>{isBackendConnected ? "Core Daemon: Online" : "Daemon Offline"}</span>
        </div>
        <div className="text-[10px] text-fleet-muted mt-1">
          v1.0.0-academic • MobileNetV3
        </div>
      </div>
    </aside>
  );
};
