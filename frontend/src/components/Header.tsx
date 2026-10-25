import React, { useState, useEffect } from "react";
import { Activity, Cpu, CheckCircle2, AlertTriangle, Clock } from "lucide-react";

interface HeaderProps {
  pageTitle: string;
  isBackendConnected: boolean;
  isModelReady: boolean;
  cameraFps?: number;
}

export const Header: React.FC<HeaderProps> = ({
  pageTitle,
  isBackendConnected,
  isModelReady,
  cameraFps = 0,
}) => {
  const [currentTime, setCurrentTime] = useState<string>("");

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setCurrentTime(
        now.toLocaleString("en-US", {
          weekday: "short",
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        })
      );
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-16 bg-white border-b border-fleet-border px-6 flex items-center justify-between shrink-0 select-none">
      {/* Title */}
      <div>
        <h1 className="text-[19px] font-bold text-fleet-text tracking-tight m-0 p-0">
          {pageTitle}
        </h1>
        <div className="text-[11px] text-fleet-muted">
          Commercial Fleet Telematics & Driver Assistance System
        </div>
      </div>

      {/* System Status Indicators */}
      <div className="flex items-center gap-3">
        {/* Backend Connectivity Status */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[12px] font-medium border ${
            isBackendConnected
              ? "bg-[#F0F8F3] border-[#D1E8D8] text-fleet-normal"
              : "bg-[#FDF2F2] border-[#F8D2D2] text-fleet-critical"
          }`}
        >
          <Activity className="w-3.5 h-3.5" />
          <span>{isBackendConnected ? "API Connected" : "API Disconnected"}</span>
        </div>

        {/* Model Readiness Status */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[12px] font-medium border ${
            isModelReady
              ? "bg-[#F0F4FA] border-[#D3E0F3] text-fleet-primary"
              : "bg-[#FFF9F0] border-[#FCE6C7] text-fleet-warning"
          }`}
        >
          <Cpu className="w-3.5 h-3.5" />
          <span>{isModelReady ? "MobileNetV3 Loaded" : "Model Checkpoint Missing"}</span>
        </div>

        {/* Live FPS */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[12px] font-medium bg-[#F4F5F7] border border-fleet-border text-fleet-text">
          <span className="text-[11px] text-fleet-muted">Capture:</span>
          <span className="font-semibold">{cameraFps > 0 ? `${cameraFps} FPS` : "Standby"}</span>
        </div>

        {/* Date / Time */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[12px] font-medium text-fleet-muted border border-fleet-border bg-white">
          <Clock className="w-3.5 h-3.5" />
          <span>{currentTime || "Loading..."}</span>
        </div>
      </div>
    </header>
  );
};
