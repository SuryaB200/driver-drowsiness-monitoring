import React, { useEffect, useState } from "react";
import {
  Truck,
  AlertTriangle,
  AlertOctagon,
  Gauge,
  Video,
  Cpu,
  ShieldCheck,
  ArrowRight,
  Clock,
} from "lucide-react";
import { fetchEvents, fetchModelStatus } from "../services/api";
import { SafetyEventRecord, MonitoringPayload } from "../types";

interface OverviewPageProps {
  telemetry: MonitoringPayload | null;
  onNavigateToMonitoring: () => void;
  isBackendConnected: boolean;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  telemetry,
  onNavigateToMonitoring,
  isBackendConnected,
}) => {
  const [recentEvents, setRecentEvents] = useState<SafetyEventRecord[]>([]);
  const [kpis, setKpis] = useState({
    total_today: 0,
    critical_today: 0,
    warning_today: 0,
    drowsy_events: 0,
    distraction_events: 0,
  });
  const [modelStatus, setModelStatus] = useState<any>(null);

  useEffect(() => {
    fetchEvents("ALL", "ALL", "", 8)
      .then((res) => {
        if (res.events) setRecentEvents(res.events);
        if (res.kpi_summary) setKpis(res.kpi_summary);
      })
      .catch(() => {});

    fetchModelStatus()
      .then((res) => setModelStatus(res))
      .catch(() => {});
  }, [telemetry]);

  const activeWarningsCount =
    (telemetry?.assessment.severity === "WARNING" ? 1 : 0) +
    (telemetry?.assessment.severity === "CRITICAL" ? 1 : 0);

  return (
    <div className="p-6 max-w-[1700px] mx-auto space-y-6">
      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Vehicles Monitored */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm">
          <div className="flex items-center justify-between text-fleet-muted text-xs font-semibold uppercase">
            <span>Local Monitoring Node</span>
            <Truck className="w-4 h-4 text-fleet-primary" />
          </div>
          <div className="text-2xl font-bold text-fleet-text mt-2 font-mono">
            1 Device
          </div>
          <div className="text-xs text-fleet-muted mt-1">
            LOCAL-TRUCK-01 (Single-Node Active)
          </div>
        </div>

        {/* Active Warnings */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm">
          <div className="flex items-center justify-between text-fleet-muted text-xs font-semibold uppercase">
            <span>Active Alerts</span>
            <AlertTriangle className="w-4 h-4 text-fleet-warning" />
          </div>
          <div className="text-2xl font-bold text-fleet-text mt-2 font-mono">
            {activeWarningsCount}
          </div>
          <div className="text-xs text-fleet-muted mt-1">
            {telemetry?.assessment.severity !== "NORMAL"
              ? `Current: ${telemetry?.assessment.primary_state}`
              : "No current alerts"}
          </div>
        </div>

        {/* Critical Events Today */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm">
          <div className="flex items-center justify-between text-fleet-muted text-xs font-semibold uppercase">
            <span>Critical Events Today</span>
            <AlertOctagon className="w-4 h-4 text-fleet-critical" />
          </div>
          <div className="text-2xl font-bold text-fleet-text mt-2 font-mono">
            {kpis.critical_today}
          </div>
          <div className="text-xs text-fleet-muted mt-1">
            {kpis.drowsy_events} fatigue / prolonged closures
          </div>
        </div>

        {/* Pipeline Latency */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm">
          <div className="flex items-center justify-between text-fleet-muted text-xs font-semibold uppercase">
            <span>Inference Latency</span>
            <Gauge className="w-4 h-4 text-fleet-primary" />
          </div>
          <div className="text-2xl font-bold text-fleet-text mt-2 font-mono">
            {telemetry?.measurements.inference_latency_ms
              ? `${telemetry.measurements.inference_latency_ms} ms`
              : "10.6 ms"}
          </div>
          <div className="text-xs text-fleet-muted mt-1">
            MobileNetV3 Edge Inference
          </div>
        </div>
      </div>

      {/* System Readiness Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Camera Node */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm flex items-start gap-3">
          <div className="p-2 rounded bg-fleet-bg border border-fleet-border text-fleet-primary shrink-0">
            <Video className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[13px] font-bold text-fleet-text">
              Vision Capture Engine
            </div>
            <div className="text-xs text-fleet-muted mt-0.5">
              {telemetry?.camera_available ? "Hardware Stream: Active" : "Hardware Stream: Standby"}
            </div>
            <div className="text-[11px] font-mono text-fleet-muted mt-1">
              Supports Integrated & External USB Webcams
            </div>
          </div>
        </div>

        {/* Deep Learning Core */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm flex items-start gap-3">
          <div className="p-2 rounded bg-fleet-bg border border-fleet-border text-fleet-primary shrink-0">
            <Cpu className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[13px] font-bold text-fleet-text">
              MobileNetV3 Core
            </div>
            <div className="text-xs text-fleet-muted mt-0.5">
              {modelStatus?.is_loaded ? "Checkpoint Loaded (Ready)" : "Checkpoint Missing"}
            </div>
            <div className="text-[11px] font-mono text-fleet-muted mt-1 truncate">
              {modelStatus?.device ? `Target Device: ${modelStatus.device}` : "Edge CPU/CUDA"}
            </div>
          </div>
        </div>

        {/* Safety & Compliance Engine */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm flex items-start gap-3">
          <div className="p-2 rounded bg-fleet-bg border border-fleet-border text-fleet-normal shrink-0">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[13px] font-bold text-fleet-text">
              Driver Safety Engine
            </div>
            <div className="text-xs text-fleet-muted mt-0.5">
              EAR, MAR & Head Pose Tracking
            </div>
            <div className="text-[11px] font-mono text-fleet-muted mt-1">
              In-Cab Voice Warning Active
            </div>
          </div>
        </div>
      </div>

      {/* Recent Safety Events Table */}
      <div className="bg-white border border-fleet-border rounded-lg shadow-sm">
        <div className="p-4 border-b border-fleet-border flex items-center justify-between">
          <div>
            <h2 className="text-[15px] font-bold text-fleet-text m-0">
              Recent Safety Telematics Events
            </h2>
            <p className="text-xs text-fleet-muted mt-0.5">
              Live audit log of drowsiness, distraction, and eye closure warnings
            </p>
          </div>
          <button
            onClick={onNavigateToMonitoring}
            className="flex items-center gap-1.5 text-xs font-semibold text-fleet-primary hover:text-fleet-primaryHover transition-colors"
          >
            <span>Launch Live Monitoring</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-[13px]">
            <thead>
              <tr className="bg-fleet-bg border-b border-fleet-border text-[11px] font-bold uppercase text-fleet-muted tracking-wider">
                <th className="py-2.5 px-4">Event ID</th>
                <th className="py-2.5 px-4">Timestamp</th>
                <th className="py-2.5 px-4">Vehicle</th>
                <th className="py-2.5 px-4">Event Type</th>
                <th className="py-2.5 px-4">Severity</th>
                <th className="py-2.5 px-4">Duration</th>
                <th className="py-2.5 px-4">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-fleet-border">
              {recentEvents.length > 0 ? (
                recentEvents.map((evt) => (
                  <tr key={evt.id} className="hover:bg-[#FAFBFC] transition-colors">
                    <td className="py-2.5 px-4 font-mono text-xs text-fleet-muted">
                      {evt.id}
                    </td>
                    <td className="py-2.5 px-4 text-fleet-text whitespace-nowrap">
                      {evt.timestamp}
                    </td>
                    <td className="py-2.5 px-4 font-semibold text-fleet-text">
                      {evt.vehicle_id}
                    </td>
                    <td className="py-2.5 px-4 font-mono font-bold text-xs">
                      {evt.event_type}
                    </td>
                    <td className="py-2.5 px-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase ${
                          evt.severity === "CRITICAL"
                            ? "bg-[#FDF2F2] text-fleet-critical border border-[#F8D2D2]"
                            : evt.severity === "WARNING"
                            ? "bg-[#FFF9F0] text-fleet-warning border border-[#FCE6C7]"
                            : "bg-[#F0F8F3] text-fleet-normal border border-[#D1E8D8]"
                        }`}
                      >
                        {evt.severity}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 font-mono text-fleet-muted">
                      {evt.duration_sec.toFixed(1)}s
                    </td>
                    <td className="py-2.5 px-4 text-fleet-muted max-w-md truncate">
                      {evt.description}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="py-6 text-center text-fleet-muted">
                    No safety events logged yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
