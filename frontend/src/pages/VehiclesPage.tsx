import React, { useState } from "react";
import { Truck, Search, ShieldAlert, Filter, CheckCircle2, AlertTriangle, AlertOctagon } from "lucide-react";
import { MonitoringPayload } from "../types";

interface VehiclesPageProps {
  telemetry: MonitoringPayload | null;
}

export const VehiclesPage: React.FC<VehiclesPageProps> = ({ telemetry }) => {
  const [searchTerm, setSearchTerm] = useState("");

  const currentState = telemetry?.assessment.primary_state || "STANDBY";
  const currentSeverity = telemetry?.assessment.severity || "NORMAL";

  const vehicles = [
    {
      id: "LOCAL-TRUCK-01",
      name: "Freightliner Cascadia #101",
      type: "Class 8 Heavy Duty",
      depot: "Central Highway Terminal",
      status: telemetry?.camera_available ? "Active Monitoring" : "Standby",
      driverState: currentState,
      severity: currentSeverity,
      lastSeen: "Just now (Live)",
      isLocal: true,
    },
    {
      id: "DEMO-FLEET-102",
      name: "Volvo VNL 860 #102",
      type: "Long Haul Tractor",
      depot: "Northern Logistics Hub",
      status: "Idle / Parked",
      driverState: "OFF DUTY",
      severity: "NORMAL",
      lastSeen: "45 mins ago",
      isLocal: false,
    },
    {
      id: "DEMO-FLEET-103",
      name: "Kenworth T680 #103",
      type: "Interstate Refrigerated",
      depot: "Western Distribution",
      status: "En Route (Remote)",
      driverState: "ALERT",
      severity: "NORMAL",
      lastSeen: "12 mins ago",
      isLocal: false,
    },
  ];

  const filtered = vehicles.filter(
    (v) =>
      v.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      v.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      v.type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="p-6 max-w-[1700px] mx-auto space-y-5">
      {/* Disclaimer Card as required by Section 14 */}
      <div className="bg-[#FFF9F0] border border-[#FCE6C7] rounded-lg p-3.5 text-[12px] text-fleet-muted flex items-start gap-2.5">
        <ShieldAlert className="w-4 h-4 text-fleet-warning shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-fleet-text">
            Fleet Telematics Scope & Single-Node Academic Prototype Notice:
          </span>{" "}
          This system is currently executing on a local edge device (
          <code>LOCAL-TRUCK-01</code>). Live camera capture, MobileNetV3 inference, and facial keypoint tracking run on the active connected device. Additional entries demonstrate enterprise telematics fleet table schema.
        </div>
      </div>

      {/* Table Toolbar */}
      <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 flex-1 max-w-md">
          <div className="relative w-full">
            <Search className="w-4 h-4 text-fleet-muted absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search by Vehicle ID, model, or type..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full text-xs pl-9 pr-3 py-2 border border-fleet-border rounded bg-fleet-bg focus:outline-none focus:ring-1 focus:ring-fleet-primary"
            />
          </div>
        </div>

        <div className="text-xs font-semibold text-fleet-muted">
          Showing {filtered.length} of {vehicles.length} fleet units
        </div>
      </div>

      {/* Vehicles Table */}
      <div className="bg-white border border-fleet-border rounded-lg shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-[13px]">
            <thead>
              <tr className="bg-fleet-bg border-b border-fleet-border text-[11px] font-bold uppercase text-fleet-muted tracking-wider">
                <th className="py-3 px-4">Vehicle Identifier</th>
                <th className="py-3 px-4">Vehicle Model / Class</th>
                <th className="py-3 px-4">Telemetry Node</th>
                <th className="py-3 px-4">Operational Status</th>
                <th className="py-3 px-4">Driver Assessment</th>
                <th className="py-3 px-4">Last Activity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-fleet-border">
              {filtered.map((v) => (
                <tr key={v.id} className="hover:bg-[#FAFBFC] transition-colors">
                  <td className="py-3 px-4">
                    <div className="font-bold text-fleet-text flex items-center gap-1.5">
                      <Truck className="w-4 h-4 text-fleet-primary" />
                      <span>{v.id}</span>
                    </div>
                    <div className="text-[11px] text-fleet-muted">{v.depot}</div>
                  </td>
                  <td className="py-3 px-4">
                    <div className="font-semibold text-fleet-text">{v.name}</div>
                    <div className="text-[11px] text-fleet-muted">{v.type}</div>
                  </td>
                  <td className="py-3 px-4">
                    {v.isLocal ? (
                      <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-[#F0F4FA] text-fleet-primary border border-[#D3E0F3]">
                        LOCAL EDGE CAMERA
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-fleet-bg text-fleet-muted border border-fleet-border">
                        REMOTE TELEMATICS
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <span className="flex items-center gap-1.5 text-xs font-medium">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          v.status.includes("Active")
                            ? "bg-fleet-normal"
                            : "bg-gray-400"
                        }`}
                      />
                      {v.status}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase ${
                        v.severity === "CRITICAL"
                          ? "bg-[#FDF2F2] text-fleet-critical border border-[#F8D2D2]"
                          : v.severity === "WARNING"
                          ? "bg-[#FFF9F0] text-fleet-warning border border-[#FCE6C7]"
                          : "bg-[#F0F8F3] text-fleet-normal border border-[#D1E8D8]"
                      }`}
                    >
                      {v.driverState}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-xs font-mono text-fleet-muted">
                    {v.lastSeen}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
