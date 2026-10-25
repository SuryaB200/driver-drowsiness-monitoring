import React, { useState, useEffect } from "react";
import { History, Download, Search, Filter, RefreshCw } from "lucide-react";
import { fetchEvents, getExportCsvUrl } from "../services/api";
import { SafetyEventRecord } from "../types";

export const EventHistoryPage: React.FC = () => {
  const [events, setEvents] = useState<SafetyEventRecord[]>([]);
  const [severityFilter, setSeverityFilter] = useState("ALL");
  const [eventTypeFilter, setEventTypeFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const loadEvents = () => {
    setIsLoading(true);
    fetchEvents(severityFilter, eventTypeFilter, search, 200)
      .then((res) => {
        if (res.events) setEvents(res.events);
      })
      .finally(() => setIsLoading(false));
  };

  useEffect(() => {
    loadEvents();
  }, [severityFilter, eventTypeFilter]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    loadEvents();
  };

  return (
    <div className="p-6 max-w-[1700px] mx-auto space-y-5">
      {/* Filters & Actions Bar */}
      <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm flex flex-wrap items-center justify-between gap-3">
        <form onSubmit={handleSearchSubmit} className="flex items-center gap-2 flex-1 max-w-md">
          <div className="relative w-full">
            <Search className="w-4 h-4 text-fleet-muted absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search description, event ID, vehicle..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full text-xs pl-9 pr-3 py-2 border border-fleet-border rounded bg-fleet-bg focus:outline-none focus:ring-1 focus:ring-fleet-primary"
            />
          </div>
          <button
            type="submit"
            className="px-3 py-2 bg-fleet-bg border border-fleet-border text-fleet-text rounded text-xs font-semibold hover:bg-fleet-border transition-colors"
          >
            Search
          </button>
        </form>

        <div className="flex items-center gap-3 flex-wrap">
          {/* Severity Filter */}
          <div className="flex items-center gap-1.5 text-xs text-fleet-muted">
            <Filter className="w-3.5 h-3.5" />
            <span>Severity:</span>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="border border-fleet-border rounded px-2.5 py-1.5 bg-white text-fleet-text text-xs font-medium focus:ring-1 focus:ring-fleet-primary"
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical</option>
              <option value="WARNING">Warning</option>
              <option value="NORMAL">Normal</option>
            </select>
          </div>

          {/* Event Type Filter */}
          <div className="flex items-center gap-1.5 text-xs text-fleet-muted">
            <span>Type:</span>
            <select
              value={eventTypeFilter}
              onChange={(e) => setEventTypeFilter(e.target.value)}
              className="border border-fleet-border rounded px-2.5 py-1.5 bg-white text-fleet-text text-xs font-medium focus:ring-1 focus:ring-fleet-primary"
            >
              <option value="ALL">All Types</option>
              <option value="DROWSY">Drowsy</option>
              <option value="PROLONGED_CLOSURE">Prolonged Closure</option>
              <option value="DISTRACTED">Distracted</option>
              <option value="YAWN">Yawn</option>
              <option value="PHONE_DETECTED">Phone</option>
            </select>
          </div>

          {/* Refresh Button */}
          <button
            onClick={loadEvents}
            title="Refresh events"
            className="p-1.5 border border-fleet-border rounded hover:bg-fleet-bg text-fleet-muted hover:text-fleet-text transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
          </button>

          {/* Export to CSV Button */}
          <a
            href={getExportCsvUrl()}
            download="driver_safety_events.csv"
            className="flex items-center gap-1.5 px-3 py-1.5 bg-fleet-primary hover:bg-fleet-primaryHover text-white rounded text-xs font-bold transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {/* Events Table */}
      <div className="bg-white border border-fleet-border rounded-lg shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-[13px]">
            <thead>
              <tr className="bg-fleet-bg border-b border-fleet-border text-[11px] font-bold uppercase text-fleet-muted tracking-wider">
                <th className="py-3 px-4">Event ID</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Vehicle Node</th>
                <th className="py-3 px-4">Camera Source</th>
                <th className="py-3 px-4">Classification</th>
                <th className="py-3 px-4">Severity</th>
                <th className="py-3 px-4">Duration</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Telemetry Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-fleet-border">
              {events.length > 0 ? (
                events.map((e) => (
                  <tr key={e.id} className="hover:bg-[#FAFBFC] transition-colors">
                    <td className="py-2.5 px-4 font-mono text-xs text-fleet-muted">
                      {e.id}
                    </td>
                    <td className="py-2.5 px-4 text-fleet-text whitespace-nowrap">
                      {e.timestamp}
                    </td>
                    <td className="py-2.5 px-4 font-semibold text-fleet-text">
                      {e.vehicle_id}
                    </td>
                    <td className="py-2.5 px-4 text-fleet-muted font-mono text-xs">
                      {e.camera_id}
                    </td>
                    <td className="py-2.5 px-4 font-mono font-bold text-xs text-fleet-text">
                      {e.event_type}
                    </td>
                    <td className="py-2.5 px-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase ${
                          e.severity === "CRITICAL"
                            ? "bg-[#FDF2F2] text-fleet-critical border border-[#F8D2D2]"
                            : e.severity === "WARNING"
                            ? "bg-[#FFF9F0] text-fleet-warning border border-[#FCE6C7]"
                            : "bg-[#F0F8F3] text-fleet-normal border border-[#D1E8D8]"
                        }`}
                      >
                        {e.severity}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 font-mono text-fleet-text">
                      {e.duration_sec.toFixed(1)}s
                    </td>
                    <td className="py-2.5 px-4 font-mono text-fleet-muted">
                      {e.confidence !== null ? `${(e.confidence * 100).toFixed(0)}%` : "N/A"}
                    </td>
                    <td className="py-2.5 px-4 text-fleet-muted max-w-sm truncate">
                      {e.description}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-fleet-muted">
                    {isLoading ? "Loading telemetry records..." : "No events match current filter."}
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
