import React, { useState, useEffect, useRef, useCallback } from "react";
import { Sidebar, NavTab } from "./components/Sidebar";
import { Header } from "./components/Header";
import { OverviewPage } from "./pages/OverviewPage";
import { LiveMonitoringPage } from "./pages/LiveMonitoringPage";
import { VehiclesPage } from "./pages/VehiclesPage";
import { ModelTrainingPage } from "./pages/ModelTrainingPage";
import { EventHistoryPage } from "./pages/EventHistoryPage";
import { SettingsPage } from "./pages/SettingsPage";
import { MonitoringPayload, SystemSettingsData } from "./types";
import { fetchHealth, fetchSettings, WS_URL } from "./services/api";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>("monitoring");
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isModelReady, setIsModelReady] = useState<boolean>(false);
  const [telemetry, setTelemetry] = useState<MonitoringPayload | null>(null);
  const [settings, setSettings] = useState<SystemSettingsData | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Poll backend health
  const checkHealth = useCallback(async () => {
    try {
      const data = await fetchHealth();
      if (data && data.status === "healthy") {
        setIsBackendConnected(true);
      }
    } catch {
      setIsBackendConnected(false);
    }
  }, []);

  // Initialize and connect WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsBackendConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.measurements) {
            setTelemetry(payload);
            if (payload.measurements.model_available !== undefined) {
              setIsModelReady(payload.measurements.model_available);
            }
          }
        } catch (e) {
          // ignore malformed frame
        }
      };

      ws.onclose = () => {
        setIsBackendConnected(false);
        // Attempt reconnect after 2 seconds
        if (!reconnectTimeoutRef.current) {
          reconnectTimeoutRef.current = window.setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connectWebSocket();
          }, 2000);
        }
      };

      ws.onerror = () => {
        setIsBackendConnected(false);
        ws.close();
      };
    } catch (e) {
      setIsBackendConnected(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    connectWebSocket();
    fetchSettings().then(setSettings).catch(() => {});

    const interval = setInterval(checkHealth, 5000);
    return () => {
      clearInterval(interval);
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [checkHealth, connectWebSocket]);

  // Send camera frame over WebSocket to backend
  const handleSendFrame = useCallback((base64Img: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "FRAME",
          image: base64Img,
        })
      );
    }
  }, []);

  // Trigger calibration command
  const handleTriggerCalibration = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "CALIBRATE" }));
    }
  }, []);

  const handleFinishCalibration = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "FINISH_CALIBRATE" }));
    }
  }, []);

  const getPageTitle = (tab: NavTab): string => {
    switch (tab) {
      case "overview":
        return "Fleet Safety Overview";
      case "monitoring":
        return "Live Commercial Driver Monitoring";
      case "vehicles":
        return "Fleet Vehicle Registry";
      case "model":
        return "Deep Learning & Model Evaluation";
      case "events":
        return "Safety Event Audit History";
      case "settings":
        return "Edge Vision Configuration";
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-fleet-bg text-fleet-text font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isBackendConnected={isBackendConnected}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Header */}
        <Header
          pageTitle={getPageTitle(activeTab)}
          isBackendConnected={isBackendConnected}
          isModelReady={isModelReady}
          cameraFps={telemetry?.telemetry.fps || 0}
        />

        {/* Scrollable Page Body */}
        <main className="flex-1 overflow-y-auto">
          {activeTab === "overview" && (
            <OverviewPage
              telemetry={telemetry}
              onNavigateToMonitoring={() => setActiveTab("monitoring")}
              isBackendConnected={isBackendConnected}
            />
          )}

          {activeTab === "monitoring" && (
            <LiveMonitoringPage
              telemetry={telemetry}
              onSendFrame={handleSendFrame}
              onTriggerCalibration={handleTriggerCalibration}
              onFinishCalibration={handleFinishCalibration}
              settings={settings}
            />
          )}

          {activeTab === "vehicles" && <VehiclesPage telemetry={telemetry} />}

          {activeTab === "model" && <ModelTrainingPage />}

          {activeTab === "events" && <EventHistoryPage />}

          {activeTab === "settings" && <SettingsPage />}
        </main>
      </div>
    </div>
  );
};

export default App;
