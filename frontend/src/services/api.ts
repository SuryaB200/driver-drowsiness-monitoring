import { SystemSettingsData, SafetyEventRecord, ModelMetricsData } from "../types";

const isBrowser = typeof window !== "undefined";
const isLocal = isBrowser && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");

// If running in local development with Vite dev server (port 5173), point to backend at port 8000
// If running in unified production/cloud deployment, point to the current origin
const defaultHttp = isBrowser
  ? (isLocal && window.location.port === "5173"
      ? "http://localhost:8000"
      : `${window.location.protocol}//${window.location.host}`)
  : "http://localhost:8000";

const defaultWs = isBrowser
  ? (isLocal && window.location.port === "5173"
      ? "ws://localhost:8000/ws/monitor"
      : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/monitor`)
  : "ws://localhost:8000/ws/monitor";

export const BACKEND_URL = (import.meta as any).env?.VITE_BACKEND_URL || defaultHttp;
export const WS_URL = (import.meta as any).env?.VITE_WS_URL || defaultWs;
const API_BASE = `${BACKEND_URL}/api`;

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchCameras() {
  const res = await fetch(`${API_BASE}/cameras`);
  return res.json();
}

export async function selectCamera(index: number) {
  const res = await fetch(`${API_BASE}/cameras/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index }),
  });
  return res.json();
}

export async function startCamera() {
  const res = await fetch(`${API_BASE}/cameras/start`, { method: "POST" });
  return res.json();
}

export async function stopCamera() {
  const res = await fetch(`${API_BASE}/cameras/stop`, { method: "POST" });
  return res.json();
}

export async function fetchEvents(severity = "ALL", event_type = "ALL", search = "", limit = 100) {
  const params = new URLSearchParams({
    severity,
    event_type,
    search,
    limit: String(limit),
  });
  const res = await fetch(`${API_BASE}/events?${params}`);
  return res.json();
}

export function getExportCsvUrl() {
  return `${API_BASE}/events/export`;
}

export async function fetchSettings(): Promise<SystemSettingsData> {
  const res = await fetch(`${API_BASE}/settings`);
  return res.json();
}

export async function updateSettings(settings: SystemSettingsData) {
  const res = await fetch(`${API_BASE}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  return res.json();
}

export async function fetchModelStatus() {
  const res = await fetch(`${API_BASE}/model/status`);
  return res.json();
}

export async function startTraining(epochs = 15, batch_size = 16, learning_rate = 0.0001, use_aug = true) {
  const res = await fetch(`${API_BASE}/model/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      epochs,
      batch_size,
      learning_rate,
      use_augmentation: use_aug,
    }),
  });
  return res.json();
}

export async function fetchTrainingStatus() {
  const res = await fetch(`${API_BASE}/model/training-status`);
  return res.json();
}

export async function evaluateModel() {
  const res = await fetch(`${API_BASE}/model/evaluate`, { method: "POST" });
  return res.json();
}

export async function fetchMetrics(): Promise<{ available: boolean; metrics?: ModelMetricsData; error?: string }> {
  const res = await fetch(`${API_BASE}/model/metrics`);
  return res.json();
}

export async function fetchDatasetStatus() {
  const res = await fetch(`${API_BASE}/dataset/status`);
  return res.json();
}

export async function prepareDataset() {
  const res = await fetch(`${API_BASE}/dataset/prepare`, { method: "POST" });
  return res.json();
}
