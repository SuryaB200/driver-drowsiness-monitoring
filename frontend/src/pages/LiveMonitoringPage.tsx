import React, { useEffect, useRef, useState } from "react";
import {
  Video,
  VideoOff,
  RefreshCw,
  Sliders,
  Volume2,
  VolumeX,
  Play,
  Square,
  AlertOctagon,
  AlertTriangle,
  CheckCircle,
  Eye,
  Smile,
  Compass,
  Smartphone,
  Gauge,
  UserCheck,
} from "lucide-react";
import { MonitoringPayload, CameraDevice, SystemSettingsData } from "../types";
import { useCameraStream } from "../hooks/useCameraStream";
import { useDriverAudioAlert } from "../hooks/useDriverAudioAlert";

interface LiveMonitoringPageProps {
  telemetry: MonitoringPayload | null;
  onSendFrame: (base64Img: string) => void;
  onTriggerCalibration: () => void;
  onFinishCalibration: () => void;
  settings: SystemSettingsData | null;
}

export const LiveMonitoringPage: React.FC<LiveMonitoringPageProps> = ({
  telemetry,
  onSendFrame,
  onTriggerCalibration,
  onFinishCalibration,
  settings,
}) => {
  const {
    devices,
    selectedDeviceId,
    isStreaming,
    cameraError,
    videoRef,
    mirrorPreview,
    setMirrorPreview,
    currentResolution,
    startCamera,
    stopCamera,
    switchCamera,
    refreshDevices,
    captureFrameJpeg,
  } = useCameraStream();

  const {
    audioEnabled,
    setAudioEnabled,
    volume,
    setVolume,
    lastSpokenMessage,
    handleAudioAlert,
    speakAlert,
  } = useDriverAudioAlert(true, 1.0);

  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);
  const [calibrationCountdown, setCalibrationCountdown] = useState<number>(0);

  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameIntervalRef = useRef<number | null>(null);

  // Auto-trigger audio warnings when incoming assessment specifies audio payload
  useEffect(() => {
    if (telemetry && telemetry.assessment && telemetry.assessment.audio_alert) {
      handleAudioAlert(telemetry.assessment.audio_alert, settings?.audio.alert_cooldown_sec || 3.5);
    }
  }, [telemetry, handleAudioAlert, settings]);

  // Frame streaming loop: capture browser frame and send to backend WebSocket
  useEffect(() => {
    if (isStreaming) {
      frameIntervalRef.current = window.setInterval(() => {
        const frameData = captureFrameJpeg();
        if (frameData) {
          onSendFrame(frameData);
        }
      }, 45); // ~22 FPS
    } else {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    }
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
    };
  }, [isStreaming, captureFrameJpeg, onSendFrame]);

  // Draw HUD overlays (bounding box, alert banner, measurements)
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!isStreaming) return;

    // Apply preview mirroring to canvas if mirrored
    ctx.save();
    if (mirrorPreview) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }

    // Draw face bounding box if detected
    if (telemetry?.face_detected) {
      let bx = 0, by = 0, bw = 0, bh = 0;
      if (telemetry.bounding_box_norm) {
        const [nx, ny, nw, nh] = telemetry.bounding_box_norm;
        bx = nx * canvas.width;
        by = ny * canvas.height;
        bw = nw * canvas.width;
        bh = nh * canvas.height;
      } else if (telemetry.bounding_box) {
        const [rawX, rawY, rawW, rawH] = telemetry.bounding_box;
        const refW = rawX + rawW > 640 ? canvas.width : 640;
        const refH = rawY + rawH > 480 ? canvas.height : 360;
        bx = (rawX / refW) * canvas.width;
        by = (rawY / refH) * canvas.height;
        bw = (rawW / refW) * canvas.width;
        bh = (rawH / refH) * canvas.height;
      }

      if (bw > 0 && bh > 0) {
        const sev = telemetry.assessment.severity;
        const boxColor =
          sev === "CRITICAL"
            ? "#C43D3D"
            : sev === "WARNING"
            ? "#B7791F"
            : "#287D52";

        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(bx, by, bw, bh);

        // Corner accents
        const len = Math.min(24, bw * 0.2);
        ctx.lineWidth = 4;
        // Top-left
        ctx.beginPath();
        ctx.moveTo(bx, by + len);
        ctx.lineTo(bx, by);
        ctx.lineTo(bx + len, by);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(bx + bw - len, by);
        ctx.lineTo(bx + bw, by);
        ctx.lineTo(bx + bw, by + len);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(bx, by + bh - len);
        ctx.lineTo(bx, by + bh);
        ctx.lineTo(bx + len, by + bh);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(bx + bw - len, by + bh);
        ctx.lineTo(bx + bw, by + bh);
        ctx.lineTo(bx + bw, by + bh - len);
        ctx.stroke();

        // Face tag badge
        ctx.fillStyle = boxColor;
        ctx.font = "bold 11px sans-serif";
        const tagText = "FACE TRACKED";
        const tagW = ctx.measureText(tagText).width + 14;
        ctx.fillRect(bx, Math.max(0, by - 20), tagW, 18);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(tagText, bx + 7, Math.max(13, by - 7));
      }
    }

    ctx.restore();
  }, [telemetry, isStreaming, mirrorPreview, videoRef]);

  // Calibration routine (3 seconds of open-eyed calibration)
  const handleStartCalibration = () => {
    setIsCalibrating(true);
    setCalibrationCountdown(3);
    onTriggerCalibration();

    const interval = setInterval(() => {
      setCalibrationCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setIsCalibrating(false);
          onFinishCalibration();
          speakAlert("Calibration complete. Driver baseline recorded.", 1);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const primaryState = telemetry?.assessment.primary_state || "CAMERA UNAVAILABLE";
  const severity = telemetry?.assessment.severity || "NORMAL";
  const m = telemetry?.measurements;

  return (
    <div className="p-5 max-w-[1700px] mx-auto space-y-4">
      {/* Camera Selection & Control Toolbar */}
      <div className="bg-white border border-fleet-border rounded-lg p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <Video className="w-4 h-4 text-fleet-primary" />
            <span className="text-[13px] font-semibold text-fleet-text">
              Active Camera:
            </span>
          </div>

          {/* Camera Device Dropdown */}
          <select
            value={selectedDeviceId}
            onChange={(e) => switchCamera(e.target.value)}
            className="text-[13px] border border-fleet-border rounded px-3 py-1.5 bg-white text-fleet-text font-medium min-w-[240px] focus:outline-none focus:ring-1 focus:ring-fleet-primary"
          >
            {devices.map((d) => (
              <option key={d.deviceId || d.index} value={d.deviceId}>
                {d.name} ({d.max_res})
              </option>
            ))}
          </select>

          {/* Refresh Devices */}
          <button
            onClick={refreshDevices}
            title="Refresh Connected Devices"
            className="p-1.5 border border-fleet-border rounded hover:bg-fleet-bg text-fleet-muted hover:text-fleet-text transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          {/* Start / Stop Toggle */}
          {!isStreaming ? (
            <button
              onClick={() => startCamera()}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-fleet-primary hover:bg-fleet-primaryHover text-white rounded text-[13px] font-semibold transition-colors"
            >
              <Play className="w-3.5 h-3.5 fill-white" />
              <span>Start Camera</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-fleet-critical hover:opacity-90 text-white rounded text-[13px] font-semibold transition-colors"
            >
              <Square className="w-3.5 h-3.5 fill-white" />
              <span>Stop Camera</span>
            </button>
          )}

          {/* Mirror Preview Checkbox */}
          <label className="flex items-center gap-1.5 text-[12px] font-medium text-fleet-muted cursor-pointer ml-2">
            <input
              type="checkbox"
              checked={mirrorPreview}
              onChange={(e) => setMirrorPreview(e.target.checked)}
              className="rounded border-fleet-border text-fleet-primary focus:ring-0"
            />
            <span>Mirror Preview</span>
          </label>
        </div>

        {/* Right side toolbar: Calibration & Device Info */}
        <div className="flex items-center gap-3">
          <button
            onClick={handleStartCalibration}
            disabled={!isStreaming || isCalibrating}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-[12px] font-semibold border ${
              isCalibrating
                ? "bg-fleet-warning text-white border-fleet-warning animate-pulse"
                : "bg-white border-fleet-border text-fleet-text hover:bg-fleet-bg"
            } disabled:opacity-50`}
          >
            <UserCheck className="w-3.5 h-3.5" />
            <span>
              {isCalibrating
                ? `Calibrating... (${calibrationCountdown}s)`
                : "Calibrate Driver EAR"}
            </span>
          </button>

          {/* Audio Alert Mute Toggle */}
          <button
            onClick={() => {
              if (audioEnabled) {
                window.speechSynthesis?.cancel();
              }
              setAudioEnabled(!audioEnabled);
            }}
            title={audioEnabled ? "Mute Voice Alerts" : "Enable Voice Alerts"}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-[12px] font-semibold border transition-colors ${
              audioEnabled
                ? "bg-white border-fleet-border text-fleet-text hover:bg-fleet-bg"
                : "bg-red-50 border-red-200 text-red-600 hover:bg-red-100"
            }`}
          >
            {audioEnabled ? <Volume2 className="w-3.5 h-3.5 text-fleet-primary" /> : <VolumeX className="w-3.5 h-3.5" />}
            <span>{audioEnabled ? "Voice Alerts ON" : "Muted"}</span>
          </button>

          <div className="text-[12px] text-fleet-muted font-mono bg-fleet-bg px-2.5 py-1 rounded border border-fleet-border">
            Res: {currentResolution} •{" "}
            {telemetry?.telemetry.fps ? `${telemetry.telemetry.fps} FPS` : "0 FPS"}
          </div>
        </div>
      </div>

      {cameraError && (
        <div className="bg-[#FDF2F2] border border-[#F8D2D2] text-fleet-critical text-[13px] px-4 py-2.5 rounded-lg flex items-center gap-2">
          <AlertOctagon className="w-4 h-4 shrink-0" />
          <span>{cameraError}</span>
        </div>
      )}

      {/* Main 2-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column (8 cols): 16:9 Camera Feed */}
        <div className="lg:col-span-8 space-y-3">
          <div className="relative aspect-video bg-[#1B222C] rounded-lg overflow-hidden border border-fleet-border shadow-sm flex items-center justify-center">
            {/* Live Native Video */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-contain ${
                mirrorPreview ? "scale-x-[-1]" : ""
              } ${!isStreaming ? "hidden" : "block"}`}
            />

            {/* Canvas Overlay for bounding boxes and landmarks */}
            <canvas
              ref={overlayCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />

            {/* Offline Standby Overlay */}
            {!isStreaming && (
              <div className="flex flex-col items-center justify-center text-center p-6 text-[#9AA5B1]">
                <VideoOff className="w-12 h-12 mb-2 opacity-60" />
                <div className="text-sm font-semibold text-white">
                  Camera Feed Standby
                </div>
                <div className="text-xs text-[#8A96A4] mt-1 max-w-sm">
                  Click "Start Camera" above or connect an external USB webcam to
                  begin real-time edge monitoring.
                </div>
              </div>
            )}

            {/* Top-Left Live State HUD Badge */}
            {isStreaming && (
              <div className="absolute top-3 left-3 flex items-center gap-2">
                <span
                  className={`px-3 py-1 rounded text-xs font-bold uppercase tracking-wider text-white shadow-sm flex items-center gap-1.5 ${
                    severity === "CRITICAL"
                      ? "bg-fleet-critical animate-pulse"
                      : severity === "WARNING"
                      ? "bg-fleet-warning"
                      : "bg-fleet-normal"
                  }`}
                >
                  <span className="w-2 h-2 rounded-full bg-white animate-ping" />
                  {primaryState}
                </span>

                {telemetry?.face_detected ? (
                  <span className="px-2.5 py-1 rounded bg-black/60 text-[#34D399] font-semibold text-[11px] backdrop-blur-sm flex items-center gap-1.5 border border-[#34D399]/40">
                    <span className="w-2 h-2 rounded-full bg-[#34D399] inline-block animate-pulse" />
                    Face Detected
                  </span>
                ) : (
                  <span className="px-2.5 py-1 rounded bg-black/60 text-[#F87171] font-semibold text-[11px] backdrop-blur-sm flex items-center gap-1.5 border border-[#F87171]/40">
                    <span className="w-2 h-2 rounded-full bg-[#F87171] inline-block" />
                    Position Face in Frame
                  </span>
                )}

                {m?.drowsy_prob !== null && m?.drowsy_prob !== undefined && (
                  <span className="px-2 py-1 rounded bg-black/60 text-white font-mono text-[11px] backdrop-blur-sm">
                    MobileNet: {(m.drowsy_prob * 100).toFixed(0)}% Drowsy
                  </span>
                )}
              </div>
            )}

            {/* Bottom-Right Device Status Pill */}
            {isStreaming && (
              <div className="absolute bottom-3 right-3 px-2 py-0.5 rounded bg-black/70 text-[11px] font-mono text-gray-300">
                Lat: {m?.inference_latency_ms ? `${m.inference_latency_ms}ms` : "<1ms"} • Edge
              </div>
            )}
          </div>

          {/* Quick Guidance / Alert Cooldown Info Banner */}
          <div className="bg-white border border-fleet-border rounded-lg p-3 text-[12px] flex items-center justify-between text-fleet-muted">
            <div>
              <span className="font-semibold text-fleet-text">Active Telematics Event:</span>{" "}
              {telemetry?.assessment.alert_message || "Monitoring initialized. Waiting for stream."}
            </div>
            <div className="font-mono text-[11px]">
              Cooldown: {settings?.audio.alert_cooldown_sec || 3.5}s
            </div>
          </div>
        </div>

        {/* Right Column (4 cols): Measurement Panel & Alert Engine */}
        <div className="lg:col-span-4 space-y-4">
          {/* Highest Priority Alert Card */}
          <div
            className={`rounded-lg border p-4 shadow-sm transition-all ${
              severity === "CRITICAL"
                ? "bg-[#FDF2F2] border-[#F8D2D2]"
                : severity === "WARNING"
                ? "bg-[#FFF9F0] border-[#FCE6C7]"
                : "bg-white border-fleet-border"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-fleet-muted">
                System Advisory State
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase ${
                  severity === "CRITICAL"
                    ? "bg-fleet-critical text-white"
                    : severity === "WARNING"
                    ? "bg-fleet-warning text-white"
                    : "bg-fleet-normal text-white"
                }`}
              >
                {severity}
              </span>
            </div>

            <div className="text-[16px] font-bold text-fleet-text flex items-center gap-2">
              {severity === "CRITICAL" ? (
                <AlertOctagon className="w-5 h-5 text-fleet-critical shrink-0" />
              ) : severity === "WARNING" ? (
                <AlertTriangle className="w-5 h-5 text-fleet-warning shrink-0" />
              ) : (
                <CheckCircle className="w-5 h-5 text-fleet-normal shrink-0" />
              )}
              <span>{primaryState}</span>
            </div>

            <p className="text-[12px] text-fleet-muted mt-1 leading-snug">
              {telemetry?.assessment.alert_message || "No safety advisory active."}
            </p>

            {/* Active Conditions Pills */}
            <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-fleet-border/60">
              {telemetry?.assessment.active_conditions.length ? (
                telemetry.assessment.active_conditions.map((c) => (
                  <span
                    key={c}
                    className="text-[10px] font-semibold px-2 py-0.5 rounded bg-white border border-fleet-border text-fleet-text"
                  >
                    {c}
                  </span>
                ))
              ) : (
                <span className="text-[11px] text-fleet-muted">
                  No abnormal conditions flagged
                </span>
              )}
            </div>
          </div>

          {/* Real-time Facial Measurements Panel */}
          <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm space-y-3.5">
            <div className="flex items-center justify-between border-b border-fleet-border pb-2">
              <span className="text-[13px] font-bold text-fleet-text flex items-center gap-1.5">
                <Gauge className="w-4 h-4 text-fleet-primary" />
                Facial Telemetry
              </span>
              <span className="text-[11px] text-fleet-muted font-mono">
                Thresh: EAR &lt; {settings?.detection.ear_threshold || 0.21}
              </span>
            </div>

            {/* Compact Metric Rows */}
            <div className="space-y-2 text-[13px]">
              {/* EAR Metrics */}
              <div className="flex items-center justify-between py-1 border-b border-fleet-border/40">
                <span className="text-fleet-muted flex items-center gap-1.5">
                  <Eye className="w-3.5 h-3.5 text-fleet-primary" />
                  Eye Aspect Ratio (EAR)
                </span>
                <span className="font-mono font-bold text-fleet-text">
                  {m ? m.ear.toFixed(3) : "—"}
                  <span className="text-[11px] font-normal text-fleet-muted ml-1">
                    (L: {m ? m.left_ear.toFixed(2) : "—"} | R: {m ? m.right_ear.toFixed(2) : "—"})
                  </span>
                </span>
              </div>

              {/* Eye Closure Duration */}
              <div className="flex items-center justify-between py-1 border-b border-fleet-border/40">
                <span className="text-fleet-muted">Closure Duration</span>
                <span
                  className={`font-mono font-bold ${
                    (m?.eye_closure_sec || 0) >= 1.2
                      ? "text-fleet-critical font-extrabold"
                      : "text-fleet-text"
                  }`}
                >
                  {m ? `${m.eye_closure_sec.toFixed(2)}s` : "0.0s"}
                </span>
              </div>

              {/* MAR (Yawn) */}
              <div className="flex items-center justify-between py-1 border-b border-fleet-border/40">
                <span className="text-fleet-muted flex items-center gap-1.5">
                  <Smile className="w-3.5 h-3.5 text-fleet-primary" />
                  Mouth Ratio (MAR)
                </span>
                <span className="font-mono font-bold text-fleet-text">
                  {m ? m.mar.toFixed(3) : "—"}
                  {m?.is_yawning && (
                    <span className="text-[10px] text-fleet-warning font-bold ml-1.5">
                      YAWNING
                    </span>
                  )}
                </span>
              </div>

              {/* Head Pose Orientation */}
              <div className="flex items-center justify-between py-1 border-b border-fleet-border/40">
                <span className="text-fleet-muted flex items-center gap-1.5">
                  <Compass className="w-3.5 h-3.5 text-fleet-primary" />
                  Head Pose (Yaw / Pitch)
                </span>
                <span className="font-mono font-bold text-fleet-text">
                  {m ? `${m.yaw.toFixed(0)}° / ${m.pitch.toFixed(0)}°` : "—"}
                  <span className="text-[11px] text-fleet-muted ml-1">
                    [{m?.head_direction || "FORWARD"}]
                  </span>
                </span>
              </div>

              {/* Distraction Duration */}
              <div className="flex items-center justify-between py-1 border-b border-fleet-border/40">
                <span className="text-fleet-muted">Distraction Timer</span>
                <span
                  className={`font-mono font-bold ${
                    (m?.distraction_sec || 0) >= 2.0
                      ? "text-fleet-warning"
                      : "text-fleet-text"
                  }`}
                >
                  {m ? `${m.distraction_sec.toFixed(1)}s` : "0.0s"}
                </span>
              </div>

              {/* Mobile Phone Detection */}
              <div className="flex items-center justify-between py-1 border-b border-fleet-border/40">
                <span className="text-fleet-muted flex items-center gap-1.5">
                  <Smartphone className="w-3.5 h-3.5 text-fleet-primary" />
                  Mobile Phone Presence
                </span>
                <span
                  className={`font-semibold text-[12px] ${
                    m?.phone_detected
                      ? "text-fleet-critical font-bold"
                      : m?.phone_available
                      ? "text-fleet-normal"
                      : "text-fleet-muted"
                  }`}
                >
                  {m?.phone_detected
                    ? `DETECTED (${(m.phone_confidence * 100).toFixed(0)}%)`
                    : m?.phone_available
                    ? "None Visible"
                    : "Detector Standby"}
                </span>
              </div>

              {/* Deep Learning Model Score */}
              <div className="flex items-center justify-between py-1">
                <span className="text-fleet-muted">MobileNetV3 Classification</span>
                <span className="font-mono font-bold text-fleet-text">
                  {m?.drowsy_prob !== null && m?.drowsy_prob !== undefined
                    ? `${(m.drowsy_prob * 100).toFixed(1)}% Drowsy`
                    : "Awaiting checkpoint"}
                </span>
              </div>
            </div>
          </div>

          {/* Audio Warning Controls */}
          <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[13px] font-bold text-fleet-text flex items-center gap-1.5">
                {audioEnabled ? (
                  <Volume2 className="w-4 h-4 text-fleet-primary" />
                ) : (
                  <VolumeX className="w-4 h-4 text-fleet-muted" />
                )}
                In-Cab Audio System
              </span>
              <button
                onClick={() => setAudioEnabled(!audioEnabled)}
                className={`text-[12px] font-semibold px-2.5 py-1 rounded border transition-colors ${
                  audioEnabled
                    ? "bg-[#F0F8F3] border-[#D1E8D8] text-fleet-normal"
                    : "bg-fleet-bg border-fleet-border text-fleet-muted"
                }`}
              >
                {audioEnabled ? "Audio Active" : "Muted"}
              </button>
            </div>

            {/* Volume Slider */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] text-fleet-muted">
                <span>Alert Volume</span>
                <span>{Math.round(volume * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-fleet-border rounded-lg appearance-none cursor-pointer accent-fleet-primary"
              />
            </div>

            {/* Manual Announcement Test Triggers */}
            <div className="pt-2 border-t border-fleet-border/60">
              <div className="text-[11px] font-semibold text-fleet-muted mb-1.5">
                Simulate In-Cab Audio Prompts:
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                <button
                  onClick={() =>
                    speakAlert(
                      settings?.audio.level_1_message ||
                        "Warning. Please focus on the road.",
                      1
                    )
                  }
                  className="px-2 py-1 text-[11px] font-medium border border-fleet-border rounded bg-white hover:bg-fleet-bg text-fleet-text text-center"
                >
                  Level 1 (Distract)
                </button>
                <button
                  onClick={() =>
                    speakAlert(
                      settings?.audio.level_2_message ||
                        "Alert! Drowsiness detected. Take a safe break.",
                      2
                    )
                  }
                  className="px-2 py-1 text-[11px] font-medium border border-fleet-border rounded bg-white hover:bg-fleet-bg text-fleet-warning text-center"
                >
                  Level 2 (Drowsy)
                </button>
                <button
                  onClick={() =>
                    speakAlert(
                      settings?.audio.level_3_message ||
                        "Critical warning! Driver attention required.",
                      3
                    )
                  }
                  className="px-2 py-1 text-[11px] font-medium border border-fleet-border rounded bg-white hover:bg-fleet-bg text-fleet-critical font-semibold text-center"
                >
                  Level 3 (Critical)
                </button>
              </div>
            </div>

            {lastSpokenMessage && (
              <div className="text-[11px] text-fleet-muted italic bg-fleet-bg p-1.5 rounded truncate">
                Last Spoken: "{lastSpokenMessage}"
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
