import { useState, useEffect, useRef, useCallback } from "react";
import { CameraDevice } from "../types";

export function useCameraStream() {
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [mirrorPreview, setMirrorPreview] = useState<boolean>(false);
  const [currentResolution, setCurrentResolution] = useState<string>("1280x720");
  const [cameraFps, setCameraFps] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const lastCaptureTimeRef = useRef<number>(0);

  // Discover connected camera devices
  const refreshDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        setCameraError("MediaDevices API is unsupported in this browser.");
        return [];
      }

      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices.filter((d) => d.kind === "videoinput");

      const mapped: CameraDevice[] = videoInputs.map((d, idx) => ({
        index: idx,
        deviceId: d.deviceId,
        name: d.label || `Camera ${idx + 1} (${idx === 0 ? "Integrated" : "External USB"})`,
        is_active: d.deviceId === selectedDeviceId,
        max_res: "1280x720",
      }));

      setDevices(mapped);

      // Default select first camera if none selected
      if (!selectedDeviceId && mapped.length > 0) {
        setSelectedDeviceId(mapped[0].deviceId || "");
      }
      return mapped;
    } catch (e: any) {
      setCameraError(`Failed to enumerate cameras: ${e.message}`);
      return [];
    }
  }, [selectedDeviceId]);

  // Stop camera tracks cleanly
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (e) {}
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  // Start video capture from selected camera device
  const startCamera = useCallback(
    async (deviceIdToUse?: string) => {
      stopCamera();
      setCameraError(null);

      const targetDeviceId = deviceIdToUse || selectedDeviceId;

      try {
        const constraints: MediaStreamConstraints = {
          video: {
            deviceId: targetDeviceId ? { exact: targetDeviceId } : undefined,
            width: { ideal: 1280, min: 640 },
            height: { ideal: 720, min: 480 },
            frameRate: { ideal: 30, min: 15 },
          },
          audio: false,
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }

        // Read actual track settings
        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack) {
          const settings = videoTrack.getSettings();
          const w = settings.width || 1280;
          const h = settings.height || 720;
          setCurrentResolution(`${w}x${h}`);
        }

        setIsStreaming(true);

        // Refresh device labels now that permission has been granted
        await refreshDevices();
      } catch (err: any) {
        let msg = err.message || "Failed to start camera.";
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          msg = "Camera permission denied. Please allow camera access in browser settings.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          msg = "Selected camera device was not found or is disconnected.";
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
          msg = "Camera is currently in use by another application.";
        }
        setCameraError(msg);
        setIsStreaming(false);
      }
    },
    [selectedDeviceId, stopCamera, refreshDevices]
  );

  // Switch camera cleanly
  const switchCamera = useCallback(
    async (newDeviceId: string) => {
      setSelectedDeviceId(newDeviceId);
      await startCamera(newDeviceId);
    },
    [startCamera]
  );

  // Capture current frame as JPEG Base64 for sending to backend WebSocket
  const captureFrameJpeg = useCallback((): string | null => {
    if (!videoRef.current || !isStreaming) return null;
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }
    const canvas = canvasRef.current;

    // Send at optimized resolution (e.g. 640x360 or 480x270) to ensure high throughput
    const targetW = 640;
    const targetH = Math.round((video.videoHeight / video.videoWidth) * targetW) || 360;

    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    // Notice: Draw non-mirrored so AI inference coordinates are always true physical camera space!
    ctx.drawImage(video, 0, 0, targetW, targetH);
    return canvas.toDataURL("image/jpeg", 0.65);
  }, [isStreaming]);

  // Initial enumeration on mount
  useEffect(() => {
    refreshDevices();

    // Listen to device connection / disconnection events
    const handleDeviceChange = () => {
      refreshDevices();
    };

    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", handleDeviceChange);
    }

    return () => {
      stopCamera();
      if (navigator.mediaDevices && navigator.mediaDevices.removeEventListener) {
        navigator.mediaDevices.removeEventListener("devicechange", handleDeviceChange);
      }
    };
  }, []);

  return {
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    isStreaming,
    cameraError,
    videoRef,
    mirrorPreview,
    setMirrorPreview,
    currentResolution,
    cameraFps,
    startCamera,
    stopCamera,
    switchCamera,
    refreshDevices,
    captureFrameJpeg,
  };
}
