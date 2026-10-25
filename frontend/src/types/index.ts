export type DriverState =
  | "ALERT"
  | "DROWSY"
  | "DISTRACTED"
  | "YAWNING"
  | "EYES CLOSED"
  | "NO FACE DETECTED"
  | "CAMERA UNAVAILABLE"
  | "MODEL UNAVAILABLE";

export type AlertSeverity = "NORMAL" | "WARNING" | "CRITICAL";

export interface AudioAlertPayload {
  should_play: boolean;
  level: number;
  message: string;
  severity: AlertSeverity;
}

export interface DriverAssessment {
  primary_state: DriverState;
  active_conditions: string[];
  severity: AlertSeverity;
  audio_alert: AudioAlertPayload | null;
  alert_message: string;
}

export interface Measurements {
  ear: number;
  left_ear: number;
  right_ear: number;
  eye_closure_sec: number;
  is_eyes_closed: boolean;
  mar: number;
  is_yawning: boolean;
  yaw: number;
  pitch: number;
  roll: number;
  head_direction: string;
  is_distracted: boolean;
  distraction_sec: number;
  drowsy_prob: number | null;
  dl_predicted_label?: string;
  inference_latency_ms: number;
  model_available: boolean;
  phone_detected: boolean;
  phone_confidence: number;
  phone_available: boolean;
}

export interface Telemetry {
  vehicle_id: string;
  fps: number;
  pipeline_latency_ms: number;
}

export interface MonitoringPayload {
  timestamp: number;
  camera_available: boolean;
  face_detected?: boolean;
  bounding_box?: [number, number, number, number] | null;
  bounding_box_norm?: [number, number, number, number] | null;
  assessment: DriverAssessment;
  measurements: Measurements;
  telemetry: Telemetry;
  frame_jpg_base64?: string | null;
}

export interface CameraDevice {
  index: number;
  name: string;
  is_active: boolean;
  max_res: string;
  deviceId?: string; // from browser enumerateDevices
}

export interface SafetyEventRecord {
  id: string;
  timestamp: string;
  epoch_time: number;
  vehicle_id: string;
  camera_id: string;
  event_type: string;
  severity: string;
  duration_sec: number;
  confidence: number | null;
  description: string;
}

export interface SystemSettingsData {
  detection: {
    ear_threshold: number;
    eye_closure_duration_sec: number;
    mar_threshold: number;
    yawn_duration_sec: number;
    head_yaw_threshold: number;
    head_pitch_threshold: number;
    distraction_duration_sec: number;
    temporal_smoothing_alpha: number;
  };
  camera: {
    selected_device_id: number;
    device_name: string;
    resolution_width: number;
    resolution_height: number;
    target_fps: number;
    preview_mirroring: boolean;
    auto_reconnect: boolean;
  };
  audio: {
    enable_audio: boolean;
    alert_cooldown_sec: number;
    volume: number;
    level_1_message: string;
    level_2_message: string;
    level_3_message: string;
  };
  inference: {
    preferred_device: string;
    image_size: number;
    mobilenet_checkpoint: string;
    enable_phone_detector: boolean;
    phone_confidence_threshold: number;
  };
}

export interface ModelMetricsData {
  evaluation_timestamp: number;
  model_architecture: string;
  device_tested: string;
  total_test_samples: number;
  class_names: string[];
  test_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  confusion_matrix: number[][];
  classification_report: Record<string, any>;
  latency_benchmarks_ms: {
    mean: number;
    median: number;
    p95: number;
  };
  plots_saved: {
    confusion_matrix: string;
    roc_curve: string;
    loss_curves: string;
    accuracy_curves: string;
  };
}
