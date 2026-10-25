"""Configuration manager for Commercial Fleet Driver Drowsiness and Distraction Monitoring System."""
import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = BASE_DIR / "backend" / "settings.json"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure directories exist
for path in [CHECKPOINTS_DIR, DATASETS_DIR, RESULTS_DIR, PLOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


class DetectionConfig(BaseModel):
    ear_threshold: float = Field(default=0.19, description="Eye Aspect Ratio threshold below which eye is considered closed")
    eye_closure_duration_sec: float = Field(default=1.2, description="Minimum seconds of eye closure to flag drowsiness/critical")
    mar_threshold: float = Field(default=0.65, description="Mouth Aspect Ratio threshold for yawning detection")
    yawn_duration_sec: float = Field(default=1.8, description="Minimum seconds of sustained mouth opening for yawn event")
    head_yaw_threshold: float = Field(default=28.0, description="Horizontal head rotation limit in degrees")
    head_pitch_threshold: float = Field(default=22.0, description="Vertical head tilt limit in degrees")
    distraction_duration_sec: float = Field(default=2.0, description="Duration in seconds before head turn triggers distraction alert")
    temporal_smoothing_alpha: float = Field(default=0.35, description="EMA smoothing factor for facial metrics")


class CameraConfig(BaseModel):
    selected_device_id: int = Field(default=0, description="Backend OpenCV camera index")
    device_name: str = Field(default="Default Camera", description="Camera display label")
    resolution_width: int = Field(default=1280, description="Capture frame width")
    resolution_height: int = Field(default=720, description="Capture frame height")
    target_fps: int = Field(default=30, description="Target frame capture rate")
    preview_mirroring: bool = Field(default=False, description="Whether preview is mirrored (does not affect pose math)")
    auto_reconnect: bool = Field(default=True, description="Attempt automatic reconnection on frame capture drop")


class AudioConfig(BaseModel):
    enable_audio: bool = Field(default=True, description="Master audio switch")
    alert_cooldown_sec: float = Field(default=3.5, description="Cooldown between successive audio alerts")
    volume: float = Field(default=1.0, description="Volume level (0.0 to 1.0)")
    level_1_message: str = Field(default="Warning. Please focus on the road.", description="Distraction warning message")
    level_2_message: str = Field(default="Alert! Drowsiness detected. Take a safe break.", description="Drowsiness warning message")
    level_3_message: str = Field(default="Critical warning! Driver attention required.", description="Prolonged eye closure message")


class InferenceConfig(BaseModel):
    preferred_device: str = Field(default="auto", description="'auto', 'cuda', or 'cpu'")
    image_size: int = Field(default=224, description="MobileNetV3 input resolution")
    mobilenet_checkpoint: str = Field(default=str(CHECKPOINTS_DIR / "best_mobilenet_v3.pth"), description="Checkpoint path")
    enable_phone_detector: bool = Field(default=True, description="Whether phone detection module is enabled")
    phone_confidence_threshold: float = Field(default=0.45, description="Confidence threshold for phone detection")


class SystemSettings(BaseModel):
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def load_settings() -> SystemSettings:
    """Load settings from JSON file or return defaults."""
    if CONFIG_FILE_PATH.exists():
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SystemSettings(**data)
        except Exception:
            return SystemSettings()
    return SystemSettings()


def save_settings(settings: SystemSettings) -> bool:
    """Persist system settings to disk."""
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(settings.model_dump_json(indent=2))
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False
