"""Real-time monitoring orchestrator connecting CV tracking, DL inference, and alerts."""
import cv2
import time
import base64
import numpy as np
from typing import Optional, Dict, Any, Tuple

from backend.config import SystemSettings
from backend.detection.face_landmarks import FaceLandmarksTracker
from backend.detection.eye_tracker import EyeTracker
from backend.detection.yawn_detector import YawnDetector
from backend.detection.head_pose import HeadPoseEstimator
from backend.detection.phone_detector import PhoneDetector
from backend.detection.alert_engine import AlertEngine, UnifiedDriverAssessment
from backend.models.inference import MobileNetInferenceEngine
from backend.services.camera_manager import CameraManager
from backend.services.events import EventService


class DriverMonitoringService:
    def __init__(self,
                 settings: SystemSettings,
                 camera_manager: CameraManager,
                 event_service: EventService):
        self.settings = settings
        self.camera_manager = camera_manager
        self.event_service = event_service

        # Initialize detection components
        self.face_tracker = FaceLandmarksTracker()
        self.eye_tracker = EyeTracker(
            ear_threshold=settings.detection.ear_threshold,
            closure_threshold_sec=settings.detection.eye_closure_duration_sec,
            smoothing_alpha=settings.detection.temporal_smoothing_alpha
        )
        self.yawn_detector = YawnDetector(
            mar_threshold=settings.detection.mar_threshold,
            yawn_duration_sec=settings.detection.yawn_duration_sec,
            smoothing_alpha=settings.detection.temporal_smoothing_alpha
        )
        self.head_pose_estimator = HeadPoseEstimator(
            yaw_threshold=settings.detection.head_yaw_threshold,
            pitch_threshold=settings.detection.head_pitch_threshold,
            distraction_duration_sec=settings.detection.distraction_duration_sec,
            smoothing_alpha=settings.detection.temporal_smoothing_alpha
        )
        self.phone_detector = PhoneDetector(
            confidence_threshold=settings.inference.phone_confidence_threshold,
            device="cpu"
        )
        self.mobilenet_engine = MobileNetInferenceEngine(
            checkpoint_path=settings.inference.mobilenet_checkpoint,
            device=settings.inference.preferred_device
        )
        self.alert_engine = AlertEngine(
            alert_cooldown_sec=settings.audio.alert_cooldown_sec,
            level_1_msg=settings.audio.level_1_message,
            level_2_msg=settings.audio.level_2_message,
            level_3_msg=settings.audio.level_3_message,
            audio_enabled=settings.audio.enable_audio
        )

        self.last_recorded_state: Optional[str] = None
        self.state_entry_time: float = time.time()
        self.active_vehicle_id: str = "LOCAL-VEHICLE-01"

    def update_settings(self, new_settings: SystemSettings):
        self.settings = new_settings
        self.eye_tracker.update_thresholds(
            new_settings.detection.ear_threshold,
            new_settings.detection.eye_closure_duration_sec,
            new_settings.detection.temporal_smoothing_alpha
        )
        self.yawn_detector.update_thresholds(
            new_settings.detection.mar_threshold,
            new_settings.detection.yawn_duration_sec
        )
        self.head_pose_estimator.update_thresholds(
            new_settings.detection.head_yaw_threshold,
            new_settings.detection.head_pitch_threshold,
            new_settings.detection.distraction_duration_sec
        )
        self.alert_engine.update_config(
            alert_cooldown_sec=new_settings.audio.alert_cooldown_sec,
            audio_enabled=new_settings.audio.enable_audio,
            level_1_msg=new_settings.audio.level_1_message,
            level_2_msg=new_settings.audio.level_2_message,
            level_3_msg=new_settings.audio.level_3_message
        )
        self.mobilenet_engine.reload_checkpoint(new_settings.inference.mobilenet_checkpoint)

    def process_frame(self,
                      frame_bgr: Optional[np.ndarray] = None,
                      include_annotated_frame: bool = False) -> Dict[str, Any]:
        """Processes a single camera frame through the end-to-end driver monitoring pipeline."""
        now = time.time()
        start_proc = time.perf_counter()

        # Step 1: Frame Acquisition
        if frame_bgr is None:
            ret, frame_bgr, _ = self.camera_manager.get_latest_frame()
            if not ret or frame_bgr is None:
                # Camera unavailable
                assessment = self.alert_engine.evaluate(
                    camera_available=False,
                    face_detected=False,
                    is_eyes_closed=False,
                    closure_duration_sec=0.0,
                    prolonged_closure=False,
                    is_yawning=False,
                    is_distracted=False,
                    distraction_duration_sec=0.0,
                    drowsy_probability=None,
                    phone_detected=False,
                    current_time=now
                )
                return {
                    "timestamp": now,
                    "camera_available": False,
                    "assessment": assessment.model_dump(),
                    "measurements": {
                        "ear": 0.0, "left_ear": 0.0, "right_ear": 0.0,
                        "mar": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                        "drowsy_prob": None, "inference_latency_ms": 0.0
                    },
                    "telemetry": {
                        "vehicle_id": self.active_vehicle_id,
                        "fps": self.camera_manager.fps_measured,
                        "pipeline_latency_ms": 0.0
                    }
                }

        h, w, _ = frame_bgr.shape

        # Step 2: Facial Landmarks Detection
        lm_result = self.face_tracker.process(frame_bgr)

        # Step 3: Facial Measurements
        eye_res = self.eye_tracker.process(lm_result.landmarks_pixel, now)
        yawn_res = self.yawn_detector.process(lm_result.landmarks_pixel, now)
        pose_res = self.head_pose_estimator.process(lm_result.landmarks_pixel, (h, w), now)

        # Step 4: Crop face for MobileNetV3 inference
        face_crop = None
        if lm_result.face_detected and lm_result.bounding_box:
            bx, by, bw, bh = lm_result.bounding_box
            # Add padding
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)
            if x2 > x1 and y2 > y1:
                face_crop = frame_bgr[y1:y2, x1:x2]

        # Step 5: MobileNetV3 Deep Learning Inference
        dl_res = self.mobilenet_engine.predict(face_crop)
        drowsy_prob = dl_res.drowsy_probability if dl_res.is_available else None

        # Step 6: Phone Detection (Optional module)
        phone_res = self.phone_detector.process(frame_bgr)

        # Step 7: Alert Engine Assessment
        assessment = self.alert_engine.evaluate(
            camera_available=True,
            face_detected=lm_result.face_detected,
            is_eyes_closed=eye_res.is_eyes_closed,
            closure_duration_sec=eye_res.closure_duration_sec,
            prolonged_closure=eye_res.prolonged_closure_detected,
            is_yawning=yawn_res.is_yawning,
            is_distracted=pose_res.is_distracted,
            distraction_duration_sec=pose_res.distraction_duration_sec,
            drowsy_probability=drowsy_prob,
            phone_detected=phone_res.phone_detected,
            current_time=now
        )

        # Step 8: Safety Event Logging
        if assessment.primary_state.value != self.last_recorded_state:
            # If transitioning to a warning or critical state, log event
            if assessment.severity.value in ["WARNING", "CRITICAL"]:
                duration = max(eye_res.closure_duration_sec, pose_res.distraction_duration_sec, yawn_res.yawn_duration_sec, 1.0)
                conf = drowsy_prob if drowsy_prob is not None else 0.90
                self.event_service.record_event(
                    event_type=assessment.primary_state.value,
                    severity=assessment.severity.value,
                    duration_sec=duration,
                    confidence=conf,
                    description=assessment.alert_message,
                    vehicle_id=self.active_vehicle_id,
                    camera_id=f"Camera-{self.camera_manager.current_index}"
                )
            self.last_recorded_state = assessment.primary_state.value
            self.state_entry_time = now

        pipeline_latency_ms = (time.perf_counter() - start_proc) * 1000.0

        # Build annotated frame if requested (or send landmark points directly)
        annotated_b64 = None
        if include_annotated_frame:
            annotated_frame = frame_bgr.copy()
            if lm_result.face_detected and lm_result.bounding_box:
                bx, by, bw, bh = lm_result.bounding_box
                box_color = (0, 200, 50) if assessment.severity.value == "NORMAL" else ((0, 165, 255) if assessment.severity.value == "WARNING" else (0, 0, 230))
                cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), box_color, 2)

            if phone_res.phone_detected and phone_res.bounding_box:
                px, py, pw, ph = phone_res.bounding_box
                cv2.rectangle(annotated_frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"PHONE {phone_res.confidence:.2f}", (px, max(20, py - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            ret_enc, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ret_enc:
                annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "timestamp": now,
            "camera_available": True,
            "face_detected": lm_result.face_detected,
            "bounding_box": lm_result.bounding_box,
            "bounding_box_norm": getattr(lm_result, "bounding_box_norm", None),
            "assessment": assessment.model_dump(),
            "measurements": {
                "ear": eye_res.smoothed_ear,
                "left_ear": eye_res.left_ear,
                "right_ear": eye_res.right_ear,
                "eye_closure_sec": eye_res.closure_duration_sec,
                "is_eyes_closed": eye_res.is_eyes_closed,
                "mar": yawn_res.smoothed_mar,
                "is_yawning": yawn_res.is_yawning,
                "yaw": pose_res.yaw,
                "pitch": pose_res.pitch,
                "roll": pose_res.roll,
                "head_direction": pose_res.direction,
                "is_distracted": pose_res.is_distracted,
                "distraction_sec": pose_res.distraction_duration_sec,
                "drowsy_prob": drowsy_prob,
                "dl_predicted_label": dl_res.predicted_label,
                "inference_latency_ms": dl_res.latency_ms,
                "model_available": dl_res.is_available,
                "phone_detected": phone_res.phone_detected,
                "phone_confidence": phone_res.confidence,
                "phone_available": phone_res.available
            },
            "telemetry": {
                "vehicle_id": self.active_vehicle_id,
                "fps": self.camera_manager.fps_measured,
                "pipeline_latency_ms": round(pipeline_latency_ms, 2)
            },
            "frame_jpg_base64": annotated_b64
        }
