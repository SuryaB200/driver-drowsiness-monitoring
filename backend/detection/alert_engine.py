"""Unified Alert Engine and Driver State Decision Logic."""
import time
from enum import Enum
from typing import Optional, List, Dict, Any, NamedTuple
from pydantic import BaseModel

class AlertSeverity(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DriverState(str, Enum):
    ALERT = "ALERT"
    DROWSY = "DROWSY"
    DISTRACTED = "DISTRACTED"
    YAWNING = "YAWNING"
    EYES_CLOSED = "EYES CLOSED"
    NO_FACE_DETECTED = "NO FACE DETECTED"
    CAMERA_UNAVAILABLE = "CAMERA UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL UNAVAILABLE"


class AudioAlertPayload(BaseModel):
    should_play: bool
    level: int           # 1, 2, or 3
    message: str
    severity: AlertSeverity


class UnifiedDriverAssessment(BaseModel):
    primary_state: DriverState
    active_conditions: List[str]
    severity: AlertSeverity
    audio_alert: Optional[AudioAlertPayload]
    alert_message: str


class AlertEngine:
    def __init__(self,
                 alert_cooldown_sec: float = 3.5,
                 level_1_msg: str = "Warning. Please focus on the road.",
                 level_2_msg: str = "Alert! Drowsiness detected. Take a safe break.",
                 level_3_msg: str = "Critical warning! Driver attention required.",
                 audio_enabled: bool = True):
        self.alert_cooldown_sec = alert_cooldown_sec
        self.level_1_msg = level_1_msg
        self.level_2_msg = level_2_msg
        self.level_3_msg = level_3_msg
        self.audio_enabled = audio_enabled

        self.last_alert_time: float = 0.0
        self.last_played_level: int = 0
        self.prolonged_closure_accum_sec: float = 0.0

    def update_config(self,
                      alert_cooldown_sec: float,
                      audio_enabled: bool,
                      level_1_msg: Optional[str] = None,
                      level_2_msg: Optional[str] = None,
                      level_3_msg: Optional[str] = None):
        self.alert_cooldown_sec = alert_cooldown_sec
        self.audio_enabled = audio_enabled
        if level_1_msg:
            self.level_1_msg = level_1_msg
        if level_2_msg:
            self.level_2_msg = level_2_msg
        if level_3_msg:
            self.level_3_msg = level_3_msg

    def evaluate(self,
                 camera_available: bool,
                 face_detected: bool,
                 is_eyes_closed: bool,
                 closure_duration_sec: float,
                 prolonged_closure: bool,
                 is_yawning: bool,
                 is_distracted: bool,
                 distraction_duration_sec: float,
                 drowsy_probability: Optional[float],
                 phone_detected: bool,
                 current_time: Optional[float] = None) -> UnifiedDriverAssessment:
        if current_time is None:
            current_time = time.time()

        active_conditions: List[str] = []

        # 1. Hardware / Sensor checks
        if not camera_available:
            return UnifiedDriverAssessment(
                primary_state=DriverState.CAMERA_UNAVAILABLE,
                active_conditions=["CAMERA_UNAVAILABLE"],
                severity=AlertSeverity.CRITICAL,
                audio_alert=None,
                alert_message="Camera feed offline or unavailable."
            )

        if not face_detected:
            return UnifiedDriverAssessment(
                primary_state=DriverState.NO_FACE_DETECTED,
                active_conditions=["NO_FACE_DETECTED"],
                severity=AlertSeverity.WARNING,
                audio_alert=None,
                alert_message="Driver face not detected in camera frame."
            )

        # 2. Gather all active visual cues
        if is_eyes_closed:
            active_conditions.append("EYES_CLOSED")
        if prolonged_closure:
            active_conditions.append("PROLONGED_CLOSURE")
        if is_yawning:
            active_conditions.append("YAWNING")
        if is_distracted:
            active_conditions.append("HEAD_DISTRACTED")
        if phone_detected:
            active_conditions.append("PHONE_VISIBLE")
        if drowsy_probability is not None and drowsy_probability >= 0.65:
            active_conditions.append("MODEL_DROWSY")

        # 3. Determine severity and primary state with strict priority hierarchy:
        # Priority 1: Critical prolonged closure (closure > 2.5s) -> CRITICAL (Level 3)
        # Priority 2: Prolonged closure or high model drowsiness -> CRITICAL (Level 2)
        # Priority 3: Sustained distraction / Head turn / Phone -> WARNING (Level 1)
        # Priority 4: Yawning / Brief eye closure -> WARNING (Level 1)
        # Priority 5: Alert -> NORMAL

        primary_state = DriverState.ALERT
        severity = AlertSeverity.NORMAL
        message = "Driver is attentive. Normal operations."
        trigger_level: int = 0
        audio_text: str = ""

        if closure_duration_sec >= 2.5:
            # Critical prolonged closure
            primary_state = DriverState.DROWSY
            severity = AlertSeverity.CRITICAL
            message = f"CRITICAL: Driver eyes closed for {closure_duration_sec:.1f}s!"
            trigger_level = 3
            audio_text = self.level_3_msg

        elif prolonged_closure or (closure_duration_sec >= 0.6 and is_eyes_closed and drowsy_probability is not None and drowsy_probability >= 0.75):
            primary_state = DriverState.DROWSY
            severity = AlertSeverity.CRITICAL
            prob_text = f" ({int(drowsy_probability*100)}%)" if drowsy_probability else ""
            message = f"ALERT: Drowsiness detected{prob_text}. Involuntary eye closure."
            trigger_level = 2
            audio_text = self.level_2_msg

        elif is_distracted:
            primary_state = DriverState.DISTRACTED
            severity = AlertSeverity.WARNING
            message = f"WARNING: Driver head turned away for {distraction_duration_sec:.1f}s."
            trigger_level = 1
            audio_text = self.level_1_msg

        elif phone_detected:
            primary_state = DriverState.DISTRACTED
            severity = AlertSeverity.WARNING
            message = "WARNING: Mobile phone visible in driver area."
            trigger_level = 1
            audio_text = self.level_1_msg

        elif is_yawning:
            primary_state = DriverState.YAWNING
            severity = AlertSeverity.WARNING
            message = "Notice: Sustained yawning detected. Early fatigue indicator."
            trigger_level = 0  # Yawn logs visual warning without blaring level 1 siren unless sustained
            audio_text = ""

        elif is_eyes_closed:
            primary_state = DriverState.EYES_CLOSED
            severity = AlertSeverity.WARNING
            message = f"Eye closure detected ({closure_duration_sec:.1f}s)."
            trigger_level = 0
            audio_text = ""

        # Check audio alert cooldown and rules
        audio_payload: Optional[AudioAlertPayload] = None
        if self.audio_enabled and trigger_level > 0:
            time_since_last = current_time - self.last_alert_time
            # Critical Level 3 can bypass cooldown if previous was lower level
            can_fire = (time_since_last >= self.alert_cooldown_sec) or (trigger_level == 3 and self.last_played_level < 3)

            if can_fire:
                self.last_alert_time = current_time
                self.last_played_level = trigger_level
                audio_payload = AudioAlertPayload(
                    should_play=True,
                    level=trigger_level,
                    message=audio_text,
                    severity=severity
                )

        return UnifiedDriverAssessment(
            primary_state=primary_state,
            active_conditions=active_conditions,
            severity=severity,
            audio_alert=audio_payload,
            alert_message=message
        )
