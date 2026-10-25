"""Automated tests for alert priority hierarchy, audio triggers, and cooldown."""
import pytest
import time
from backend.detection.alert_engine import AlertEngine, AlertSeverity, DriverState

def test_alert_hierarchy_normal():
    engine = AlertEngine()
    res = engine.evaluate(
        camera_available=True,
        face_detected=True,
        is_eyes_closed=False,
        closure_duration_sec=0.0,
        prolonged_closure=False,
        is_yawning=False,
        is_distracted=False,
        distraction_duration_sec=0.0,
        drowsy_probability=0.10,
        phone_detected=False
    )
    assert res.primary_state == DriverState.ALERT
    assert res.severity == AlertSeverity.NORMAL
    assert res.audio_alert is None


def test_alert_prolonged_closure_level2():
    engine = AlertEngine()
    now = time.time()
    res = engine.evaluate(
        camera_available=True,
        face_detected=True,
        is_eyes_closed=True,
        closure_duration_sec=1.4,
        prolonged_closure=True,
        is_yawning=False,
        is_distracted=False,
        distraction_duration_sec=0.0,
        drowsy_probability=0.85,
        phone_detected=False,
        current_time=now
    )
    assert res.primary_state == DriverState.DROWSY
    assert res.severity == AlertSeverity.CRITICAL
    assert res.audio_alert is not None
    assert res.audio_alert.level == 2
    assert "Drowsiness detected" in res.audio_alert.message


def test_critical_eye_closure_level3():
    engine = AlertEngine()
    now = time.time()
    res = engine.evaluate(
        camera_available=True,
        face_detected=True,
        is_eyes_closed=True,
        closure_duration_sec=2.8, # exceeds 2.5s critical threshold
        prolonged_closure=True,
        is_yawning=False,
        is_distracted=False,
        distraction_duration_sec=0.0,
        drowsy_probability=0.95,
        phone_detected=False,
        current_time=now
    )
    assert res.primary_state == DriverState.DROWSY
    assert res.severity == AlertSeverity.CRITICAL
    assert res.audio_alert is not None
    assert res.audio_alert.level == 3
    assert "Critical warning" in res.audio_alert.message


def test_alert_cooldown():
    engine = AlertEngine(alert_cooldown_sec=3.0)
    t0 = 1000.0

    # First alert fires
    r1 = engine.evaluate(
        camera_available=True, face_detected=True, is_eyes_closed=False,
        closure_duration_sec=0.0, prolonged_closure=False, is_yawning=False,
        is_distracted=True, distraction_duration_sec=2.5,
        drowsy_probability=0.2, phone_detected=False, current_time=t0
    )
    assert r1.audio_alert is not None
    assert r1.audio_alert.level == 1

    # Second call 1 second later (within 3.0s cooldown) should NOT fire audio
    r2 = engine.evaluate(
        camera_available=True, face_detected=True, is_eyes_closed=False,
        closure_duration_sec=0.0, prolonged_closure=False, is_yawning=False,
        is_distracted=True, distraction_duration_sec=3.5,
        drowsy_probability=0.2, phone_detected=False, current_time=t0 + 1.0
    )
    assert r2.audio_alert is None

    # Third call after cooldown (t0 + 3.5s) can fire audio again
    r3 = engine.evaluate(
        camera_available=True, face_detected=True, is_eyes_closed=False,
        closure_duration_sec=0.0, prolonged_closure=False, is_yawning=False,
        is_distracted=True, distraction_duration_sec=6.0,
        drowsy_probability=0.2, phone_detected=False, current_time=t0 + 3.5
    )
    assert r3.audio_alert is not None


def test_alert_open_eyes_not_drowsy():
    engine = AlertEngine()
    # Driver eyes are wide open, but ML model output has high drowsy prob
    res = engine.evaluate(
        camera_available=True,
        face_detected=True,
        is_eyes_closed=False,
        closure_duration_sec=0.0,
        prolonged_closure=False,
        is_yawning=False,
        is_distracted=False,
        distraction_duration_sec=0.0,
        drowsy_probability=0.92,  # high model probability on ambiguous face crop
        phone_detected=False
    )
    # Physical open eyes must NOT trigger false DROWSY alarms
    assert res.primary_state == DriverState.ALERT
    assert res.severity == AlertSeverity.NORMAL
    assert res.audio_alert is None

