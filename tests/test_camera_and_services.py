"""Automated tests for camera discovery, camera switching, event service, and settings."""
import os
import pytest
from backend.services.camera_manager import CameraManager
from backend.services.events import EventService
from backend.config import SystemSettings, save_settings, load_settings


def test_camera_discovery():
    mgr = CameraManager(default_index=0)
    cameras = mgr.list_available_cameras(max_probed=2)
    assert isinstance(cameras, list)
    assert len(cameras) >= 1
    assert "index" in cameras[0]
    assert "name" in cameras[0]


def test_camera_switching_and_stop():
    mgr = CameraManager(default_index=0)
    # Start and switch safely
    mgr.switch_camera(0)
    status = mgr.get_status()
    assert status["selected_index"] == 0

    # Stop and verify resource release
    mgr.stop()
    assert mgr.is_running is False
    assert mgr.is_connected is False


def test_event_service_and_csv_export():
    svc = EventService(max_stored_events=50)
    evt = svc.record_event(
        event_type="DROWSY",
        severity="CRITICAL",
        duration_sec=2.2,
        confidence=0.96,
        description="Test prolonged eye closure",
        vehicle_id="UNIT-TEST-1"
    )

    assert evt.event_type == "DROWSY"
    assert evt.severity == "CRITICAL"

    # Filter test
    crit_events = svc.get_events(severity="CRITICAL")
    assert any(e["id"] == evt.id for e in crit_events)

    # CSV export test
    csv_str = svc.export_csv()
    assert "Event ID,Timestamp,Vehicle ID" in csv_str
    assert "UNIT-TEST-1" in csv_str


def test_settings_validation_and_persistence():
    settings = load_settings()
    assert settings.detection.ear_threshold > 0.0

    # Mutate and save
    settings.detection.ear_threshold = 0.225
    success = save_settings(settings)
    assert success is True

    reloaded = load_settings()
    assert reloaded.detection.ear_threshold == 0.225
