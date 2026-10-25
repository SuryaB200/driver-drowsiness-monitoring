"""Automated tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "Commercial Fleet Driver" in data["service"]


def test_model_status_endpoint():
    res = client.get("/api/model/status")
    assert res.status_code == 200
    data = res.json()
    assert data["architecture"] == "MobileNetV3-Small"
    assert "checkpoint_exists" in data


def test_cameras_endpoint():
    res = client.get("/api/cameras")
    assert res.status_code == 200
    data = res.json()
    assert "cameras" in data
    assert len(data["cameras"]) >= 1


def test_events_endpoint():
    res = client.get("/api/events")
    assert res.status_code == 200
    data = res.json()
    assert "events" in data
    assert "kpi_summary" in data


def test_settings_get_and_put():
    res = client.get("/api/settings")
    assert res.status_code == 200
    cfg = res.json()
    assert "detection" in cfg
    assert "audio" in cfg

    # Test update
    cfg["detection"]["ear_threshold"] = 0.215
    put_res = client.put("/api/settings", json=cfg)
    assert put_res.status_code == 200
    updated = put_res.json()
    assert updated["settings"]["detection"]["ear_threshold"] == 0.215
