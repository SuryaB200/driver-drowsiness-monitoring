"""FastAPI REST routes for model, dataset, cameras, events, and settings."""
import os
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, Query
from pydantic import BaseModel

from backend.config import SystemSettings, save_settings, CHECKPOINTS_DIR, RESULTS_DIR, DATASETS_DIR
from backend.services.camera_manager import CameraManager
from backend.services.events import EventService
from backend.services.monitoring import DriverMonitoringService

router = APIRouter(prefix="/api")

# Reference to singleton services initialized in main.py
camera_mgr: Optional[CameraManager] = None
event_svc: Optional[EventService] = None
monitoring_svc: Optional[DriverMonitoringService] = None

# Global training background state
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 20,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "status": "Idle",
    "logs": []
}


def init_routes(c_mgr: CameraManager, e_svc: EventService, m_svc: DriverMonitoringService):
    global camera_mgr, event_svc, monitoring_svc
    camera_mgr = c_mgr
    event_svc = e_svc
    monitoring_svc = m_svc


# ------------------ Health & Status ------------------
@router.get("/health")
def get_health():
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU)"
    except Exception:
        cuda_avail = False
        gpu_name = "N/A"

    return {
        "status": "healthy",
        "service": "Commercial Fleet Driver Drowsiness Monitoring",
        "version": "1.0.0",
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "active_camera_index": camera_mgr.current_index if camera_mgr else None,
        "camera_connected": camera_mgr.is_connected if camera_mgr else False
    }


# ------------------ Camera Management ------------------
class CameraSelectRequest(BaseModel):
    index: int


@router.get("/cameras")
def list_cameras():
    if not camera_mgr:
        raise HTTPException(status_code=500, detail="Camera service uninitialized")
    return {
        "cameras": camera_mgr.list_available_cameras(),
        "selected_index": camera_mgr.current_index,
        "is_running": camera_mgr.is_running
    }


@router.get("/cameras/status")
def get_camera_status():
    if not camera_mgr:
        raise HTTPException(status_code=500, detail="Camera service uninitialized")
    return camera_mgr.get_status()


@router.post("/cameras/select")
def select_camera(req: CameraSelectRequest):
    if not camera_mgr:
        raise HTTPException(status_code=500, detail="Camera service uninitialized")
    success = camera_mgr.switch_camera(req.index)
    return {
        "success": success,
        "selected_index": req.index,
        "status": camera_mgr.get_status()
    }


@router.post("/cameras/start")
def start_camera():
    if not camera_mgr:
        raise HTTPException(status_code=500, detail="Camera service uninitialized")
    success = camera_mgr.start()
    return {"success": success, "status": camera_mgr.get_status()}


@router.post("/cameras/stop")
def stop_camera():
    if not camera_mgr:
        raise HTTPException(status_code=500, detail="Camera service uninitialized")
    camera_mgr.stop()
    return {"success": True, "status": camera_mgr.get_status()}


# ------------------ Safety Events ------------------
@router.get("/events")
def get_events(severity: Optional[str] = "ALL",
               event_type: Optional[str] = "ALL",
               search: Optional[str] = None,
               limit: int = 100):
    if not event_svc:
        raise HTTPException(status_code=500, detail="Event service uninitialized")
    return {
        "events": event_svc.get_events(severity=severity, event_type=event_type, search=search, limit=limit),
        "kpi_summary": event_svc.get_today_summary()
    }


@router.get("/events/export")
def export_events_csv():
    if not event_svc:
        raise HTTPException(status_code=500, detail="Event service uninitialized")
    csv_data = event_svc.export_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=driver_safety_events.csv"}
    )


# ------------------ Settings ------------------
@router.get("/settings")
def get_settings():
    if not monitoring_svc:
        raise HTTPException(status_code=500, detail="Monitoring service uninitialized")
    return monitoring_svc.settings.model_dump()


@router.put("/settings")
def update_settings(new_settings: SystemSettings):
    if not monitoring_svc:
        raise HTTPException(status_code=500, detail="Monitoring service uninitialized")
    save_settings(new_settings)
    monitoring_svc.update_settings(new_settings)
    return {"success": True, "settings": new_settings.model_dump()}


# ------------------ Dataset Status ------------------
@router.get("/dataset/status")
def get_dataset_status():
    from training.prepare_dataset import check_dataset_status
    return check_dataset_status()


@router.post("/dataset/prepare")
def prepare_dataset(background_tasks: BackgroundTasks):
    from training.prepare_dataset import prepare_benchmark_dataset
    summary = prepare_benchmark_dataset()
    return {"success": True, "summary": summary}


# ------------------ Model Status, Training & Evaluation ------------------
@router.get("/model/status")
def get_model_status():
    if not monitoring_svc:
        raise HTTPException(status_code=500, detail="Monitoring service uninitialized")
    eng = monitoring_svc.mobilenet_engine
    ckpt_file = Path(eng.checkpoint_path)
    exists = ckpt_file.exists()

    return {
        "architecture": "MobileNetV3-Small",
        "checkpoint_path": eng.checkpoint_path,
        "checkpoint_exists": exists,
        "is_loaded": eng.model is not None,
        "metadata": eng.meta,
        "device": eng.device,
        "status_message": "Ready for live inference" if eng.model is not None else "No trained checkpoint loaded"
    }


class TrainRequest(BaseModel):
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.0001
    use_augmentation: bool = True


@router.post("/model/train")
def start_model_training(req: TrainRequest, background_tasks: BackgroundTasks):
    global training_state
    if training_state["is_training"]:
        return {"success": False, "message": "Training is already in progress"}

    def run_training_worker():
        from training.train import train_mobilenet_v3
        train_mobilenet_v3(
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.learning_rate,
            use_aug=req.use_augmentation,
            progress_callback=training_state
        )
        if monitoring_svc:
            monitoring_svc.mobilenet_engine.reload_checkpoint()

    thread = threading.Thread(target=run_training_worker, daemon=True)
    thread.start()

    return {"success": True, "message": f"Training started for {req.epochs} epochs"}


@router.get("/model/training-status")
def get_training_status():
    return training_state


@router.post("/model/evaluate")
def evaluate_model():
    from training.evaluate import run_model_evaluation
    results = run_model_evaluation()
    return results


@router.get("/model/metrics")
def get_model_metrics():
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"available": True, "metrics": data}
        except Exception as e:
            return {"available": False, "error": str(e)}
    return {"available": False, "message": "No evaluation metrics found. Please train and evaluate the model."}
