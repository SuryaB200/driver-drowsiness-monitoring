"""Commercial Fleet Driver Drowsiness and Distraction Monitoring - FastAPI Backend Entrypoint."""
import os
import cv2
import json
import base64
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import load_settings, PLOTS_DIR, RESULTS_DIR
from backend.services.camera_manager import CameraManager
from backend.services.events import EventService
from backend.services.monitoring import DriverMonitoringService
from backend.api.routes import router as api_router, init_routes

# Initialize global singletons
settings = load_settings()
camera_manager = CameraManager(
    default_index=settings.camera.selected_device_id,
    target_width=settings.camera.resolution_width,
    target_height=settings.camera.resolution_height,
    target_fps=settings.camera.target_fps,
    auto_reconnect=settings.camera.auto_reconnect
)
event_service = EventService()
monitoring_service = DriverMonitoringService(settings, camera_manager, event_service)

# Connect routes to singletons
init_routes(camera_manager, event_service, monitoring_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: do not auto-grab physical camera so browser/localhost has exclusive access
    print("Commercial Fleet Driver Monitoring Backend Initializing...")
    yield
    # Shutdown: safely release camera handles and threads
    print("Shutting down camera capture and releasing resources...")
    camera_manager.stop()
    if monitoring_service.face_tracker:
        monitoring_service.face_tracker.close()


app = FastAPI(
    title="Commercial Fleet Driver Drowsiness & Distraction Monitoring System",
    description="Edge Vision & Deep Learning API for Driver Monitoring using MobileNetV3 and MediaPipe",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup for Vite / React local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount static plots directory for viewing generated evaluation charts
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


# ------------------ WebSocket Real-Time Monitoring ------------------
@app.websocket("/ws/monitor")
async def websocket_monitor_endpoint(websocket: WebSocket):
    """
    Bi-directional real-time telemetry stream.
    Supports either:
    1. Server-side OpenCV camera capture (default), streaming telemetry to client.
    2. Client-side browser webcam stream: client sends base64/binary JPEG frames,
       backend runs AI inference on that exact frame and returns measurements.
    """
    await websocket.accept()
    client_streaming_active = False

    try:
        while True:
            # Check if client sent an incoming message or frame
            data = None
            try:
                # Non-blocking poll for incoming client message with a short timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.035)
            except asyncio.TimeoutError:
                data = None

            frame_to_process = None

            if data:
                try:
                    payload = json.loads(data)
                    msg_type = payload.get("type", "")

                    if msg_type == "FRAME":
                        # Client sent a video frame from browser getUserMedia
                        b64_img = payload.get("image", "")
                        if "," in b64_img:
                            b64_img = b64_img.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_img)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame_to_process = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        client_streaming_active = True
                        # Also feed camera manager so status matches
                        if frame_to_process is not None:
                            camera_manager.set_external_frame(frame_to_process)

                    elif msg_type == "CALIBRATE":
                        monitoring_service.eye_tracker.start_calibration()

                    elif msg_type == "FINISH_CALIBRATE":
                        new_th = monitoring_service.eye_tracker.finish_calibration()
                        await websocket.send_json({
                            "type": "CALIBRATION_RESULT",
                            "calibrated_ear_threshold": new_th
                        })

                except Exception as e:
                    # Ignore malformed client message
                    pass

            # If client is streaming frames from browser, wait for next frame if none arrived in this tick
            if client_streaming_active and frame_to_process is None:
                continue

            # Run full monitoring pipeline
            result = monitoring_service.process_frame(
                frame_bgr=frame_to_process,
                include_annotated_frame=False  # Keep payload lightweight; frontend can overlay on native canvas
            )

            # Send telemetry JSON to client
            await websocket.send_json(result)

            # Throttle slightly to avoid overloading connection
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket session terminated: {e}")
