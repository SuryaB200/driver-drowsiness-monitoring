"""Camera hardware discovery, capture, switching, and resource management."""
import cv2
import time
import threading
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

class CameraDeviceInfo:
    def __init__(self, index: int, name: str, is_active: bool = False, max_res: str = "Unknown"):
        self.index = index
        self.name = name
        self.is_active = is_active
        self.max_res = max_res

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "is_active": self.is_active,
            "max_res": self.max_res
        }


class CameraManager:
    def __init__(self,
                 default_index: int = 0,
                 target_width: int = 1280,
                 target_height: int = 720,
                 target_fps: int = 30,
                 auto_reconnect: bool = True):
        self.current_index = default_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.auto_reconnect = auto_reconnect

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_connected = False
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_time: float = 0.0
        self.fps_measured: float = 0.0

        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None
        self.device_name: str = f"Camera Device #{default_index}"

    def list_available_cameras(self, max_probed: int = 5) -> List[Dict[str, Any]]:
        """
        Discovers connected cameras on Windows using DirectShow (CAP_DSHOW)
        without freezing or blocking.
        """
        cameras = []
        for idx in range(max_probed):
            try:
                # Use CAP_DSHOW on Windows for fast device querying
                temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret and frame is not None:
                        w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        name = f"Camera {idx}" + (" (Integrated/Default)" if idx == 0 else f" (External USB #{idx})")
                        is_current = (self.is_running and self.current_index == idx)
                        cameras.append(CameraDeviceInfo(idx, name, is_current, f"{w}x{h}").to_dict())
                    temp_cap.release()
            except Exception as e:
                # Ignore failed indices
                pass

        if not cameras:
            # Fallback entry if no hardware camera was found
            cameras.append({
                "index": 0,
                "name": "Default Video Source (Virtual/Webcam 0)",
                "is_active": self.is_running and self.current_index == 0,
                "max_res": f"{self.target_width}x{self.target_height}"
            })

        return cameras

    def start(self, index: Optional[int] = None) -> bool:
        """Starts capture thread for the given or current camera index."""
        with self.lock:
            if index is not None:
                self.current_index = index

            if self.is_running and self.cap is not None and self.cap.isOpened():
                return True

            self._open_capture()
            if self.cap and self.cap.isOpened():
                self.is_running = True
                self.is_connected = True
                self.last_error = None
                self.thread = threading.Thread(target=self._capture_worker, daemon=True)
                self.thread.start()
                return True
            else:
                self.is_running = False
                self.is_connected = False
                self.last_error = f"Failed to open camera index {self.current_index}"
                return False

    def stop(self) -> None:
        """Stops capture thread and releases video capture handle."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
            self.thread = None

        with self.lock:
            self._release_capture()
            self.is_connected = False
            self.latest_frame = None

    def switch_camera(self, new_index: int) -> bool:
        """Safely releases current camera, resets buffers, and opens new camera device."""
        if self.current_index == new_index and self.is_running and self.is_connected:
            return True

        self.stop()
        time.sleep(0.15)  # Short pause to ensure OS driver releases handle
        self.current_index = new_index
        self.device_name = f"Camera #{new_index}"
        return self.start(new_index)

    def _open_capture(self):
        self._release_capture()
        try:
            # Attempt opening with DirectShow backend for Windows
            cap = cv2.VideoCapture(self.current_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback to standard backend
                cap = cv2.VideoCapture(self.current_index)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.cap = cap
            else:
                self.cap = None
        except Exception as e:
            self.last_error = str(e)
            self.cap = None

    def _release_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _capture_worker(self):
        """Dedicated background thread capturing frames at target frame rate."""
        frame_count = 0
        fps_start = time.time()

        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                if self.auto_reconnect:
                    time.sleep(1.0)
                    with self.lock:
                        self._open_capture()
                    continue
                else:
                    break

            ret, frame = self.cap.read()
            now = time.time()

            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.latest_frame_time = now
                    self.is_connected = True

                frame_count += 1
                elapsed = now - fps_start
                if elapsed >= 1.0:
                    self.fps_measured = round(frame_count / elapsed, 1)
                    frame_count = 0
                    fps_start = now

                # Throttle to approximately target FPS
                sleep_time = max(0.001, (1.0 / self.target_fps) - 0.005)
                time.sleep(sleep_time)
            else:
                with self.lock:
                    self.is_connected = False
                if self.auto_reconnect:
                    time.sleep(0.5)
                    with self.lock:
                        self._open_capture()
                else:
                    break

        self.is_connected = False

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Thread-safe retrieval of latest frame."""
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy(), self.latest_frame_time
            return False, None, 0.0

    def set_external_frame(self, frame_bgr: np.ndarray):
        """Allows injecting frame directly (e.g. from browser WebSocket stream)."""
        with self.lock:
            self.latest_frame = frame_bgr
            self.latest_frame_time = time.time()
            self.is_connected = True

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "selected_index": self.current_index,
            "device_name": self.device_name,
            "target_resolution": f"{self.target_width}x{self.target_height}",
            "measured_fps": self.fps_measured,
            "last_error": self.last_error
        }
