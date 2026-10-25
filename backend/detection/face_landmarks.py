"""Face landmark detection wrapper supporting MediaPipe Tasks (1.x+), legacy MediaPipe Solutions, and OpenCV fallback."""
import os
import cv2
import urllib.request
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, NamedTuple, List

class FaceLandmarksResult(NamedTuple):
    face_detected: bool
    landmarks_pixel: Optional[np.ndarray]   # (468/478, 2) in pixel coordinates (x, y)
    landmarks_norm: Optional[np.ndarray]    # (468/478, 3) normalized (x, y, z)
    bounding_box: Optional[Tuple[int, int, int, int]] # (x, y, width, height)
    bounding_box_norm: Optional[Tuple[float, float, float, float]] = None # (x_norm, y_norm, w_norm, h_norm)


class FaceLandmarksTracker:
    def __init__(self,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.35,
                 min_tracking_confidence: float = 0.35,
                 model_path: Optional[str] = None):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Locate or configure face_landmarker.task model
        if model_path:
            self.model_path = Path(model_path)
        else:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.model_path = base_dir / "checkpoints" / "face_landmarker.task"

        self.landmarker = None        # MediaPipe Tasks (1.x+)
        self.legacy_face_mesh = None  # MediaPipe Solutions (0.10.x)
        self.haar_cascade = None      # OpenCV fallback detector
        self.backend_type = "none"

        self._init_detector()

    def _ensure_task_model(self) -> bool:
        """Ensures the face_landmarker.task file exists, auto-downloading if necessary."""
        if self.model_path.exists() and self.model_path.stat().st_size > 1000000:
            return True
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            print(f"Downloading face_landmarker model to {self.model_path}...")
            urllib.request.urlretrieve(url, str(self.model_path))
            return self.model_path.exists() and self.model_path.stat().st_size > 1000000
        except Exception as e:
            print(f"Warning: Could not download face_landmarker model: {e}")
            return False

    def _init_detector(self):
        # 1. Try MediaPipe Tasks API (MediaPipe 1.x+)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if self._ensure_task_model():
                base_options = python.BaseOptions(model_asset_path=str(self.model_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=self.max_num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_face_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.backend_type = "mediapipe_tasks"
                print("FaceLandmarksTracker: Initialized MediaPipe Tasks FaceLandmarker successfully.")
        except Exception as e:
            print(f"MediaPipe Tasks initialization failed or skipped: {e}")
            self.landmarker = None

        # 2. Try legacy MediaPipe Solutions API if Tasks not active
        if self.landmarker is None:
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                    self.legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=self.max_num_faces,
                        refine_landmarks=self.refine_landmarks,
                        min_detection_confidence=self.min_detection_confidence,
                        min_tracking_confidence=self.min_tracking_confidence
                    )
                    self.backend_type = "mediapipe_legacy"
                    print("FaceLandmarksTracker: Initialized legacy MediaPipe FaceMesh successfully.")
            except Exception as e:
                print(f"Legacy MediaPipe FaceMesh initialization failed: {e}")
                self.legacy_face_mesh = None

        # 3. Always initialize OpenCV Haar Cascade as emergency fallback
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"OpenCV Haar Cascade fallback initialization failed: {e}")
            self.haar_cascade = None

    def _detect_tasks(self, frame_rgb: np.ndarray, w: int, h: int) -> Optional[FaceLandmarksResult]:
        """Inference with MediaPipe Tasks FaceLandmarker."""
        if self.landmarker is None:
            return None

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]
        num_pts = len(face_landmarks)

        landmarks_pixel = np.zeros((num_pts, 2), dtype=np.float32)
        landmarks_norm = np.zeros((num_pts, 3), dtype=np.float32)

        for i, lm in enumerate(face_landmarks):
            landmarks_pixel[i] = [lm.x * w, lm.y * h]
            landmarks_norm[i] = [lm.x, lm.y, lm.z]

        min_x = max(0, int(np.min(landmarks_pixel[:, 0])))
        min_y = max(0, int(np.min(landmarks_pixel[:, 1])))
        max_x = min(w, int(np.max(landmarks_pixel[:, 0])))
        max_y = min(h, int(np.max(landmarks_pixel[:, 1])))
        bw = max(0, max_x - min_x)
        bh = max(0, max_y - min_y)
        bbox = (min_x, min_y, bw, bh)
        bbox_norm = (min_x / w, min_y / h, bw / w, bh / h)

        return FaceLandmarksResult(
            face_detected=True,
            landmarks_pixel=landmarks_pixel,
            landmarks_norm=landmarks_norm,
            bounding_box=bbox,
            bounding_box_norm=bbox_norm
        )

    def _detect_legacy(self, frame_rgb: np.ndarray, w: int, h: int) -> Optional[FaceLandmarksResult]:
        """Inference with legacy MediaPipe FaceMesh."""
        if self.legacy_face_mesh is None:
            return None

        results = self.legacy_face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        num_pts = len(face_landmarks.landmark)

        landmarks_pixel = np.zeros((num_pts, 2), dtype=np.float32)
        landmarks_norm = np.zeros((num_pts, 3), dtype=np.float32)

        for i, lm in enumerate(face_landmarks.landmark):
            landmarks_pixel[i] = [lm.x * w, lm.y * h]
            landmarks_norm[i] = [lm.x, lm.y, lm.z]

        min_x = max(0, int(np.min(landmarks_pixel[:, 0])))
        min_y = max(0, int(np.min(landmarks_pixel[:, 1])))
        max_x = min(w, int(np.max(landmarks_pixel[:, 0])))
        max_y = min(h, int(np.max(landmarks_pixel[:, 1])))
        bw = max(0, max_x - min_x)
        bh = max(0, max_y - min_y)
        bbox = (min_x, min_y, bw, bh)
        bbox_norm = (min_x / w, min_y / h, bw / w, bh / h)

        return FaceLandmarksResult(
            face_detected=True,
            landmarks_pixel=landmarks_pixel,
            landmarks_norm=landmarks_norm,
            bounding_box=bbox,
            bounding_box_norm=bbox_norm
        )

    def _detect_haar(self, frame_bgr: np.ndarray, w: int, h: int) -> Optional[FaceLandmarksResult]:
        """OpenCV Haar Cascade fallback for face detection."""
        if self.haar_cascade is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) == 0:
            return None

        # Pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        bx, by, bw, bh = faces[0]
        bbox = (int(bx), int(by), int(bw), int(bh))
        bbox_norm = (bx / w, by / h, bw / w, bh / h)

        return FaceLandmarksResult(
            face_detected=True,
            landmarks_pixel=None,
            landmarks_norm=None,
            bounding_box=bbox,
            bounding_box_norm=bbox_norm
        )

    def process(self, frame_bgr: np.ndarray) -> FaceLandmarksResult:
        """Processes a single BGR frame and extracts facial landmarks and bounding box."""
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceLandmarksResult(False, None, None, None, None)

        if self.landmarker is None and self.legacy_face_mesh is None:
            self._init_detector()

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Primary detection with MediaPipe Tasks
        res = self._detect_tasks(frame_rgb, w, h)
        if res and res.face_detected:
            return res

        # 2. Try with low-light CLAHE enhancement if image is dark
        mean_brightness = float(np.mean(frame_bgr))
        if mean_brightness < 60.0:
            try:
                lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l_channel)
                enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                res = self._detect_tasks(enhanced_rgb, w, h)
                if res and res.face_detected:
                    return res
            except Exception:
                pass

        # 3. Fallback to legacy MediaPipe if configured
        res = self._detect_legacy(frame_rgb, w, h)
        if res and res.face_detected:
            return res

        # 4. Fallback to OpenCV Haar Cascade
        res = self._detect_haar(frame_bgr, w, h)
        if res and res.face_detected:
            return res

        return FaceLandmarksResult(False, None, None, None, None)

    def close(self):
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

        if self.legacy_face_mesh:
            try:
                self.legacy_face_mesh.close()
            except Exception:
                pass
            self.legacy_face_mesh = None
