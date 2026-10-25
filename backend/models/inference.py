"""Inference engine for MobileNetV3 drowsiness classification."""
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, NamedTuple
from backend.models.mobilenet import load_checkpoint, DrowsinessMobileNetV3

class DrowsinessInferenceResult(NamedTuple):
    is_available: bool
    predicted_label: str
    class_id: int
    drowsy_probability: float
    alert_probability: float
    latency_ms: float
    status_message: str


class MobileNetInferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.checkpoint_path = checkpoint_path
        self.device = self._resolve_device(device)
        self.model: Optional[DrowsinessMobileNetV3] = None
        self.meta: Dict[str, Any] = {}
        self.transform = None
        self._init_transforms()
        self.reload_checkpoint(checkpoint_path)

    def _resolve_device(self, pref: str) -> str:
        try:
            import torch
            if pref == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif pref == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            return "cpu"
        except Exception:
            return "cpu"

    def _init_transforms(self):
        try:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            self.transform = None

    def reload_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Reloads checkpoint from specified or configured path."""
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path

        try:
            import torch
            model, meta = load_checkpoint(self.checkpoint_path, device=self.device)
            if model is not None:
                self.model = model
                self.meta = meta
                return True
            else:
                self.model = None
                self.meta = meta
                return False
        except Exception as e:
            self.model = None
            self.meta = {'error': str(e)}
            return False

    def predict(self, face_bgr_crop: Optional[np.ndarray]) -> DrowsinessInferenceResult:
        """Runs MobileNetV3 inference on a cropped face image."""
        if self.model is None:
            err = self.meta.get('error', 'No checkpoint loaded. Train model or supply weights.')
            return DrowsinessInferenceResult(
                is_available=False,
                predicted_label="UNAVAILABLE",
                class_id=-1,
                drowsy_probability=0.0,
                alert_probability=0.0,
                latency_ms=0.0,
                status_message=f"Model Unavailable ({err})"
            )

        if face_bgr_crop is None or face_bgr_crop.size == 0:
            return DrowsinessInferenceResult(
                is_available=True,
                predicted_label="NO_FACE",
                class_id=-1,
                drowsy_probability=0.0,
                alert_probability=0.0,
                latency_ms=0.0,
                status_message="No face crop supplied for classification"
            )

        try:
            import torch
            import torch.nn.functional as F

            start_t = time.perf_counter()
            face_rgb = cv2.cvtColor(face_bgr_crop, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
            else:
                # Fallback manual transform
                resized = cv2.resize(face_rgb, (224, 224)).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                normalized = (resized - mean) / std
                tensor = torch.from_numpy(normalized.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            latency_ms = (time.perf_counter() - start_t) * 1000.0

            alert_prob = float(probs[0])
            drowsy_prob = float(probs[1]) if len(probs) > 1 else (1.0 - alert_prob)
            pred_id = int(np.argmax(probs))
            class_names = self.meta.get('class_names', ["Alert", "Drowsy"])
            label = class_names[pred_id] if pred_id < len(class_names) else str(pred_id)

            return DrowsinessInferenceResult(
                is_available=True,
                predicted_label=label,
                class_id=pred_id,
                drowsy_probability=round(drowsy_prob, 3),
                alert_probability=round(alert_prob, 3),
                latency_ms=round(latency_ms, 2),
                status_message="OK"
            )

        except Exception as e:
            return DrowsinessInferenceResult(
                is_available=False,
                predicted_label="ERROR",
                class_id=-1,
                drowsy_probability=0.0,
                alert_probability=0.0,
                latency_ms=0.0,
                status_message=f"Inference execution failed: {e}"
            )
