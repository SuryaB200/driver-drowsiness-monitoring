"""Optional mobile phone presence detector for driver distraction monitoring."""
import cv2
import numpy as np
from typing import Optional, List, Tuple, NamedTuple

class PhoneDetectionResult(NamedTuple):
    available: bool
    phone_detected: bool
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] # (x, y, w, h)
    status_message: str


class PhoneDetector:
    def __init__(self, confidence_threshold: float = 0.45, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.is_loaded = False
        self.status = "Initializing..."
        self._load_detector()

    def _load_detector(self):
        """Attempts to load a lightweight COCO object detector (e.g., SSDLite MobileNetV3)."""
        try:
            import torch
            import torchvision
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = ssdlite320_mobilenet_v3_large(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            self.status = "Ready (SSDLite MobileNetV3)"
        except Exception as e:
            self.is_loaded = False
            self.model = None
            self.status = f"Phone Detection Unavailable ({type(e).__name__})"

    def process(self, frame_bgr: np.ndarray) -> PhoneDetectionResult:
        """Runs detection specifically searching for COCO class 'cell phone' (id 77)."""
        if not self.is_loaded or self.model is None or frame_bgr is None:
            return PhoneDetectionResult(
                available=False,
                phone_detected=False,
                confidence=0.0,
                bounding_box=None,
                status_message=self.status
            )

        try:
            import torch
            import torchvision.transforms.functional as F

            h, w, _ = frame_bgr.shape
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(tensor)[0]

            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()

            # COCO category 77 is 'cell phone'
            phone_indices = np.where((labels == 77) & (scores >= self.confidence_threshold))[0]

            if len(phone_indices) > 0:
                best_idx = phone_indices[np.argmax(scores[phone_indices])]
                best_score = float(scores[best_idx])
                box = boxes[best_idx]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                bbox = (max(0, x1), max(0, y1), max(0, x2 - x1), max(0, y2 - y1))

                return PhoneDetectionResult(
                    available=True,
                    phone_detected=True,
                    confidence=round(best_score, 3),
                    bounding_box=bbox,
                    status_message="Phone Detected (Visible Device)"
                )

            return PhoneDetectionResult(
                available=True,
                phone_detected=False,
                confidence=0.0,
                bounding_box=None,
                status_message="No Phone Detected"
            )

        except Exception as e:
            return PhoneDetectionResult(
                available=False,
                phone_detected=False,
                confidence=0.0,
                bounding_box=None,
                status_message=f"Detection error: {e}"
            )
