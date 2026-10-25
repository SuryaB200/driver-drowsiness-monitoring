"""Mouth Aspect Ratio (MAR) and sustained yawn detection."""
import time
import numpy as np
from typing import Optional, NamedTuple
from backend.detection.eye_tracker import euclidean_distance

# Mouth landmarks in MediaPipe Face Mesh
# Outer/Inner lip landmarks
MOUTH_INDICES = {
    'left_corner': 61,
    'right_corner': 291,
    'top_lip': 13,       # upper inner lip
    'bottom_lip': 14,    # lower inner lip
    'top_lip_l': 81,     # upper lip mid-left
    'bottom_lip_l': 178, # lower lip mid-left
    'top_lip_r': 311,    # upper lip mid-right
    'bottom_lip_r': 402  # lower lip mid-right
}


def calculate_mar(landmarks_pixel: np.ndarray) -> float:
    """Calculates MAR using multi-point vertical opening / horizontal corner distance."""
    if landmarks_pixel is None or len(landmarks_pixel) <= max(MOUTH_INDICES.values()):
        return 0.0

    p_left = landmarks_pixel[MOUTH_INDICES['left_corner']]
    p_right = landmarks_pixel[MOUTH_INDICES['right_corner']]
    p_top = landmarks_pixel[MOUTH_INDICES['top_lip']]
    p_bottom = landmarks_pixel[MOUTH_INDICES['bottom_lip']]
    p_top_l = landmarks_pixel[MOUTH_INDICES['top_lip_l']]
    p_bot_l = landmarks_pixel[MOUTH_INDICES['bottom_lip_l']]
    p_top_r = landmarks_pixel[MOUTH_INDICES['top_lip_r']]
    p_bot_r = landmarks_pixel[MOUTH_INDICES['bottom_lip_r']]

    horiz = euclidean_distance(p_left, p_right)
    if horiz <= 1e-6:
        return 0.0

    v_center = euclidean_distance(p_top, p_bottom)
    v_left = euclidean_distance(p_top_l, p_bot_l)
    v_right = euclidean_distance(p_top_r, p_bot_r)

    # 3-vertical lines over 2 * horizontal width
    mar = (v_center + v_left + v_right) / (2.0 * horiz)
    return float(np.clip(mar, 0.0, 2.0))


class YawnResult(NamedTuple):
    mar: float
    smoothed_mar: float
    is_mouth_open: bool
    yawn_duration_sec: float
    is_yawning: bool


class YawnDetector:
    def __init__(self,
                 mar_threshold: float = 0.65,
                 yawn_duration_sec: float = 1.8,
                 smoothing_alpha: float = 0.35):
        self.mar_threshold = mar_threshold
        self.yawn_duration_sec = yawn_duration_sec
        self.smoothing_alpha = smoothing_alpha

        self.smoothed_mar = 0.20
        self.open_start_time: Optional[float] = None
        self.current_duration_sec = 0.0

    def update_thresholds(self, mar_threshold: float, yawn_duration_sec: float):
        self.mar_threshold = mar_threshold
        self.yawn_duration_sec = yawn_duration_sec

    def process(self, landmarks_pixel: Optional[np.ndarray], current_time: Optional[float] = None) -> YawnResult:
        if current_time is None:
            current_time = time.time()

        if landmarks_pixel is None or len(landmarks_pixel) < 468:
            self.open_start_time = None
            self.current_duration_sec = 0.0
            return YawnResult(0.0, self.smoothed_mar, False, 0.0, False)

        mar = calculate_mar(landmarks_pixel)

        # Smooth MAR to avoid speech flutter
        self.smoothed_mar = (self.smoothing_alpha * mar) + ((1.0 - self.smoothing_alpha) * self.smoothed_mar)

        is_open = self.smoothed_mar >= self.mar_threshold

        if is_open:
            if self.open_start_time is None:
                self.open_start_time = current_time
            self.current_duration_sec = current_time - self.open_start_time
        else:
            self.open_start_time = None
            self.current_duration_sec = 0.0

        # Yawning requires sustained opening longer than standard speech vowels
        is_yawning = self.current_duration_sec >= self.yawn_duration_sec

        return YawnResult(
            mar=round(mar, 3),
            smoothed_mar=round(self.smoothed_mar, 3),
            is_mouth_open=is_open,
            yawn_duration_sec=round(self.current_duration_sec, 2),
            is_yawning=is_yawning
        )

    def reset(self):
        self.open_start_time = None
        self.current_duration_sec = 0.0
        self.smoothed_mar = 0.20
