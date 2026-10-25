"""Eye Aspect Ratio (EAR) calculation and eye closure duration tracking."""
import time
import numpy as np
from typing import Optional, Tuple, NamedTuple

# Landmark indices for MediaPipe Face Mesh
# Left eye
LEFT_EYE_INDICES = {
    'p1': 33,   # outer corner
    'p2': 160,  # top outer
    'p3': 158,  # top inner
    'p4': 133,  # inner corner
    'p5': 153,  # bottom inner
    'p6': 144   # bottom outer
}

# Right eye
RIGHT_EYE_INDICES = {
    'p1': 362,  # inner corner
    'p2': 385,  # top inner
    'p3': 387,  # top outer
    'p4': 263,  # outer corner
    'p5': 373,  # bottom outer
    'p6': 380   # bottom inner
}


def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Compute Euclidean distance between two 2D points."""
    return float(np.linalg.norm(pt1 - pt2))


def calculate_single_ear(landmarks_pixel: np.ndarray, indices: dict) -> float:
    """Calculates EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)."""
    if landmarks_pixel is None or len(landmarks_pixel) <= max(indices.values()):
        return 0.0

    p1 = landmarks_pixel[indices['p1']]
    p2 = landmarks_pixel[indices['p2']]
    p3 = landmarks_pixel[indices['p3']]
    p4 = landmarks_pixel[indices['p4']]
    p5 = landmarks_pixel[indices['p5']]
    p6 = landmarks_pixel[indices['p6']]

    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    horiz = euclidean_distance(p1, p4)

    if horiz <= 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * horiz)
    return float(np.clip(ear, 0.0, 1.0))


class EyeTrackingResult(NamedTuple):
    left_ear: float
    right_ear: float
    avg_ear: float
    smoothed_ear: float
    is_eyes_closed: bool
    closure_duration_sec: float
    prolonged_closure_detected: bool


class EyeTracker:
    def __init__(self,
                 ear_threshold: float = 0.21,
                 closure_threshold_sec: float = 1.2,
                 smoothing_alpha: float = 0.35):
        self.ear_threshold = ear_threshold
        self.closure_threshold_sec = closure_threshold_sec
        self.smoothing_alpha = smoothing_alpha

        self.smoothed_ear = 0.30
        self.closure_start_time: Optional[float] = None
        self.closure_duration_sec = 0.0
        self.last_update_time: Optional[float] = None

        # Calibration support
        self.calibration_samples = []
        self.is_calibrating = False

    def update_thresholds(self, ear_threshold: float, closure_threshold_sec: float, smoothing_alpha: Optional[float] = None):
        self.ear_threshold = ear_threshold
        self.closure_threshold_sec = closure_threshold_sec
        if smoothing_alpha is not None:
            self.smoothing_alpha = smoothing_alpha

    def start_calibration(self):
        self.calibration_samples.clear()
        self.is_calibrating = True

    def finish_calibration(self) -> float:
        self.is_calibrating = False
        if len(self.calibration_samples) >= 10:
            baseline_open_ear = float(np.median(self.calibration_samples))
            # Set personalized threshold as ~70% of baseline open EAR
            self.ear_threshold = max(0.16, round(baseline_open_ear * 0.70, 3))
        return self.ear_threshold

    def process(self, landmarks_pixel: Optional[np.ndarray], current_time: Optional[float] = None) -> EyeTrackingResult:
        if current_time is None:
            current_time = time.time()

        if landmarks_pixel is None or len(landmarks_pixel) < 468:
            # Face lost or invalid: reset closure timer to prevent false prolonged events during absence
            self.closure_start_time = None
            self.closure_duration_sec = 0.0
            return EyeTrackingResult(
                left_ear=0.0,
                right_ear=0.0,
                avg_ear=0.0,
                smoothed_ear=self.smoothed_ear,
                is_eyes_closed=False,
                closure_duration_sec=0.0,
                prolonged_closure_detected=False
            )

        left_ear = calculate_single_ear(landmarks_pixel, LEFT_EYE_INDICES)
        right_ear = calculate_single_ear(landmarks_pixel, RIGHT_EYE_INDICES)

        # Compute average
        if left_ear > 0 and right_ear > 0:
            avg_ear = (left_ear + right_ear) / 2.0
        else:
            avg_ear = left_ear if left_ear > 0 else right_ear

        # Exponential moving average smoothing
        self.smoothed_ear = (self.smoothing_alpha * avg_ear) + ((1.0 - self.smoothing_alpha) * self.smoothed_ear)

        if self.is_calibrating and avg_ear > 0:
            self.calibration_samples.append(avg_ear)

        # Check eye closure against threshold
        is_closed = self.smoothed_ear < self.ear_threshold

        if is_closed:
            if self.closure_start_time is None:
                self.closure_start_time = current_time
            self.closure_duration_sec = current_time - self.closure_start_time
        else:
            self.closure_start_time = None
            self.closure_duration_sec = 0.0

        prolonged = self.closure_duration_sec >= self.closure_threshold_sec
        self.last_update_time = current_time

        return EyeTrackingResult(
            left_ear=round(left_ear, 3),
            right_ear=round(right_ear, 3),
            avg_ear=round(avg_ear, 3),
            smoothed_ear=round(self.smoothed_ear, 3),
            is_eyes_closed=is_closed,
            closure_duration_sec=round(self.closure_duration_sec, 2),
            prolonged_closure_detected=prolonged
        )

    def reset(self):
        """Reset internal state."""
        self.closure_start_time = None
        self.closure_duration_sec = 0.0
        self.smoothed_ear = 0.30
