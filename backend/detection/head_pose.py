"""Head Pose Estimation using OpenCV solvePnP and MediaPipe facial keypoints."""
import cv2
import time
import numpy as np
from typing import Optional, Tuple, NamedTuple

# 3D Reference Model Points for a canonical human face (in millimeters)
# Standard anthropometric model points
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (index 1)
    [0.0, -330.0, -65.0],      # Chin (index 199 / 152)
    [-225.0, 170.0, -135.0],   # Left eye outer corner (index 33)
    [225.0, 170.0, -135.0],    # Right eye outer corner (index 263)
    [-150.0, -150.0, -125.0],  # Left mouth corner (index 61)
    [150.0, -150.0, -125.0]    # Right mouth corner (index 291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
LANDMARK_INDICES_FOR_POSE = [1, 152, 33, 263, 61, 291]


class HeadPoseResult(NamedTuple):
    yaw: float        # Left / Right rotation (- is left, + is right) in degrees
    pitch: float      # Down / Up tilt (- is down, + is up) in degrees
    roll: float       # Tilt sideways in degrees
    is_distracted: bool
    distraction_duration_sec: float
    direction: str    # "FORWARD", "LEFT", "RIGHT", "DOWN", "UP"


class HeadPoseEstimator:
    def __init__(self,
                 yaw_threshold: float = 28.0,
                 pitch_threshold: float = 22.0,
                 distraction_duration_sec: float = 2.0,
                 smoothing_alpha: float = 0.3):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.distraction_duration_sec = distraction_duration_sec
        self.smoothing_alpha = smoothing_alpha

        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_roll = 0.0

        self.distraction_start_time: Optional[float] = None
        self.current_duration_sec = 0.0

    def update_thresholds(self, yaw_threshold: float, pitch_threshold: float, distraction_duration_sec: float):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.distraction_duration_sec = distraction_duration_sec

    def process(self,
                landmarks_pixel: Optional[np.ndarray],
                frame_shape: Tuple[int, int],
                current_time: Optional[float] = None) -> HeadPoseResult:
        """
        Estimate head pose (yaw, pitch, roll) from 2D facial landmarks.
        frame_shape: (height, width)
        """
        if current_time is None:
            current_time = time.time()

        if landmarks_pixel is None or len(landmarks_pixel) < 468:
            self.distraction_start_time = None
            self.current_duration_sec = 0.0
            return HeadPoseResult(
                yaw=round(self.smoothed_yaw, 1),
                pitch=round(self.smoothed_pitch, 1),
                roll=round(self.smoothed_roll, 1),
                is_distracted=False,
                distraction_duration_sec=0.0,
                direction="UNKNOWN"
            )

        h, w = frame_shape
        # Extract corresponding 2D coordinates
        image_points = np.array([
            landmarks_pixel[idx] for idx in LANDMARK_INDICES_FOR_POSE
        ], dtype=np.float64)

        # Approximate camera intrinsic matrix (pinhole model)
        focal_length = w * 1.0
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPoseResult(
                yaw=round(self.smoothed_yaw, 1),
                pitch=round(self.smoothed_pitch, 1),
                roll=round(self.smoothed_roll, 1),
                is_distracted=False,
                distraction_duration_sec=0.0,
                direction="FORWARD"
            )

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose projection/rotation matrix to Euler angles
        # Using RQDecomp3x3 for robust Euler angle extraction
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Normalize Euler angles to [-90, 90] centered on forward-facing camera
        def normalize_euler(a: float) -> float:
            val = (a + 180.0) % 360.0 - 180.0
            if val > 90.0:
                val -= 180.0
            elif val < -90.0:
                val += 180.0
            return val

        # In camera coordinates relative to forward facing face:
        # pitch: negative is DOWN (towards lap/phone), positive is UP (towards ceiling)
        # yaw: negative is LEFT, positive is RIGHT
        # roll: sideways tilt
        raw_pitch = -normalize_euler(float(angles[0]))
        raw_yaw = -normalize_euler(float(angles[1]))
        raw_roll = normalize_euler(float(angles[2]))

        # Apply exponential moving average filter
        self.smoothed_pitch = (self.smoothing_alpha * raw_pitch) + ((1.0 - self.smoothing_alpha) * self.smoothed_pitch)
        self.smoothed_yaw = (self.smoothing_alpha * raw_yaw) + ((1.0 - self.smoothing_alpha) * self.smoothed_yaw)
        self.smoothed_roll = (self.smoothing_alpha * raw_roll) + ((1.0 - self.smoothing_alpha) * self.smoothed_roll)

        yaw_deg = self.smoothed_yaw
        pitch_deg = self.smoothed_pitch

        # Determine direction
        direction = "FORWARD"
        is_turning = False

        if yaw_deg > self.yaw_threshold:
            direction = "RIGHT"
            is_turning = True
        elif yaw_deg < -self.yaw_threshold:
            direction = "LEFT"
            is_turning = True
        elif pitch_deg < -self.pitch_threshold:
            direction = "DOWN"
            is_turning = True
        elif pitch_deg > self.pitch_threshold:
            direction = "UP"
            is_turning = True

        if is_turning:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            self.current_duration_sec = current_time - self.distraction_start_time
        else:
            self.distraction_start_time = None
            self.current_duration_sec = 0.0

        is_distracted = self.current_duration_sec >= self.distraction_duration_sec

        return HeadPoseResult(
            yaw=round(yaw_deg, 1),
            pitch=round(pitch_deg, 1),
            roll=round(self.smoothed_roll, 1),
            is_distracted=is_distracted,
            distraction_duration_sec=round(self.current_duration_sec, 2),
            direction=direction
        )

    def reset(self):
        self.distraction_start_time = None
        self.current_duration_sec = 0.0
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_roll = 0.0
