"""Automated tests for EAR, MAR, Head Pose, and geometry math."""
import pytest
import numpy as np
import time

from backend.detection.eye_tracker import (
    calculate_single_ear,
    euclidean_distance,
    EyeTracker,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)
from backend.detection.yawn_detector import calculate_mar, YawnDetector
from backend.detection.head_pose import HeadPoseEstimator


def test_euclidean_distance():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])
    assert euclidean_distance(p1, p2) == 5.0


def test_ear_open_vs_closed():
    # 478 mock landmarks
    mock_lms = np.zeros((478, 2), dtype=np.float32)

    # Setup wide open left eye
    mock_lms[33] = [100.0, 200.0]   # p1 (outer)
    mock_lms[133] = [140.0, 200.0]  # p4 (inner, width = 40)
    mock_lms[160] = [115.0, 192.0]  # p2 (top outer)
    mock_lms[144] = [115.0, 208.0]  # p6 (bot outer, v1 = 16)
    mock_lms[158] = [125.0, 192.0]  # p3 (top inner)
    mock_lms[153] = [125.0, 208.0]  # p5 (bot inner, v2 = 16)

    ear_open = calculate_single_ear(mock_lms, LEFT_EYE_INDICES)
    # ear = (16 + 16) / (2 * 40) = 32 / 80 = 0.40
    assert pytest.approx(ear_open, 0.01) == 0.40

    # Setup closed left eye (vertical distances collapsed to 2px)
    mock_lms[160] = [115.0, 199.0]
    mock_lms[144] = [115.0, 201.0]
    mock_lms[158] = [125.0, 199.0]
    mock_lms[153] = [125.0, 201.0]

    ear_closed = calculate_single_ear(mock_lms, LEFT_EYE_INDICES)
    # ear = (2 + 2) / (2 * 40) = 4 / 80 = 0.05
    assert pytest.approx(ear_closed, 0.01) == 0.05
    assert ear_closed < ear_open


def test_ear_invalid_landmarks():
    # None landmarks
    assert calculate_single_ear(None, LEFT_EYE_INDICES) == 0.0
    # Short array
    assert calculate_single_ear(np.zeros((10, 2)), LEFT_EYE_INDICES) == 0.0
    # Collinear/zero horizontal width (divide by zero safeguard)
    collinear = np.zeros((478, 2), dtype=np.float32)
    assert calculate_single_ear(collinear, LEFT_EYE_INDICES) == 0.0


def test_mar_open_vs_closed():
    mock_lms = np.zeros((478, 2), dtype=np.float32)

    # Setup resting mouth: width = 60, height = 6
    mock_lms[61] = [100.0, 300.0]   # left corner
    mock_lms[291] = [160.0, 300.0]  # right corner (width = 60)
    mock_lms[13] = [130.0, 297.0]   # top inner
    mock_lms[14] = [130.0, 303.0]   # bot inner (v = 6)
    mock_lms[81] = [120.0, 297.0]
    mock_lms[178] = [120.0, 303.0]
    mock_lms[311] = [140.0, 297.0]
    mock_lms[402] = [140.0, 303.0]

    mar_closed = calculate_mar(mock_lms)
    assert mar_closed < 0.30

    # Setup wide yawn opening: height = 50
    mock_lms[13] = [130.0, 275.0]
    mock_lms[14] = [130.0, 325.0]  # v = 50
    mock_lms[81] = [120.0, 277.0]
    mock_lms[178] = [120.0, 323.0]
    mock_lms[311] = [140.0, 277.0]
    mock_lms[402] = [140.0, 323.0]

    mar_yawn = calculate_mar(mock_lms)
    assert mar_yawn > 0.65


def test_head_pose_estimation():
    estimator = HeadPoseEstimator(yaw_threshold=28.0, pitch_threshold=22.0)
    mock_lms = np.zeros((478, 2), dtype=np.float32)

    # Standard forward facing face projection on 1280x720 frame
    mock_lms[1] = [640.0, 360.0]    # Nose tip
    mock_lms[152] = [640.0, 520.0]  # Chin
    mock_lms[33] = [530.0, 300.0]   # Left eye corner
    mock_lms[263] = [750.0, 300.0]  # Right eye corner
    mock_lms[61] = [570.0, 440.0]   # Left mouth corner
    mock_lms[291] = [710.0, 440.0]  # Right mouth corner

    res = estimator.process(mock_lms, (720, 1280), current_time=time.time())
    assert isinstance(res.yaw, float)
    assert isinstance(res.pitch, float)
    assert isinstance(res.roll, float)
    assert res.direction in ["FORWARD", "LEFT", "RIGHT", "DOWN", "UP"]


def test_face_landmarks_tracker():
    import cv2
    import os
    from backend.detection.face_landmarks import FaceLandmarksTracker

    tracker = FaceLandmarksTracker()
    assert tracker.backend_type in ["mediapipe_tasks", "mediapipe_legacy", "none"]

    # Test blank frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res_blank = tracker.process(blank)
    assert res_blank.face_detected is False

    # Test real face image if available
    sample_path = "datasets/processed/test/Alert/sub_11_alert_000.jpg"
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        res = tracker.process(img)
        assert res.face_detected is True
        assert res.landmarks_pixel is not None
        assert len(res.landmarks_pixel) >= 468
        assert res.bounding_box is not None
        assert res.bounding_box_norm is not None

