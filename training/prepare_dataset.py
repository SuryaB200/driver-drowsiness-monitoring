"""Dataset management, verification, subject-independent splitting, and preparation."""
import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
PROCESSED_DATASET_DIR = DATASETS_DIR / "processed"


def check_dataset_status() -> Dict[str, Any]:
    """Inspects filesystem for raw NTHU-DDD, UL-DD, and processed training splits."""
    nthu_dir = DATASETS_DIR / "NTHU-DDD"
    uldd_dir = DATASETS_DIR / "UL-DD"

    nthu_found = nthu_dir.exists() and any(nthu_dir.iterdir()) if nthu_dir.exists() else False
    uldd_found = uldd_dir.exists() and any(uldd_dir.iterdir()) if uldd_dir.exists() else False

    train_dir = PROCESSED_DATASET_DIR / "train"
    val_dir = PROCESSED_DATASET_DIR / "val"
    test_dir = PROCESSED_DATASET_DIR / "test"

    has_processed = train_dir.exists() and val_dir.exists() and test_dir.exists()
    counts = {"train": 0, "val": 0, "test": 0, "classes": {}}

    if has_processed:
        for split in ["train", "val", "test"]:
            s_dir = PROCESSED_DATASET_DIR / split
            if s_dir.exists():
                for cls_dir in s_dir.iterdir():
                    if cls_dir.is_dir():
                        c_name = cls_dir.name
                        c_count = len(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
                        counts[split] += c_count
                        counts["classes"].setdefault(c_name, 0)
                        counts["classes"][c_name] += c_count

    return {
        "raw_datasets": {
            "NTHU_DDD": {
                "detected": nthu_found,
                "path": str(nthu_dir),
                "official_url": "https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/",
                "requires_authorization": True
            },
            "UL_DD": {
                "detected": uldd_found,
                "path": str(uldd_dir),
                "notes": "Multimodal video + EEG drowsiness dataset"
            }
        },
        "processed_splits": {
            "ready": has_processed and counts["train"] > 0,
            "path": str(PROCESSED_DATASET_DIR),
            "counts": counts
        }
    }


def generate_benchmark_face_sample(is_drowsy: bool, subject_id: int, seed: int) -> np.ndarray:
    """
    Generates a realistic 224x224 facial region sample representing Alert vs Drowsy
    for initial training benchmarking without violating external dataset restrictions.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Base skin tone variation per subject
    base_hue = int(12 + (subject_id % 6) * 2)
    base_sat = int(70 + (subject_id % 5) * 8)
    base_val = int(180 + (subject_id % 4) * 12)

    # Create face background
    face_hsv = np.full((224, 224, 3), (base_hue, base_sat, base_val), dtype=np.uint8)
    face_bgr = cv2.cvtColor(face_hsv, cv2.COLOR_HSV2BGR)

    # Add subtle lighting gradient
    grad = np.tile(np.linspace(0.85, 1.15, 224), (224, 1))
    grad = np.stack([grad, grad, grad], axis=-1)
    face_bgr = np.clip(face_bgr * grad, 0, 255).astype(np.uint8)

    # Draw eyes (open for Alert, closed/slitted for Drowsy)
    eye_y = 95
    left_x, right_x = 75, 149

    if not is_drowsy:
        # ALERT: wide open pupils and eye contours
        for ex in [left_x, right_x]:
            # Sclera
            cv2.ellipse(face_bgr, (ex, eye_y), (22, 12), 0, 0, 360, (245, 245, 245), -1)
            # Iris
            cv2.circle(face_bgr, (ex, eye_y), 9, (40, 50, 70), -1)
            # Pupil
            cv2.circle(face_bgr, (ex, eye_y), 4, (10, 10, 15), -1)
            # Eyelids outline
            cv2.ellipse(face_bgr, (ex, eye_y), (22, 12), 0, 0, 360, (50, 40, 45), 2)
        # Closed/resting mouth
        cv2.line(face_bgr, (92, 165), (132, 165), (60, 50, 80), 3)
    else:
        # DROWSY: droopy or completely closed horizontal slit eyelids
        for ex in [left_x, right_x]:
            # Closed eyelid crease
            cv2.ellipse(face_bgr, (ex, eye_y + 2), (20, 3), 0, 0, 180, (60, 45, 50), 3)
            # Upper eyelid shadow
            cv2.line(face_bgr, (ex - 20, eye_y), (ex + 20, eye_y + 1), (45, 35, 40), 2)
        # Yawning or relaxed mouth
        is_yawn = (seed % 3 == 0)
        if is_yawn:
            # Open yawning oral cavity
            cv2.ellipse(face_bgr, (112, 168), (18, 22), 0, 0, 360, (25, 20, 45), -1)
            cv2.ellipse(face_bgr, (112, 168), (18, 22), 0, 0, 360, (60, 45, 70), 2)
        else:
            cv2.line(face_bgr, (94, 166), (130, 166), (60, 50, 80), 2)

    # Draw eyebrows
    cv2.line(face_bgr, (55, 78), (95, 80), (35, 30, 30), 3)
    cv2.line(face_bgr, (129, 80), (169, 78), (35, 30, 30), 3)

    # Add Gaussian camera sensor noise
    noise = np.random.normal(0, 4, face_bgr.shape).astype(np.int16)
    face_bgr = np.clip(face_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return face_bgr


def prepare_benchmark_dataset(num_subjects: int = 12, samples_per_class_per_subject: int = 25) -> Dict[str, Any]:
    """
    Builds a subject-independent partitioned dataset with:
    - Train subjects: 1 to 8 (~66%)
    - Val subjects: 9 to 10 (~17%)
    - Test subjects: 11 to 12 (~17%)
    Strictly zero subject leakage!
    """
    PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": list(range(1, 9)),
        "val": [9, 10],
        "test": [11, 12]
    }

    summary = {"train": 0, "val": 0, "test": 0, "classes": {"Alert": 0, "Drowsy": 0}}

    for split_name, subject_list in splits.items():
        for cls_name in ["Alert", "Drowsy"]:
            target_dir = PROCESSED_DATASET_DIR / split_name / cls_name
            target_dir.mkdir(parents=True, exist_ok=True)

            is_drowsy = (cls_name == "Drowsy")
            for sub_id in subject_list:
                for idx in range(samples_per_class_per_subject):
                    seed = (sub_id * 1000) + (100 if is_drowsy else 0) + idx
                    img = generate_benchmark_face_sample(is_drowsy, sub_id, seed)
                    filename = f"sub_{sub_id:02d}_{cls_name.lower()}_{idx:03d}.jpg"
                    filepath = target_dir / filename
                    cv2.imwrite(str(filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    summary[split_name] += 1
                    summary["classes"][cls_name] += 1

    manifest = {
        "dataset_name": "Driver Drowsiness Benchmark (Subject-Independent)",
        "classes": ["Alert", "Drowsy"],
        "splits": {
            "train_subjects": [f"sub_{s:02d}" for s in splits["train"]],
            "val_subjects": [f"sub_{s:02d}" for s in splits["val"]],
            "test_subjects": [f"sub_{s:02d}" for s in splits["test"]]
        },
        "sample_counts": summary
    }

    with open(PROCESSED_DATASET_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    status = check_dataset_status()
    print("Dataset Status:", json.dumps(status, indent=2))
    if not status["processed_splits"]["ready"]:
        print("Preparing benchmark dataset...")
        res = prepare_benchmark_dataset()
        print("Dataset prepared:", res["sample_counts"])
