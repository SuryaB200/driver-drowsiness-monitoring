"""Extracts and prepares real driver datasets from uploaded ZIP archives into balanced training splits."""
import os
import cv2
import json
import random
import zipfile
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"

def extract_and_prepare(max_train_per_class: int = 3000,
                        max_val_per_class: int = 600,
                        max_test_per_class: int = 400):
    """
    Extracts real images from drowsiness.zip and Driver Behavior Image Dataset.zip,
    resizes to 224x224 for efficient training, and balances classes.
    """
    drowsiness_zip = PROCESSED_DIR / "drowsiness.zip"
    if not drowsiness_zip.exists():
        # Fallback to alternate zip name if present
        alt_zip = PROCESSED_DIR / "Detect Open and Closed eye.zip"
        if alt_zip.exists():
            drowsiness_zip = alt_zip
        else:
            raise FileNotFoundError("Could not find drowsiness.zip or Detect Open and Closed eye.zip in datasets/processed")

    driver_behavior_zip = PROCESSED_DIR / "Driver Behavior Image Dataset.zip"

    # Target directories
    splits = ["train", "val", "test"]
    classes = ["Alert", "Drowsy"]

    for s in splits:
        for c in classes:
            target = PROCESSED_DIR / s / c
            target.mkdir(parents=True, exist_ok=True)
            # Remove old synthetic files if present
            for old_f in target.glob("sub_*.jpg"):
                try:
                    old_f.unlink()
                except Exception:
                    pass

    print(f"Reading from {drowsiness_zip.name}...")
    counts = {s: {c: 0 for c in classes} for s in splits}

    with zipfile.ZipFile(drowsiness_zip, "r") as zf:
        namelist = zf.namelist()
        random.seed(42)

        mapping = [
            # (zip_prefix, target_split, target_class, max_count)
            ("train/active", "train", "Alert", max_train_per_class),
            ("train/fatigue", "train", "Drowsy", max_train_per_class),
            ("val/active", "val", "Alert", max_val_per_class),
            ("val/fatigue", "val", "Drowsy", max_val_per_class),
            ("test/active", "test", "Alert", max_test_per_class),
            ("test/fatigue", "test", "Drowsy", max_test_per_class),
        ]

        for zip_prefix, split_name, cls_name, max_cnt in mapping:
            candidates = [n for n in namelist if n.startswith(zip_prefix + "/") and (n.endswith(".jpg") or n.endswith(".png"))]
            random.shuffle(candidates)
            selected = candidates[:max_cnt]
            print(f"Processing {len(selected)} images for {split_name}/{cls_name} (from {zip_prefix})...")

            target_dir = PROCESSED_DIR / split_name / cls_name
            for i, name in enumerate(selected):
                try:
                    data = zf.read(name)
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    # Standardize to 224x224
                    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                    out_path = target_dir / f"real_{cls_name.lower()}_{i:05d}.jpg"
                    cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    counts[split_name][cls_name] += 1
                except Exception as e:
                    pass

    # Also incorporate safe_driving from Driver Behavior Image Dataset if present
    if driver_behavior_zip.exists():
        print(f"Incorporating supplementary safe driving samples from {driver_behavior_zip.name}...")
        try:
            with zipfile.ZipFile(driver_behavior_zip, "r") as zf:
                safe_files = [n for n in zf.namelist() if "safe_driving" in n and (n.endswith(".jpg") or n.endswith(".png"))]
                target_dir = PROCESSED_DIR / "train" / "Alert"
                curr_cnt = counts["train"]["Alert"]
                for i, name in enumerate(safe_files):
                    data = zf.read(name)
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                        out_path = target_dir / f"driver_safe_{i:04d}.jpg"
                        cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        counts["train"]["Alert"] += 1
        except Exception as e:
            print(f"Note: Could not read Driver Behavior Image Dataset: {e}")

    # Write updated manifest
    manifest = {
        "dataset_name": "Real Driver Drowsiness & Behavior Dataset",
        "sources": [
            "Detect Open and Closed eye / Drowsiness Dataset (active, fatigue)",
            "Driver Behavior Image Dataset (safe driving)"
        ],
        "classes": ["Alert", "Drowsy"],
        "sample_counts": counts
    }
    with open(PROCESSED_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDataset Preparation Complete:")
    print(json.dumps(manifest, indent=2))
    return manifest

if __name__ == "__main__":
    extract_and_prepare()
