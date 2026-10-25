"""Automated tests for MobileNetV3 architecture, inference, and dataset split integrity."""
import os
import json
import torch
import numpy as np
from pathlib import Path

from backend.models.mobilenet import DrowsinessMobileNetV3, load_checkpoint
from backend.models.inference import MobileNetInferenceEngine
from backend.config import CHECKPOINTS_DIR, DATASETS_DIR


def test_mobilenet_v3_small_forward():
    model = DrowsinessMobileNetV3(num_classes=2, pretrained=False)
    model.eval()

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, 2)


def test_checkpoint_load_and_inference():
    ckpt_path = str(CHECKPOINTS_DIR / "best_mobilenet_v3.pth")
    assert os.path.exists(ckpt_path), "Trained checkpoint must exist"

    engine = MobileNetInferenceEngine(checkpoint_path=ckpt_path, device="cpu")
    assert engine.model is not None

    # Test mock face crop inference
    dummy_face = (np.random.rand(150, 150, 3) * 255).astype(np.uint8)
    res = engine.predict(dummy_face)

    assert res.is_available is True
    assert res.predicted_label in ["Alert", "Drowsy"]
    assert 0.0 <= res.drowsy_probability <= 1.0
    assert 0.0 <= res.alert_probability <= 1.0
    assert res.latency_ms > 0.0


def test_dataset_split_integrity_zero_subject_leakage():
    manifest_file = DATASETS_DIR / "processed" / "manifest.json"
    assert manifest_file.exists(), "Processed dataset manifest must exist"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_subs = set(manifest["splits"]["train_subjects"])
    val_subs = set(manifest["splits"]["val_subjects"])
    test_subs = set(manifest["splits"]["test_subjects"])

    # Assert mutual disjointness: strictly zero subject leakage
    assert len(train_subs.intersection(val_subs)) == 0
    assert len(train_subs.intersection(test_subs)) == 0
    assert len(val_subs.intersection(test_subs)) == 0
