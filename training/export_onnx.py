"""Export trained MobileNetV3 PyTorch model to ONNX format for edge deployment."""
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
from backend.config import CHECKPOINTS_DIR
from backend.models.mobilenet import load_checkpoint

def export_to_onnx(checkpoint_name: str = "best_mobilenet_v3.pth",
                   onnx_name: str = "mobilenet_v3_small.onnx",
                   image_size: int = 224) -> str:
    ckpt_path = str(CHECKPOINTS_DIR / checkpoint_name)
    onnx_path = str(CHECKPOINTS_DIR / onnx_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file '{ckpt_path}' not found.")

    model, meta = load_checkpoint(ckpt_path, device="cpu")
    if model is None:
        raise ValueError("Failed to load model from checkpoint")

    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, requires_grad=False)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_face_crop'],
        output_names=['logits']
    )

    print(f"Successfully exported ONNX model to: {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    export_to_onnx()
