"""MobileNetV3-Small transfer learning architecture for drowsiness detection."""
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

class DrowsinessMobileNetV3(nn.Module):
    """
    MobileNetV3-Small classifier adapted for driver drowsiness state detection.
    Default classes: 0: Alert, 1: Drowsy
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.25):
        super().__init__()
        self.num_classes = num_classes

        try:
            import torchvision.models as models
            from torchvision.models import MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
        except Exception:
            import torchvision.models as models
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)

        # Replace classification head
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_trainable_params_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self, freeze: bool = True):
        """Freeze feature extraction layers for initial transfer learning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze


def save_checkpoint(model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    epoch: int,
                    val_loss: float,
                    val_acc: float,
                    filepath: str,
                    class_names: Optional[list] = None) -> None:
    """Save model checkpoint with metadata."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'num_classes': getattr(model, 'num_classes', 2),
        'class_names': class_names or ["Alert", "Drowsy"],
        'architecture': 'MobileNetV3-Small'
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, filepath)


def load_checkpoint(filepath: str,
                    device: str = "cpu") -> Tuple[Optional[DrowsinessMobileNetV3], Dict[str, Any]]:
    """Loads a checkpoint from disk or returns (None, metadata)."""
    if not os.path.exists(filepath):
        return None, {'error': f"Checkpoint file '{filepath}' does not exist"}

    try:
        checkpoint = torch.load(filepath, map_location=device)
        num_classes = checkpoint.get('num_classes', 2)
        model = DrowsinessMobileNetV3(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        meta = {
            'epoch': checkpoint.get('epoch', 0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'val_acc': checkpoint.get('val_acc', 0.0),
            'num_classes': num_classes,
            'class_names': checkpoint.get('class_names', ["Alert", "Drowsy"]),
            'architecture': checkpoint.get('architecture', 'MobileNetV3-Small')
        }
        return model, meta
    except Exception as e:
        return None, {'error': f"Failed to load checkpoint: {e}"}
