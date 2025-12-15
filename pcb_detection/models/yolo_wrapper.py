"""
PyTorch nn.Module wrapper for YOLODetector to make it compatible with standard training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .yolo_detector import YOLODetector


class YOLOWrapper(nn.Module):
    """PyTorch nn.Module wrapper for YOLODetector."""
    
    def __init__(self, model_config: Dict[str, Any], num_classes: int = 5):
        """
        Initialize YOLO wrapper.
        
        Args:
            model_config: Model configuration
            num_classes: Number of classes
        """
        super().__init__()
        
        # Create the detector
        self.detector = YOLODetector(model_config, num_classes)
        
        # Register the components as modules
        self.backbone = self.detector.backbone
        self.neck = self.detector.neck
        self.head = self.detector.head
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.detector.forward(x)
        
    def compute_loss(self, predictions: torch.Tensor, targets: Dict) -> torch.Tensor:
        """Compute loss."""
        loss_dict = self.detector.compute_loss(predictions, targets)
        # Return total loss for training
        if isinstance(loss_dict, dict):
            return loss_dict.get('total_loss', sum(loss_dict.values()))
        return loss_dict
        
    def predict(self, image):
        """Predict detections."""
        return self.detector.predict(image)
        
    def load_weights(self, path: str):
        """Load weights."""
        return self.detector.load_weights(path)
        
    def save_weights(self, path: str):
        """Save weights."""
        return self.detector.save_weights(path)