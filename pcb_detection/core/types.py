"""
Core data types for PCB defect detection system.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) normalized coordinates
    confidence: float                        # confidence score
    class_id: int                           # class ID
    class_name: str                         # class name

    def __post_init__(self):
        """Validate detection data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if not (0 <= self.class_id <= 4):
            raise ValueError(f"Class ID must be between 0 and 4, got {self.class_id}")
        if len(self.bbox) != 4:
            raise ValueError(f"Bbox must have 4 elements, got {len(self.bbox)}")


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for model performance."""
    map_50: float                           # mAP@IoU=0.5
    ap_per_class: Dict[str, float]          # per-class AP
    precision: float                        # precision
    recall: float                          # recall
    f1_score: float                        # F1 score

    def __post_init__(self):
        """Validate metrics."""
        metrics = [self.map_50, self.precision, self.recall, self.f1_score]
        for metric in metrics:
            if not (0.0 <= metric <= 1.0):
                raise ValueError(f"Metrics must be between 0 and 1, got {metric}")


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    model_name: str = "yolov8n"            # model name
    epochs: int = 300                      # training epochs
    batch_size: int = 16                   # batch size
    learning_rate: float = 0.01            # learning rate
    image_size: int = 640                  # image size
    augmentation: bool = True              # use data augmentation
    device: str = "auto"                   # device (auto, cpu, cuda)
    workers: int = 8                       # number of workers
    patience: int = 50                     # early stopping patience
    save_period: int = 10                  # save checkpoint every N epochs

    def __post_init__(self):
        """Validate configuration."""
        if self.epochs <= 0:
            raise ValueError(f"Epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        if self.image_size <= 0:
            raise ValueError(f"Image size must be positive, got {self.image_size}")


# Class mapping for PCB defects
CLASS_MAPPING = {
    0: "Mouse_bite",      # 鼠标咬痕
    1: "Open_circuit",    # 开路
    2: "Short",           # 短路  
    3: "Spur",            # 毛刺
    4: "Spurious_copper"  # 杂散铜
}

# Reverse mapping
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_MAPPING.items()}

# Single letter labels for visualization
CLASS_LABELS = {
    0: "M",  # Mouse_bite
    1: "O",  # Open_circuit
    2: "S",  # Short
    3: "P",  # Spur (P for spur)
    4: "C",  # Spurious_copper (C for copper)
}

# Colors for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # Green for Mouse_bite
    1: (255, 0, 0),      # Blue for Open_circuit
    2: (0, 0, 255),      # Red for Short
    3: (255, 255, 0),    # Cyan for Spur
    4: (255, 0, 255),    # Magenta for Spurious_copper
}

# Number of classes
NUM_CLASSES = len(CLASS_MAPPING)