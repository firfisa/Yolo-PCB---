"""
Core interfaces defining system boundaries.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch

from .types import Detection, EvaluationMetrics, TrainingConfig


class DatasetInterface(ABC):
    """Abstract interface for dataset handling."""

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        pass

    @abstractmethod
    def load_annotations(self) -> List[Dict]:
        """Load annotation data."""
        pass

    @abstractmethod
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution statistics."""
        pass


class ModelInterface(ABC):
    """Abstract interface for detection models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Detection]:
        """Predict detections for a single image."""
        pass

    @abstractmethod
    def train_model(self, config: TrainingConfig) -> Dict[str, Any]:
        """Train the model with given configuration."""
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Load model weights from file."""
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Save model weights to file."""
        pass


class EvaluatorInterface(ABC):
    """Abstract interface for model evaluation."""

    @abstractmethod
    def calculate_map(self, predictions: List[List[Detection]], 
                     ground_truths: List[List[Detection]]) -> float:
        """Calculate mean Average Precision."""
        pass

    @abstractmethod
    def calculate_ap_per_class(self, predictions: List[List[Detection]], 
                              ground_truths: List[List[Detection]]) -> Dict[str, float]:
        """Calculate Average Precision per class."""
        pass

    @abstractmethod
    def generate_metrics_report(self, predictions: List[List[Detection]], 
                               ground_truths: List[List[Detection]]) -> EvaluationMetrics:
        """Generate comprehensive evaluation metrics."""
        pass

    @abstractmethod
    def save_results(self, results: EvaluationMetrics, path: str, 
                    format: str = "json") -> None:
        """Save evaluation results to file."""
        pass


class VisualizerInterface(ABC):
    """Abstract interface for visualization."""

    @abstractmethod
    def draw_detections(self, image: np.ndarray, 
                       detections: List[Detection]) -> np.ndarray:
        """Draw detections on image."""
        pass

    @abstractmethod
    def create_comparison_image(self, image: np.ndarray,
                               gt_detections: List[Detection],
                               pred_detections: List[Detection]) -> np.ndarray:
        """Create side-by-side comparison image."""
        pass

    @abstractmethod
    def create_comparison_grid(self, images: List[np.ndarray],
                              gt_detections: List[List[Detection]],
                              pred_detections: List[List[Detection]]) -> np.ndarray:
        """Create grid of comparison images."""
        pass

    @abstractmethod
    def save_visualization(self, image: np.ndarray, path: str) -> None:
        """Save visualization to file."""
        pass


class DataAugmentationInterface(ABC):
    """Abstract interface for data augmentation."""

    @abstractmethod
    def random_rotation(self, image: np.ndarray, 
                       angle_range: Tuple[int, int]) -> np.ndarray:
        """Apply random rotation."""
        pass

    @abstractmethod
    def random_scaling(self, image: np.ndarray, 
                      scale_range: Tuple[float, float]) -> np.ndarray:
        """Apply random scaling."""
        pass

    @abstractmethod
    def color_jittering(self, image: np.ndarray, 
                       brightness: float, contrast: float) -> np.ndarray:
        """Apply color jittering."""
        pass

    @abstractmethod
    def apply_augmentation(self, image: np.ndarray, 
                          annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Apply augmentation to image and annotations."""
        pass