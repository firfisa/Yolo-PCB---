"""
PCB Defect Detection System

A comprehensive system for detecting PCB defects using YOLO-based object detection.
Supports five defect types: Mouse_bite, Open_circuit, Short, Spur, Spurious_copper.
"""

__version__ = "0.1.0"
__author__ = "PCB Detection Team"

from .core.types import Detection, EvaluationMetrics, TrainingConfig
from .data.dataset import PCBDataset
from .models.yolo_detector import YOLODetector
from .evaluation.evaluator import Evaluator
from .visualization.visualizer import Visualizer

__all__ = [
    "Detection",
    "EvaluationMetrics", 
    "TrainingConfig",
    "PCBDataset",
    "YOLODetector",
    "Evaluator",
    "Visualizer",
]