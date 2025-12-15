"""Core module containing data types and interfaces."""

from .types import Detection, EvaluationMetrics, TrainingConfig
from .interfaces import DatasetInterface, ModelInterface, EvaluatorInterface

__all__ = [
    "Detection",
    "EvaluationMetrics", 
    "TrainingConfig",
    "DatasetInterface",
    "ModelInterface", 
    "EvaluatorInterface",
]