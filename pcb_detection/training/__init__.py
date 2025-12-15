"""Training management for PCB defect detection."""

from .trainer import Trainer
from .checkpoint_manager import CheckpointManager

__all__ = [
    "Trainer",
    "CheckpointManager",
]