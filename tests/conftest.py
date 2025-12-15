"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
from typing import List
import tempfile
import os

from pcb_detection.core.types import Detection, EvaluationMetrics, TrainingConfig


@pytest.fixture
def sample_detection():
    """Sample detection for testing."""
    return Detection(
        bbox=(0.1, 0.2, 0.3, 0.4),
        confidence=0.85,
        class_id=0,
        class_name="Mouse_bite"
    )


@pytest.fixture
def sample_detections():
    """Sample list of detections for testing."""
    return [
        Detection(bbox=(0.1, 0.2, 0.3, 0.4), confidence=0.85, class_id=0, class_name="Mouse_bite"),
        Detection(bbox=(0.5, 0.6, 0.2, 0.3), confidence=0.92, class_id=1, class_name="Open_circuit"),
        Detection(bbox=(0.7, 0.1, 0.15, 0.25), confidence=0.78, class_id=2, class_name="Short"),
    ]


@pytest.fixture
def sample_image():
    """Sample image for testing."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_training_config():
    """Sample training configuration for testing."""
    return TrainingConfig(
        model_name="yolov8n",
        epochs=10,
        batch_size=4,
        learning_rate=0.001,
        image_size=640,
        augmentation=True
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_metrics():
    """Sample evaluation metrics for testing."""
    return EvaluationMetrics(
        map_50=0.75,
        ap_per_class={
            "Mouse_bite": 0.80,
            "Open_circuit": 0.70,
            "Short": 0.75,
            "Spur": 0.72,
            "Spurious_copper": 0.78
        },
        precision=0.82,
        recall=0.76,
        f1_score=0.79
    )