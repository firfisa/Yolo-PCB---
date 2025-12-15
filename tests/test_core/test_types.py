"""
Tests for core data types.
"""

import pytest
from pcb_detection.core.types import Detection, EvaluationMetrics, TrainingConfig


class TestDetection:
    """Test Detection data class."""
    
    def test_valid_detection(self):
        """Test creating valid detection."""
        detection = Detection(
            bbox=(0.1, 0.2, 0.3, 0.4),
            confidence=0.85,
            class_id=0,
            class_name="Mouse_bite"
        )
        assert detection.bbox == (0.1, 0.2, 0.3, 0.4)
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.class_name == "Mouse_bite"
    
    def test_invalid_confidence(self):
        """Test detection with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Detection(
                bbox=(0.1, 0.2, 0.3, 0.4),
                confidence=1.5,
                class_id=0,
                class_name="Mouse_bite"
            )
    
    def test_invalid_class_id(self):
        """Test detection with invalid class ID."""
        with pytest.raises(ValueError, match="Class ID must be between 0 and 4"):
            Detection(
                bbox=(0.1, 0.2, 0.3, 0.4),
                confidence=0.85,
                class_id=10,
                class_name="Invalid"
            )
    
    def test_invalid_bbox(self):
        """Test detection with invalid bbox."""
        with pytest.raises(ValueError, match="Bbox must have 4 elements"):
            Detection(
                bbox=(0.1, 0.2, 0.3),
                confidence=0.85,
                class_id=0,
                class_name="Mouse_bite"
            )


class TestEvaluationMetrics:
    """Test EvaluationMetrics data class."""
    
    def test_valid_metrics(self):
        """Test creating valid metrics."""
        metrics = EvaluationMetrics(
            map_50=0.75,
            ap_per_class={"Mouse_bite": 0.80, "Open_circuit": 0.70},
            precision=0.82,
            recall=0.76,
            f1_score=0.79
        )
        assert metrics.map_50 == 0.75
        assert metrics.precision == 0.82
    
    def test_invalid_metrics(self):
        """Test metrics with invalid values."""
        with pytest.raises(ValueError, match="Metrics must be between 0 and 1"):
            EvaluationMetrics(
                map_50=1.5,
                ap_per_class={},
                precision=0.82,
                recall=0.76,
                f1_score=0.79
            )


class TestTrainingConfig:
    """Test TrainingConfig data class."""
    
    def test_valid_config(self):
        """Test creating valid training config."""
        config = TrainingConfig(
            model_name="yolov8n",
            epochs=300,
            batch_size=16,
            learning_rate=0.01
        )
        assert config.model_name == "yolov8n"
        assert config.epochs == 300
    
    def test_invalid_epochs(self):
        """Test config with invalid epochs."""
        with pytest.raises(ValueError, match="Epochs must be positive"):
            TrainingConfig(epochs=-10)
    
    def test_invalid_batch_size(self):
        """Test config with invalid batch size."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            TrainingConfig(batch_size=0)
    
    def test_invalid_learning_rate(self):
        """Test config with invalid learning rate."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            TrainingConfig(learning_rate=-0.01)