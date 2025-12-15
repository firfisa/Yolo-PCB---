"""
Tests for configuration utilities.
"""

import pytest
from pcb_detection.utils.config_utils import ConfigUtils
from pcb_detection.core.types import TrainingConfig


class TestConfigUtils:
    """Test ConfigUtils class."""
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = ConfigUtils.create_default_config()
        
        assert "model" in config
        assert "training" in config
        assert "data" in config
        assert "evaluation" in config
        assert "visualization" in config
        
        assert config["model"]["name"] == "yolov8n"
        assert config["model"]["num_classes"] == 5
    
    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = ConfigUtils.create_default_config()
        assert ConfigUtils.validate_config(valid_config) is True
        
        invalid_config = {"model": {}}
        assert ConfigUtils.validate_config(invalid_config) is False
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {
            "model": {"name": "yolov8n", "size": 640},
            "training": {"epochs": 100}
        }
        override_config = {
            "model": {"name": "yolov8s"},
            "training": {"batch_size": 32}
        }
        
        merged = ConfigUtils.merge_configs(base_config, override_config)
        
        assert merged["model"]["name"] == "yolov8s"  # overridden
        assert merged["model"]["size"] == 640        # preserved
        assert merged["training"]["epochs"] == 100   # preserved
        assert merged["training"]["batch_size"] == 32 # added
    
    def test_config_to_training_config(self):
        """Test conversion to TrainingConfig."""
        config = ConfigUtils.create_default_config()
        training_config = ConfigUtils.config_to_training_config(config)
        
        assert isinstance(training_config, TrainingConfig)
        assert training_config.model_name == "yolov8n"
        assert training_config.epochs == 300
        assert training_config.batch_size == 16