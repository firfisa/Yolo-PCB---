"""
Configuration utility functions.
"""

import os
from typing import Dict, Any
from ..core.types import TrainingConfig, CLASS_MAPPING


class ConfigUtils:
    """Utility functions for configuration management."""
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "model": {
                "name": "yolov8n",
                "num_classes": 5,
                "input_size": 640,
            },
            "training": {
                "epochs": 300,
                "batch_size": 16,
                "learning_rate": 0.01,
                "patience": 50,
                "save_period": 10,
            },
            "data": {
                "train_path": "训练集-PCB_DATASET",
                "test_path": "PCB_瑕疵测试集",
                "augmentation": True,
            },
            "evaluation": {
                "iou_threshold": 0.5,
                "confidence_threshold": 0.25,
            },
            "visualization": {
                "class_mapping": CLASS_MAPPING,
                "save_visualizations": True,
            }
        }
        
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_sections = ["model", "training", "data", "evaluation"]
        for section in required_sections:
            if section not in config:
                return False
        return True
        
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = ConfigUtils.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
        
    @staticmethod
    def config_to_training_config(config: Dict[str, Any]) -> TrainingConfig:
        """
        Convert configuration dictionary to TrainingConfig.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            TrainingConfig object
        """
        training_config = config.get("training", {})
        model_config = config.get("model", {})
        
        return TrainingConfig(
            model_name=model_config.get("name", "yolov8n"),
            epochs=training_config.get("epochs", 300),
            batch_size=training_config.get("batch_size", 16),
            learning_rate=training_config.get("learning_rate", 0.01),
            image_size=model_config.get("input_size", 640),
            augmentation=config.get("data", {}).get("augmentation", True),
            patience=training_config.get("patience", 50),
            save_period=training_config.get("save_period", 10),
        )