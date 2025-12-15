"""
Configuration Interface for PCB Defect Detection System.

This module provides a high-level interface for managing and applying
different technology configurations.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

from .tech_config_manager import TechConfigManager, TechStackConfig
from .config_utils import ConfigUtils


class ConfigInterface:
    """High-level interface for configuration management."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration interface.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.tech_manager = TechConfigManager(
            str(self.config_dir / "advanced_config.yaml")
        )
        self.current_config = None
        
    def list_available_configs(self) -> Dict[str, str]:
        """
        List all available technology configurations.
        
        Returns:
            Dictionary mapping config names to descriptions
        """
        configs = {}
        for name in self.tech_manager.list_configs():
            config = self.tech_manager.get_config(name)
            if config:
                configs[name] = config.description
        return configs
        
    def get_config_details(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Detailed configuration information or None if not found
        """
        config = self.tech_manager.get_config(config_name)
        if not config:
            return None
            
        return {
            'name': config.name,
            'description': config.description,
            'model': config.model_config,
            'loss': config.loss_config,
            'augmentation': config.augmentation_config,
            'training': config.training_config,
            'inference': config.inference_config,
            'expected_performance': config.expected_performance,
            'resource_requirements': config.resource_requirements
        }
        
    def select_config(self, config_name: str) -> bool:
        """
        Select a configuration for use.
        
        Args:
            config_name: Name of the configuration to select
            
        Returns:
            True if selection successful, False otherwise
        """
        config = self.tech_manager.get_config(config_name)
        if not config:
            return False
            
        # Validate the configuration
        is_valid, errors = self.tech_manager.validate_config(config)
        if not is_valid:
            print(f"Configuration validation failed: {errors}")
            return False
            
        self.current_config = config
        return True
        
    def get_current_config(self) -> Optional[TechStackConfig]:
        """
        Get the currently selected configuration.
        
        Returns:
            Current configuration or None if none selected
        """
        return self.current_config
        
    def convert_to_training_config(self) -> Optional[Dict[str, Any]]:
        """
        Convert current configuration to training format.
        
        Returns:
            Training configuration dictionary or None if no config selected
        """
        if not self.current_config:
            return None
            
        # Merge all configuration sections
        training_config = {
            'model': self.current_config.model_config,
            'loss': self.current_config.loss_config,
            'augmentation': self.current_config.augmentation_config,
            'training': self.current_config.training_config,
            'inference': self.current_config.inference_config,
            'data': {
                'train_path': "训练集-PCB_DATASET",
                'test_path': "PCB_瑕疵测试集",
                'num_classes': 5
            },
            'evaluation': {
                'iou_threshold': 0.5,
                'save_results': True,
                'results_format': 'json'
            },
            'visualization': {
                'save_visualizations': True,
                'output_dir': 'outputs/visualizations'
            }
        }
        
        return training_config
        
    def get_optimization_suggestions(self, target: str = 'map') -> List[str]:
        """
        Get optimization suggestions for current configuration.
        
        Args:
            target: Target metric to optimize ('map', 'fps', 'memory')
            
        Returns:
            List of optimization suggestions
        """
        if not self.current_config:
            return ["No configuration selected"]
            
        return self.tech_manager.get_optimization_suggestions(
            self.current_config, target
        )
        
    def compare_configurations(self, config1: str, config2: str) -> Dict[str, Any]:
        """
        Compare two configurations.
        
        Args:
            config1: Name of first configuration
            config2: Name of second configuration
            
        Returns:
            Comparison results
        """
        return self.tech_manager.compare_configs(config1, config2)
        
    def create_baseline_config(self) -> Dict[str, Any]:
        """
        Create a baseline configuration for initial training.
        
        Returns:
            Baseline configuration dictionary
        """
        baseline_config = {
            'model': {
                'name': 'yolov8n',
                'backbone': 'yolov8n',
                'num_classes': 5,
                'input_size': 640,
                'pretrained': True,
                'attention': None,  # No attention for baseline
                'use_fpn': False
            },
            'loss': {
                'classification': 'ce',  # Standard cross-entropy
                'bbox_regression': 'smooth_l1',  # Standard smooth L1
                'objectness': 'bce',
                'weights': {
                    'cls': 1.0,
                    'bbox': 1.0,
                    'obj': 1.0
                }
            },
            'augmentation': {
                'basic': {
                    'rotation_range': [0, 0],  # No rotation for baseline
                    'scale_range': [1.0, 1.0],  # No scaling
                    'brightness_range': [0, 0],  # No brightness change
                    'contrast_range': [1.0, 1.0],  # No contrast change
                    'flip_horizontal': False,
                    'flip_vertical': False,
                    'prob': 0.0  # No augmentation
                },
                'advanced': {
                    'mosaic_prob': 0.0,
                    'copy_paste_prob': 0.0,
                    'mixup_prob': 0.0,
                    'use_albumentations': False
                }
            },
            'training': {
                'epochs': 50,  # Shorter training for baseline
                'batch_size': 16,
                'learning_rate': 0.01,
                'multi_scale': False,
                'image_size': 640,
                'warmup_epochs': 0,
                'lr_scheduler': 'step',
                'early_stopping': False,
                'patience': 10,
                'device': 'auto',
                'workers': 4
            },
            'inference': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'use_tta': False
            },
            'data': {
                'train_path': "训练集-PCB_DATASET",
                'test_path': "PCB_瑕疵测试集",
                'num_classes': 5,
                'train_split': 0.8,
                'val_split': 0.2
            },
            'evaluation': {
                'iou_threshold': 0.5,
                'save_results': True,
                'results_format': 'json'
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'log_dir': 'logs/baseline'
            },
            'output': {
                'model_dir': 'models/baseline',
                'results_dir': 'results/baseline',
                'visualization_dir': 'visualizations/baseline'
            }
        }
        
        return baseline_config
        
    def save_current_config(self, filepath: str) -> bool:
        """
        Save the current configuration to file.
        
        Args:
            filepath: Path to save the configuration
            
        Returns:
            True if save successful, False otherwise
        """
        if not self.current_config:
            return False
            
        try:
            self.tech_manager.save_config(self.current_config, filepath)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
            
    def validate_current_config(self) -> tuple[bool, List[str]]:
        """
        Validate the current configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self.current_config:
            return False, ["No configuration selected"]
            
        return self.tech_manager.validate_config(self.current_config)
        
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all available configurations.
        
        Returns:
            Configuration summary
        """
        return self.tech_manager.get_config_summary()
        
    def recommend_config(self, requirements: Dict[str, Any]) -> str:
        """
        Recommend a configuration based on requirements.
        
        Args:
            requirements: Dictionary containing requirements like:
                - target_map: Target mAP value
                - max_training_time: Maximum training time in hours
                - gpu_memory: Available GPU memory in GB
                - target_fps: Target inference FPS
                
        Returns:
            Recommended configuration name
        """
        target_map = requirements.get('target_map', 0.3)
        max_training_time = requirements.get('max_training_time', 8)
        gpu_memory = requirements.get('gpu_memory', 4)
        target_fps = requirements.get('target_fps', 30)
        
        # Simple recommendation logic
        if target_map < 0.2 and target_fps > 50:
            return 'basic'
        elif target_map > 0.4 and max_training_time > 12:
            return 'performance'
        else:
            return 'balanced'
            
    def create_experiment_config(self, base_config: str, experiment_name: str,
                               modifications: Dict[str, Any]) -> bool:
        """
        Create an experimental configuration.
        
        Args:
            base_config: Base configuration name
            experiment_name: Name for the experiment
            modifications: Modifications to apply
            
        Returns:
            True if creation successful, False otherwise
        """
        try:
            custom_config = self.tech_manager.create_custom_config(
                base_config, modifications, f"exp_{experiment_name}",
                f"实验配置: {experiment_name}"
            )
            
            # Add to available configurations
            self.tech_manager.configs[f"exp_{experiment_name}"] = custom_config
            return True
            
        except Exception as e:
            print(f"Error creating experiment config: {e}")
            return False


def create_config_interface() -> ConfigInterface:
    """
    Factory function to create a configuration interface.
    
    Returns:
        Configured ConfigInterface instance
    """
    return ConfigInterface()


def get_recommended_config(task_type: str = "baseline") -> Dict[str, Any]:
    """
    Get a recommended configuration for a specific task.
    
    Args:
        task_type: Type of task ('baseline', 'development', 'production')
        
    Returns:
        Recommended configuration dictionary
    """
    interface = create_config_interface()
    
    if task_type == "baseline":
        return interface.create_baseline_config()
    elif task_type == "development":
        interface.select_config("balanced")
        return interface.convert_to_training_config()
    elif task_type == "production":
        interface.select_config("performance")
        return interface.convert_to_training_config()
    else:
        # Default to baseline
        return interface.create_baseline_config()