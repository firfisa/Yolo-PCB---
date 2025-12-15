"""
Advanced Technology Configuration Manager for PCB Defect Detection.

This module provides a comprehensive configuration system for managing different
technology stacks (basic, balanced, performance) with validation and optimization
suggestions.
"""

import os
import yaml
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings


@dataclass
class TechStackConfig:
    """Configuration for a specific technology stack."""
    name: str
    description: str
    model_config: Dict[str, Any]
    loss_config: Dict[str, Any]
    augmentation_config: Dict[str, Any]
    training_config: Dict[str, Any]
    inference_config: Dict[str, Any]
    expected_performance: Dict[str, float]
    resource_requirements: Dict[str, Any]


class TechConfigManager:
    """Manager for advanced technology configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the technology configuration manager.
        
        Args:
            config_path: Path to the advanced configuration file
        """
        self.config_path = config_path or "config/advanced_config.yaml"
        self.configs = {}
        self.validation_rules = self._setup_validation_rules()
        self.load_configurations()
        
    def load_configurations(self) -> None:
        """Load all technology configurations from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
                
            # Load predefined technology stacks
            self._load_tech_stacks(raw_config)
            
        except FileNotFoundError:
            warnings.warn(f"Configuration file not found: {self.config_path}")
            self._create_default_configs()
        except Exception as e:
            warnings.warn(f"Error loading configuration: {e}")
            self._create_default_configs()
            
    def _load_tech_stacks(self, raw_config: Dict[str, Any]) -> None:
        """Load technology stacks from raw configuration."""
        # Basic configuration
        if 'basic_config' in raw_config:
            self.configs['basic'] = TechStackConfig(
                name="basic",
                description="快速原型和基线建立 - 轻量级配置",
                model_config=raw_config['basic_config']['model'],
                loss_config=raw_config['basic_config']['loss'],
                augmentation_config=raw_config['basic_config']['augmentation'],
                training_config=raw_config['basic_config']['training'],
                inference_config=raw_config['basic_config']['inference'],
                expected_performance={'map': 0.15, 'fps': 60},
                resource_requirements={'gpu_memory': '2GB', 'training_time': '2-4h'}
            )
            
        # Balanced configuration
        if 'balanced_config' in raw_config:
            self.configs['balanced'] = TechStackConfig(
                name="balanced",
                description="精度与速度兼顾 - 平衡配置",
                model_config=raw_config['balanced_config']['model'],
                loss_config=raw_config['balanced_config']['loss'],
                augmentation_config=raw_config['balanced_config']['augmentation'],
                training_config=raw_config['balanced_config']['training'],
                inference_config=raw_config['balanced_config']['inference'],
                expected_performance={'map': 0.35, 'fps': 30},
                resource_requirements={'gpu_memory': '4GB', 'training_time': '4-8h'}
            )
            
        # Performance configuration
        if 'performance_config' in raw_config:
            self.configs['performance'] = TechStackConfig(
                name="performance",
                description="追求最高精度 - 性能配置",
                model_config=raw_config['performance_config']['model'],
                loss_config=raw_config['performance_config']['loss'],
                augmentation_config=raw_config['performance_config']['augmentation'],
                training_config=raw_config['performance_config']['training'],
                inference_config=raw_config['performance_config']['inference'],
                expected_performance={'map': 0.50, 'fps': 15},
                resource_requirements={'gpu_memory': '8GB', 'training_time': '8-16h'}
            )
            
        # Load experiment configurations
        if 'experiment_configs' in raw_config:
            for name, config in raw_config['experiment_configs'].items():
                base_config = self.configs.get(config.get('base', 'basic_config').replace('_config', ''))
                if base_config:
                    exp_config = self._merge_configs(asdict(base_config), config)
                    self.configs[f"exp_{name}"] = TechStackConfig(
                        name=f"exp_{name}",
                        description=f"实验配置: {name}",
                        **exp_config
                    )
                    
    def _create_default_configs(self) -> None:
        """Create default configurations if file loading fails."""
        self.configs['basic'] = TechStackConfig(
            name="basic",
            description="基础配置 - 快速原型",
            model_config={
                'backbone': 'yolov8n',
                'attention': 'se',
                'use_fpn': False
            },
            loss_config={
                'classification': 'focal',
                'bbox_regression': 'ciou',
                'weights': {'cls': 1.0, 'bbox': 1.0, 'obj': 1.0}
            },
            augmentation_config={
                'basic': {
                    'rotation_range': [-10, 10],
                    'scale_range': [0.9, 1.1],
                    'prob': 0.5
                },
                'advanced': {
                    'mosaic_prob': 0.0,
                    'copy_paste_prob': 0.0,
                    'mixup_prob': 0.0
                }
            },
            training_config={
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'multi_scale': False
            },
            inference_config={
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'use_tta': False
            },
            expected_performance={'map': 0.15, 'fps': 60},
            resource_requirements={'gpu_memory': '2GB', 'training_time': '2-4h'}
        )
        
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup validation rules for configurations."""
        return {
            'model': {
                'backbone': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                'attention': ['se', 'cbam', 'eca', 'coord', None],
                'use_fpn': [True, False],
                'use_panet': [True, False]
            },
            'loss': {
                'classification': ['focal', 'ce', 'combo', 'class_balanced'],
                'bbox_regression': ['iou', 'giou', 'diou', 'ciou', 'smooth_l1'],
                'objectness': ['bce', 'focal']
            },
            'training': {
                'epochs': (1, 1000),
                'batch_size': (1, 128),
                'learning_rate': (1e-6, 1.0),
                'multi_scale': [True, False]
            },
            'inference': {
                'confidence_threshold': (0.01, 0.99),
                'iou_threshold': (0.01, 0.99),
                'use_tta': [True, False]
            }
        }
        
    def get_config(self, config_name: str) -> Optional[TechStackConfig]:
        """
        Get a specific technology configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Technology stack configuration or None if not found
        """
        return self.configs.get(config_name)
        
    def list_configs(self) -> List[str]:
        """
        List all available configuration names.
        
        Returns:
            List of configuration names
        """
        return list(self.configs.keys())
        
    def validate_config(self, config: TechStackConfig) -> Tuple[bool, List[str]]:
        """
        Validate a technology configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate model configuration
        model_errors = self._validate_section(config.model_config, 'model')
        errors.extend(model_errors)
        
        # Validate loss configuration
        loss_errors = self._validate_section(config.loss_config, 'loss')
        errors.extend(loss_errors)
        
        # Validate training configuration
        training_errors = self._validate_section(config.training_config, 'training')
        errors.extend(training_errors)
        
        # Validate inference configuration
        inference_errors = self._validate_section(config.inference_config, 'inference')
        errors.extend(inference_errors)
        
        # Check for logical inconsistencies
        logical_errors = self._validate_logical_consistency(config)
        errors.extend(logical_errors)
        
        return len(errors) == 0, errors
        
    def _validate_section(self, section_config: Dict[str, Any], section_name: str) -> List[str]:
        """Validate a configuration section."""
        errors = []
        rules = self.validation_rules.get(section_name, {})
        
        for key, value in section_config.items():
            if key in rules:
                rule = rules[key]
                if isinstance(rule, list):
                    if value not in rule:
                        errors.append(f"{section_name}.{key}: {value} not in allowed values {rule}")
                elif isinstance(rule, tuple) and len(rule) == 2:
                    if not (rule[0] <= value <= rule[1]):
                        errors.append(f"{section_name}.{key}: {value} not in range {rule}")
                        
        return errors
        
    def _validate_logical_consistency(self, config: TechStackConfig) -> List[str]:
        """Validate logical consistency between configuration sections."""
        errors = []
        
        # Check if advanced augmentation is used with appropriate training epochs
        aug_config = config.augmentation_config.get('advanced', {})
        training_config = config.training_config
        
        advanced_aug_used = any([
            aug_config.get('mosaic_prob', 0) > 0,
            aug_config.get('copy_paste_prob', 0) > 0,
            aug_config.get('mixup_prob', 0) > 0
        ])
        
        if advanced_aug_used and training_config.get('epochs', 0) < 50:
            errors.append("Advanced augmentation requires at least 50 training epochs")
            
        # Check if large models use appropriate batch sizes
        backbone = config.model_config.get('backbone', '')
        batch_size = training_config.get('batch_size', 16)
        
        if backbone in ['yolov8l', 'yolov8x'] and batch_size > 16:
            errors.append(f"Large model {backbone} should use batch_size <= 16")
            
        # Check if TTA is used with appropriate confidence threshold
        inference_config = config.inference_config
        if inference_config.get('use_tta', False):
            conf_thresh = inference_config.get('confidence_threshold', 0.25)
            if conf_thresh > 0.2:
                errors.append("TTA should use lower confidence threshold (<= 0.2)")
                
        return errors
        
    def get_optimization_suggestions(self, config: TechStackConfig, 
                                   target_metric: str = 'map') -> List[str]:
        """
        Get optimization suggestions for a configuration.
        
        Args:
            config: Configuration to analyze
            target_metric: Target metric to optimize ('map', 'fps', 'memory')
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        if target_metric == 'map':
            suggestions.extend(self._get_accuracy_suggestions(config))
        elif target_metric == 'fps':
            suggestions.extend(self._get_speed_suggestions(config))
        elif target_metric == 'memory':
            suggestions.extend(self._get_memory_suggestions(config))
            
        return suggestions
        
    def _get_accuracy_suggestions(self, config: TechStackConfig) -> List[str]:
        """Get suggestions to improve accuracy."""
        suggestions = []
        
        # Model suggestions
        backbone = config.model_config.get('backbone', '')
        if backbone in ['yolov8n', 'yolov8s']:
            suggestions.append("考虑使用更大的模型 (yolov8m/l) 提升精度")
            
        attention = config.model_config.get('attention')
        if attention != 'cbam':
            suggestions.append("使用CBAM注意力机制可能提升小目标检测精度")
            
        # Loss suggestions
        loss_type = config.loss_config.get('classification', '')
        if loss_type != 'combo':
            suggestions.append("使用组合损失函数 (combo) 可能提升整体性能")
            
        # Augmentation suggestions
        aug_config = config.augmentation_config.get('advanced', {})
        if aug_config.get('mosaic_prob', 0) < 0.3:
            suggestions.append("增加Mosaic增强概率到0.3-0.5可能提升小目标检测")
            
        # Training suggestions
        epochs = config.training_config.get('epochs', 0)
        if epochs < 200:
            suggestions.append("增加训练轮数到200+可能提升模型收敛")
            
        return suggestions
        
    def _get_speed_suggestions(self, config: TechStackConfig) -> List[str]:
        """Get suggestions to improve inference speed."""
        suggestions = []
        
        # Model suggestions
        backbone = config.model_config.get('backbone', '')
        if backbone in ['yolov8l', 'yolov8x']:
            suggestions.append("使用更小的模型 (yolov8n/s) 提升推理速度")
            
        attention = config.model_config.get('attention')
        if attention == 'cbam':
            suggestions.append("使用SE或ECA注意力机制减少计算开销")
            
        # Inference suggestions
        if config.inference_config.get('use_tta', False):
            suggestions.append("禁用TTA (测试时增强) 可显著提升推理速度")
            
        return suggestions
        
    def _get_memory_suggestions(self, config: TechStackConfig) -> List[str]:
        """Get suggestions to reduce memory usage."""
        suggestions = []
        
        # Training suggestions
        batch_size = config.training_config.get('batch_size', 16)
        if batch_size > 8:
            suggestions.append("减少batch_size到8或更小以降低内存使用")
            
        # Model suggestions
        backbone = config.model_config.get('backbone', '')
        if backbone in ['yolov8l', 'yolov8x']:
            suggestions.append("使用更小的模型减少内存占用")
            
        return suggestions
        
    def create_custom_config(self, base_config: str, modifications: Dict[str, Any],
                           name: str, description: str) -> TechStackConfig:
        """
        Create a custom configuration based on an existing one.
        
        Args:
            base_config: Name of base configuration
            modifications: Modifications to apply
            name: Name for the new configuration
            description: Description of the new configuration
            
        Returns:
            New technology stack configuration
        """
        base = self.configs.get(base_config)
        if not base:
            raise ValueError(f"Base configuration '{base_config}' not found")
            
        # Deep copy base configuration
        new_config_dict = asdict(base)
        new_config_dict['name'] = name
        new_config_dict['description'] = description
        
        # Apply modifications
        new_config_dict = self._merge_configs(new_config_dict, modifications)
        
        # Create new configuration
        custom_config = TechStackConfig(**new_config_dict)
        
        # Validate the new configuration
        is_valid, errors = self.validate_config(custom_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
            
        return custom_config
        
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def save_config(self, config: TechStackConfig, filepath: str) -> None:
        """
        Save a configuration to file.
        
        Args:
            config: Configuration to save
            filepath: Path to save the configuration
        """
        config_dict = asdict(config)
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError("Unsupported file format. Use .yaml or .json")
            
    def compare_configs(self, config1_name: str, config2_name: str) -> Dict[str, Any]:
        """
        Compare two configurations and highlight differences.
        
        Args:
            config1_name: Name of first configuration
            config2_name: Name of second configuration
            
        Returns:
            Dictionary containing comparison results
        """
        config1 = self.configs.get(config1_name)
        config2 = self.configs.get(config2_name)
        
        if not config1 or not config2:
            raise ValueError("One or both configurations not found")
            
        comparison = {
            'config1': config1_name,
            'config2': config2_name,
            'differences': {},
            'performance_comparison': {
                'config1_performance': config1.expected_performance,
                'config2_performance': config2.expected_performance
            }
        }
        
        # Compare each section
        sections = ['model_config', 'loss_config', 'augmentation_config', 
                   'training_config', 'inference_config']
        
        for section in sections:
            section1 = getattr(config1, section)
            section2 = getattr(config2, section)
            
            diff = self._find_differences(section1, section2)
            if diff:
                comparison['differences'][section] = diff
                
        return comparison
        
    def _find_differences(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Find differences between two dictionaries."""
        differences = {}
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if val1 != val2:
                if isinstance(val1, dict) and isinstance(val2, dict):
                    nested_diff = self._find_differences(val1, val2)
                    if nested_diff:
                        differences[key] = nested_diff
                else:
                    differences[key] = {'config1': val1, 'config2': val2}
                    
        return differences
        
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all available configurations.
        
        Returns:
            Summary of configurations with key characteristics
        """
        summary = {}
        
        for name, config in self.configs.items():
            summary[name] = {
                'description': config.description,
                'backbone': config.model_config.get('backbone', 'unknown'),
                'attention': config.model_config.get('attention', 'none'),
                'expected_map': config.expected_performance.get('map', 'unknown'),
                'expected_fps': config.expected_performance.get('fps', 'unknown'),
                'gpu_memory': config.resource_requirements.get('gpu_memory', 'unknown'),
                'training_time': config.resource_requirements.get('training_time', 'unknown')
            }
            
        return summary