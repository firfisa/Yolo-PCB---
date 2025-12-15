"""
Configuration Optimizer for PCB Defect Detection System.

This module provides automatic configuration optimization and validation
based on hardware constraints and performance targets.
"""

import psutil
import torch
from typing import Dict, Any, List, Tuple, Optional
import warnings
from dataclasses import dataclass

from .tech_config_manager import TechStackConfig


@dataclass
class HardwareInfo:
    """Hardware information for configuration optimization."""
    gpu_available: bool
    gpu_memory_gb: float
    cpu_cores: int
    ram_gb: float
    gpu_name: Optional[str] = None


@dataclass
class PerformanceTarget:
    """Performance targets for optimization."""
    target_map: float = 0.3
    max_training_time_hours: float = 8.0
    target_fps: float = 30.0
    max_gpu_memory_gb: float = 4.0
    priority: str = "balanced"  # "speed", "accuracy", "balanced"


class ConfigOptimizer:
    """Optimizer for technology configurations based on hardware and targets."""
    
    def __init__(self):
        """Initialize the configuration optimizer."""
        self.hardware_info = self._detect_hardware()
        
    def _detect_hardware(self) -> HardwareInfo:
        """Detect available hardware resources."""
        # GPU detection
        gpu_available = torch.cuda.is_available()
        gpu_memory_gb = 0.0
        gpu_name = None
        
        if gpu_available:
            try:
                gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory_bytes / (1024**3)
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_available = False
                
        # CPU and RAM detection
        cpu_cores = psutil.cpu_count(logical=False)
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024**3)
        
        return HardwareInfo(
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_name=gpu_name
        )
        
    def get_hardware_info(self) -> HardwareInfo:
        """Get detected hardware information."""
        return self.hardware_info
        
    def optimize_config(self, base_config: TechStackConfig, 
                       target: PerformanceTarget) -> TechStackConfig:
        """
        Optimize a configuration based on hardware and performance targets.
        
        Args:
            base_config: Base configuration to optimize
            target: Performance targets
            
        Returns:
            Optimized configuration
        """
        # Create a copy of the base configuration
        optimized = TechStackConfig(
            name=f"{base_config.name}_optimized",
            description=f"{base_config.description} (优化版)",
            model_config=base_config.model_config.copy(),
            loss_config=base_config.loss_config.copy(),
            augmentation_config=base_config.augmentation_config.copy(),
            training_config=base_config.training_config.copy(),
            inference_config=base_config.inference_config.copy(),
            expected_performance=base_config.expected_performance.copy(),
            resource_requirements=base_config.resource_requirements.copy()
        )
        
        # Apply hardware-based optimizations
        optimized = self._optimize_for_hardware(optimized)
        
        # Apply target-based optimizations
        optimized = self._optimize_for_targets(optimized, target)
        
        return optimized
        
    def _optimize_for_hardware(self, config: TechStackConfig) -> TechStackConfig:
        """Optimize configuration based on available hardware."""
        
        # GPU memory optimization
        if self.hardware_info.gpu_available:
            if self.hardware_info.gpu_memory_gb < 4:
                # Low GPU memory - use smaller models and batch sizes
                if config.model_config.get('backbone') in ['yolov8l', 'yolov8x']:
                    config.model_config['backbone'] = 'yolov8m'
                    warnings.warn("Reduced model size due to limited GPU memory")
                    
                if config.training_config.get('batch_size', 16) > 8:
                    config.training_config['batch_size'] = 8
                    warnings.warn("Reduced batch size due to limited GPU memory")
                    
            elif self.hardware_info.gpu_memory_gb > 8:
                # High GPU memory - can use larger models and batch sizes
                if config.model_config.get('backbone') == 'yolov8n':
                    config.model_config['backbone'] = 'yolov8s'
                    
                if config.training_config.get('batch_size', 16) < 32:
                    config.training_config['batch_size'] = min(32, 
                        config.training_config.get('batch_size', 16) * 2)
        else:
            # No GPU - optimize for CPU
            config.model_config['backbone'] = 'yolov8n'  # Smallest model
            config.training_config['batch_size'] = 4  # Small batch size
            config.training_config['workers'] = min(4, self.hardware_info.cpu_cores)
            warnings.warn("No GPU detected - using CPU-optimized configuration")
            
        # CPU cores optimization
        config.training_config['workers'] = min(
            config.training_config.get('workers', 8),
            self.hardware_info.cpu_cores
        )
        
        return config
        
    def _optimize_for_targets(self, config: TechStackConfig, 
                            target: PerformanceTarget) -> TechStackConfig:
        """Optimize configuration based on performance targets."""
        
        if target.priority == "speed":
            config = self._optimize_for_speed(config, target)
        elif target.priority == "accuracy":
            config = self._optimize_for_accuracy(config, target)
        else:  # balanced
            config = self._optimize_for_balance(config, target)
            
        return config
        
    def _optimize_for_speed(self, config: TechStackConfig, 
                          target: PerformanceTarget) -> TechStackConfig:
        """Optimize configuration for inference speed."""
        
        # Use smaller, faster models
        if config.model_config.get('backbone') not in ['yolov8n', 'yolov8s']:
            config.model_config['backbone'] = 'yolov8s'
            
        # Use lighter attention mechanisms
        if config.model_config.get('attention') == 'cbam':
            config.model_config['attention'] = 'se'
            
        # Disable expensive features
        config.model_config['use_fpn'] = False
        config.model_config['use_panet'] = False
        
        # Optimize inference settings
        config.inference_config['use_tta'] = False
        config.inference_config['confidence_threshold'] = 0.3  # Higher threshold
        
        # Reduce augmentation for faster training
        aug_config = config.augmentation_config.get('advanced', {})
        aug_config['mosaic_prob'] = 0.0
        aug_config['copy_paste_prob'] = 0.0
        aug_config['mixup_prob'] = 0.0
        
        return config
        
    def _optimize_for_accuracy(self, config: TechStackConfig, 
                             target: PerformanceTarget) -> TechStackConfig:
        """Optimize configuration for accuracy."""
        
        # Use larger, more accurate models if hardware allows
        if (self.hardware_info.gpu_memory_gb > 6 and 
            config.model_config.get('backbone') in ['yolov8n', 'yolov8s']):
            config.model_config['backbone'] = 'yolov8m'
            
        # Use best attention mechanism
        config.model_config['attention'] = 'cbam'
        config.model_config['use_fpn'] = True
        
        # Use advanced loss functions
        config.loss_config['classification'] = 'combo'
        config.loss_config['bbox_regression'] = 'ciou'
        
        # Increase augmentation
        aug_config = config.augmentation_config.get('advanced', {})
        aug_config['mosaic_prob'] = 0.5
        aug_config['copy_paste_prob'] = 0.3
        aug_config['mixup_prob'] = 0.2
        
        # Optimize inference for accuracy
        config.inference_config['confidence_threshold'] = 0.15  # Lower threshold
        config.inference_config['use_tta'] = True
        
        # Increase training epochs if time allows
        if target.max_training_time_hours > 8:
            config.training_config['epochs'] = min(300, 
                int(target.max_training_time_hours * 25))
            
        return config
        
    def _optimize_for_balance(self, config: TechStackConfig, 
                            target: PerformanceTarget) -> TechStackConfig:
        """Optimize configuration for balanced performance."""
        
        # Choose medium-sized model
        if config.model_config.get('backbone') in ['yolov8n', 'yolov8x']:
            config.model_config['backbone'] = 'yolov8m'
            
        # Use CBAM attention if hardware allows
        if self.hardware_info.gpu_memory_gb > 4:
            config.model_config['attention'] = 'cbam'
        else:
            config.model_config['attention'] = 'se'
            
        # Moderate augmentation
        aug_config = config.augmentation_config.get('advanced', {})
        aug_config['mosaic_prob'] = 0.3
        aug_config['copy_paste_prob'] = 0.2
        aug_config['mixup_prob'] = 0.1
        
        # Balanced loss configuration
        config.loss_config['classification'] = 'focal'
        config.loss_config['bbox_regression'] = 'ciou'
        
        return config
        
    def validate_config_feasibility(self, config: TechStackConfig) -> Tuple[bool, List[str]]:
        """
        Validate if a configuration is feasible with current hardware.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_feasible, warning_messages)
        """
        warnings_list = []
        is_feasible = True
        
        # Check GPU requirements
        if not self.hardware_info.gpu_available:
            backbone = config.model_config.get('backbone', '')
            if backbone in ['yolov8l', 'yolov8x']:
                warnings_list.append(
                    f"Large model {backbone} may be very slow without GPU"
                )
                
        # Check GPU memory requirements
        if self.hardware_info.gpu_available:
            estimated_memory = self._estimate_gpu_memory_usage(config)
            if estimated_memory > self.hardware_info.gpu_memory_gb:
                warnings_list.append(
                    f"Estimated GPU memory usage ({estimated_memory:.1f}GB) "
                    f"exceeds available memory ({self.hardware_info.gpu_memory_gb:.1f}GB)"
                )
                is_feasible = False
                
        # Check training time estimates
        estimated_time = self._estimate_training_time(config)
        if estimated_time > 24:  # More than 24 hours
            warnings_list.append(
                f"Estimated training time ({estimated_time:.1f}h) is very long"
            )
            
        return is_feasible, warnings_list
        
    def _estimate_gpu_memory_usage(self, config: TechStackConfig) -> float:
        """Estimate GPU memory usage for a configuration."""
        
        # Base memory usage by model size
        backbone = config.model_config.get('backbone', 'yolov8n')
        base_memory = {
            'yolov8n': 1.5,
            'yolov8s': 2.0,
            'yolov8m': 3.0,
            'yolov8l': 4.5,
            'yolov8x': 6.0
        }.get(backbone, 2.0)
        
        # Batch size multiplier
        batch_size = config.training_config.get('batch_size', 16)
        memory_usage = base_memory * (batch_size / 16)
        
        # Additional memory for attention mechanisms
        attention = config.model_config.get('attention')
        if attention == 'cbam':
            memory_usage *= 1.2
        elif attention in ['se', 'eca', 'coord']:
            memory_usage *= 1.1
            
        # Additional memory for FPN/PANet
        if config.model_config.get('use_fpn', False):
            memory_usage *= 1.15
        if config.model_config.get('use_panet', False):
            memory_usage *= 1.1
            
        return memory_usage
        
    def _estimate_training_time(self, config: TechStackConfig) -> float:
        """Estimate training time in hours for a configuration."""
        
        # Base time per epoch (minutes)
        backbone = config.model_config.get('backbone', 'yolov8n')
        base_time_per_epoch = {
            'yolov8n': 2,
            'yolov8s': 3,
            'yolov8m': 5,
            'yolov8l': 8,
            'yolov8x': 12
        }.get(backbone, 3)
        
        # Adjust for hardware
        if not self.hardware_info.gpu_available:
            base_time_per_epoch *= 10  # Much slower on CPU
        elif self.hardware_info.gpu_memory_gb < 4:
            base_time_per_epoch *= 1.5  # Slower with limited memory
            
        # Adjust for batch size
        batch_size = config.training_config.get('batch_size', 16)
        time_multiplier = 16 / batch_size  # Smaller batches take longer
        
        # Adjust for augmentation
        aug_config = config.augmentation_config.get('advanced', {})
        if any([aug_config.get('mosaic_prob', 0) > 0,
                aug_config.get('copy_paste_prob', 0) > 0,
                aug_config.get('mixup_prob', 0) > 0]):
            time_multiplier *= 1.3
            
        epochs = config.training_config.get('epochs', 100)
        total_time_minutes = base_time_per_epoch * epochs * time_multiplier
        
        return total_time_minutes / 60  # Convert to hours
        
    def get_optimization_report(self, original_config: TechStackConfig,
                              optimized_config: TechStackConfig) -> Dict[str, Any]:
        """
        Generate a report comparing original and optimized configurations.
        
        Args:
            original_config: Original configuration
            optimized_config: Optimized configuration
            
        Returns:
            Optimization report
        """
        report = {
            'hardware_info': {
                'gpu_available': self.hardware_info.gpu_available,
                'gpu_memory_gb': self.hardware_info.gpu_memory_gb,
                'gpu_name': self.hardware_info.gpu_name,
                'cpu_cores': self.hardware_info.cpu_cores,
                'ram_gb': self.hardware_info.ram_gb
            },
            'changes': {},
            'performance_estimates': {
                'original': {
                    'gpu_memory_usage': self._estimate_gpu_memory_usage(original_config),
                    'training_time_hours': self._estimate_training_time(original_config)
                },
                'optimized': {
                    'gpu_memory_usage': self._estimate_gpu_memory_usage(optimized_config),
                    'training_time_hours': self._estimate_training_time(optimized_config)
                }
            }
        }
        
        # Find changes between configurations
        sections = ['model_config', 'loss_config', 'augmentation_config', 
                   'training_config', 'inference_config']
        
        for section in sections:
            original_section = getattr(original_config, section)
            optimized_section = getattr(optimized_config, section)
            
            changes = self._find_config_changes(original_section, optimized_section)
            if changes:
                report['changes'][section] = changes
                
        return report
        
    def _find_config_changes(self, original: Dict[str, Any], 
                           optimized: Dict[str, Any]) -> Dict[str, Any]:
        """Find changes between two configuration dictionaries."""
        changes = {}
        
        all_keys = set(original.keys()) | set(optimized.keys())
        
        for key in all_keys:
            orig_val = original.get(key)
            opt_val = optimized.get(key)
            
            if orig_val != opt_val:
                if isinstance(orig_val, dict) and isinstance(opt_val, dict):
                    nested_changes = self._find_config_changes(orig_val, opt_val)
                    if nested_changes:
                        changes[key] = nested_changes
                else:
                    changes[key] = {
                        'original': orig_val,
                        'optimized': opt_val
                    }
                    
        return changes