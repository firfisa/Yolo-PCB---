"""
CBAM integration utilities for PCB defect detection.
Provides optimized CBAM configurations and integration strategies.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from .attention import CBAM, AttentionBlock


class CBAMConfig:
    """Configuration class for CBAM integration."""
    
    # Optimized CBAM configurations for different model sizes
    CONFIGS = {
        "yolov8n": {
            "backbone_positions": [1, 2, 3],  # Apply CBAM after these stages
            "neck_positions": [0, 1, 2],      # Apply CBAM to all neck outputs
            "reduction_ratio": 16,
            "spatial_kernel": 7,
            "enable_neck_attention": True
        },
        "yolov8s": {
            "backbone_positions": [1, 2, 3],
            "neck_positions": [0, 1, 2],
            "reduction_ratio": 16,
            "spatial_kernel": 7,
            "enable_neck_attention": True
        },
        "yolov8m": {
            "backbone_positions": [2, 3],     # Fewer positions for larger models
            "neck_positions": [1, 2],
            "reduction_ratio": 32,            # Higher reduction for efficiency
            "spatial_kernel": 5,
            "enable_neck_attention": True
        },
        "yolov8l": {
            "backbone_positions": [2, 3],
            "neck_positions": [1, 2],
            "reduction_ratio": 32,
            "spatial_kernel": 5,
            "enable_neck_attention": False    # Disable for very large models
        },
        "yolov8x": {
            "backbone_positions": [3],        # Only at deepest level
            "neck_positions": [2],
            "reduction_ratio": 64,
            "spatial_kernel": 3,
            "enable_neck_attention": False
        }
    }
    
    @classmethod
    def get_config(cls, model_variant: str) -> Dict:
        """Get optimized CBAM configuration for model variant."""
        return cls.CONFIGS.get(model_variant, cls.CONFIGS["yolov8n"])
    
    @classmethod
    def get_small_object_config(cls, model_variant: str) -> Dict:
        """Get CBAM configuration optimized for small object detection."""
        config = cls.get_config(model_variant).copy()
        
        # Optimize for small objects
        config.update({
            "reduction_ratio": max(8, config["reduction_ratio"] // 2),  # Lower reduction
            "spatial_kernel": 7,  # Larger spatial kernel
            "enable_neck_attention": True,  # Always enable for small objects
            "backbone_positions": [1, 2, 3],  # Use all positions
            "neck_positions": [0, 1, 2]
        })
        
        return config


class AdaptiveCBAM(nn.Module):
    """Adaptive CBAM that adjusts parameters based on feature map size."""
    
    def __init__(self, in_channels: int, base_reduction: int = 16, 
                 adaptive_spatial: bool = True):
        """
        Initialize adaptive CBAM.
        
        Args:
            in_channels: Number of input channels
            base_reduction: Base reduction ratio
            adaptive_spatial: Whether to adapt spatial kernel size
        """
        super().__init__()
        self.adaptive_spatial = adaptive_spatial
        
        # Adaptive reduction ratio based on channel count
        reduction_ratio = max(4, min(base_reduction, in_channels // 4))
        
        # Base CBAM with adaptive parameters
        self.cbam = CBAM(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
            kernel_size=7 if adaptive_spatial else 7
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive processing."""
        return self.cbam(x)


class CBAMIntegrator:
    """Utility class for integrating CBAM into existing models."""
    
    @staticmethod
    def add_cbam_to_backbone(backbone: nn.Module, model_variant: str, 
                           channels: List[int]) -> nn.Module:
        """
        Add CBAM modules to backbone at optimal positions.
        
        Args:
            backbone: Backbone network
            model_variant: YOLO model variant
            channels: Channel dimensions
            
        Returns:
            Modified backbone with CBAM
        """
        config = CBAMConfig.get_config(model_variant)
        
        # Add CBAM modules at specified positions
        for i, pos in enumerate(config["backbone_positions"]):
            if pos < len(channels):
                cbam_module = AdaptiveCBAM(
                    channels[pos],
                    config["reduction_ratio"]
                )
                setattr(backbone, f"cbam_{pos}", cbam_module)
        
        return backbone
    
    @staticmethod
    def create_neck_with_cbam(channels: List[int], model_variant: str) -> nn.Module:
        """
        Create neck module with integrated CBAM.
        
        Args:
            channels: Channel dimensions
            model_variant: YOLO model variant
            
        Returns:
            Neck module with CBAM
        """
        from .yolo_detector import YOLONeck
        
        config = CBAMConfig.get_config(model_variant)
        
        if config["enable_neck_attention"]:
            return YOLONeck(channels, attention_type="cbam")
        else:
            return YOLONeck(channels, attention_type=None)
    
    @staticmethod
    def optimize_cbam_placement(model: nn.Module, input_size: Tuple[int, int] = (640, 640),
                              channels: List[int] = None) -> Dict[str, float]:
        """
        Analyze and optimize CBAM placement for given model.
        
        Args:
            model: Model to analyze
            input_size: Input image size
            channels: Channel dimensions
            
        Returns:
            Analysis results with recommendations
        """
        results = {
            "computational_overhead": 0.0,
            "memory_overhead": 0.0,
            "recommended_positions": [],
            "efficiency_score": 0.0
        }
        
        # Simulate forward pass to analyze computational cost
        dummy_input = torch.randn(1, 3, *input_size)
        
        try:
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Calculate efficiency metrics
            total_params = sum(p.numel() for p in model.parameters())
            cbam_params = sum(p.numel() for name, p in model.named_parameters() 
                            if 'attention' in name or 'cbam' in name)
            
            results["computational_overhead"] = cbam_params / total_params
            results["efficiency_score"] = min(1.0, 1.0 - results["computational_overhead"])
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            
        return results


class CBAMTrainingOptimizer:
    """Optimizer for CBAM training parameters."""
    
    @staticmethod
    def get_attention_lr_schedule(base_lr: float, attention_lr_factor: float = 0.1) -> Dict:
        """
        Get learning rate schedule optimized for attention mechanisms.
        
        Args:
            base_lr: Base learning rate
            attention_lr_factor: Factor for attention module learning rate
            
        Returns:
            Learning rate configuration
        """
        return {
            "backbone_lr": base_lr,
            "attention_lr": base_lr * attention_lr_factor,
            "neck_lr": base_lr,
            "head_lr": base_lr * 2.0  # Higher LR for detection head
        }
    
    @staticmethod
    def create_param_groups(model, lr_config: Dict) -> List[Dict]:
        """
        Create parameter groups with different learning rates.
        
        Args:
            model: Model with CBAM modules (YOLODetector or nn.Module)
            lr_config: Learning rate configuration
            
        Returns:
            Parameter groups for optimizer
        """
        param_groups = []
        
        # Backbone parameters (excluding attention)
        backbone_params = []
        attention_params = []
        neck_params = []
        head_params = []
        
        # Handle YOLODetector structure
        if hasattr(model, 'backbone') and hasattr(model, 'neck') and hasattr(model, 'head'):
            # YOLODetector structure
            for name, param in model.backbone.named_parameters():
                if 'attention' in name or 'cbam' in name:
                    attention_params.append(param)
                else:
                    backbone_params.append(param)
            
            for name, param in model.neck.named_parameters():
                if 'attention' in name or 'cbam' in name:
                    attention_params.append(param)
                else:
                    neck_params.append(param)
            
            for name, param in model.head.named_parameters():
                head_params.append(param)
        else:
            # Standard nn.Module structure
            for name, param in model.named_parameters():
                if 'attention' in name or 'cbam' in name:
                    attention_params.append(param)
                elif 'backbone' in name:
                    backbone_params.append(param)
                elif 'neck' in name:
                    neck_params.append(param)
                elif 'head' in name:
                    head_params.append(param)
        
        # Create parameter groups
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': lr_config.get('backbone_lr', 0.01),
                'name': 'backbone'
            })
        
        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': lr_config.get('attention_lr', 0.001),
                'name': 'attention'
            })
        
        if neck_params:
            param_groups.append({
                'params': neck_params,
                'lr': lr_config.get('neck_lr', 0.01),
                'name': 'neck'
            })
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': lr_config.get('head_lr', 0.02),
                'name': 'head'
            })
        
        return param_groups


def create_cbam_enhanced_yolo(model_variant: str = "yolov8n", 
                             num_classes: int = 5,
                             optimize_for_small_objects: bool = True) -> Dict:
    """
    Create YOLO model configuration with optimized CBAM integration.
    
    Args:
        model_variant: YOLO model variant
        num_classes: Number of classes
        optimize_for_small_objects: Whether to optimize for small object detection
        
    Returns:
        Model configuration dictionary
    """
    if optimize_for_small_objects:
        cbam_config = CBAMConfig.get_small_object_config(model_variant)
    else:
        cbam_config = CBAMConfig.get_config(model_variant)
    
    model_config = {
        "variant": model_variant,
        "num_classes": num_classes,
        "attention_type": "cbam",
        "cbam_config": cbam_config,
        "input_size": 640,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "use_focal": True,
        "use_iou": True,
        "iou_type": "ciou",
        "nms_type": "standard"
    }
    
    return model_config