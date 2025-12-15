"""
Advanced loss function combinations for PCB defect detection.
Implements sophisticated loss strategies optimized for small object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from .losses import FocalLoss, IoULoss, ClassBalancedLoss


class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss that adjusts parameters based on training progress."""
    
    def __init__(self, alpha: float = 1.0, gamma_init: float = 2.0, 
                 gamma_final: float = 0.5, total_epochs: int = 300):
        """
        Initialize Adaptive Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma_init: Initial focusing parameter
            gamma_final: Final focusing parameter
            total_epochs: Total training epochs for adaptation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set current epoch for adaptive gamma calculation."""
        self.current_epoch = epoch
        
    def get_current_gamma(self) -> float:
        """Calculate current gamma based on training progress."""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        # Gradually reduce gamma from init to final
        gamma = self.gamma_init - (self.gamma_init - self.gamma_final) * progress
        return gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive gamma."""
        gamma = self.get_current_gamma()
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss for joint classification and localization quality estimation."""
    
    def __init__(self, beta: float = 2.0):
        """
        Initialize Quality Focal Loss.
        
        Args:
            beta: Focusing parameter for quality estimation
        """
        super().__init__()
        self.beta = beta
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor,
                pred_quality: torch.Tensor, target_quality: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Quality Focal Loss.
        
        Args:
            pred_scores: Predicted classification scores [N, C]
            target_scores: Target classification scores [N, C]
            pred_quality: Predicted localization quality (IoU) [N]
            target_quality: Target localization quality [N]
            
        Returns:
            Quality focal loss
        """
        # Classification component
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_scores, target_scores, reduction='none'
        )
        
        # Quality weighting - expand to match classification dimensions
        quality_weight = torch.abs(target_quality - torch.sigmoid(pred_quality)) ** self.beta
        quality_weight = quality_weight.unsqueeze(1).expand_as(cls_loss)
        
        # Weighted loss
        qfl_loss = quality_weight * cls_loss
        
        return qfl_loss.mean()


class VarifocalLoss(nn.Module):
    """Varifocal Loss for dense object detection with quality prediction."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        """
        Initialize Varifocal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor,
                target_quality: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Varifocal Loss.
        
        Args:
            pred_scores: Predicted classification scores
            target_scores: Target classification scores (0 or 1)
            target_quality: Target quality scores (IoU for positive samples)
            
        Returns:
            Varifocal loss
        """
        pred_sigmoid = torch.sigmoid(pred_scores)
        
        # For positive samples, target is quality score
        # For negative samples, target is 0
        # Expand target_quality to match target_scores dimensions
        if target_quality.dim() == 1 and target_scores.dim() == 2:
            target_quality = target_quality.unsqueeze(1).expand_as(target_scores)
        target_weighted = target_scores * target_quality
        
        # Focal weight calculation
        focal_weight = target_weighted * (target_weighted - pred_sigmoid).abs() ** self.gamma + \
                      (1 - target_scores) * pred_sigmoid ** self.gamma
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores, target_weighted, reduction='none'
        )
        
        # Varifocal loss
        vfl_loss = focal_weight * bce_loss
        
        return vfl_loss.mean()


class SmallObjectLoss(nn.Module):
    """Specialized loss for small object detection in PCB defects."""
    
    def __init__(self, small_threshold: float = 32.0, scale_factor: float = 2.0):
        """
        Initialize Small Object Loss.
        
        Args:
            small_threshold: Threshold for considering objects as small (in pixels)
            scale_factor: Scaling factor for small object loss
        """
        super().__init__()
        self.small_threshold = small_threshold
        self.scale_factor = scale_factor
        
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Forward pass with small object emphasis.
        
        Args:
            pred_boxes: Predicted bounding boxes [N, 4]
            target_boxes: Target bounding boxes [N, 4]
            image_size: Image size (H, W)
            
        Returns:
            Small object weighted loss
        """
        # Calculate target box areas
        target_w = (target_boxes[:, 2] - target_boxes[:, 0]) * image_size[1]
        target_h = (target_boxes[:, 3] - target_boxes[:, 1]) * image_size[0]
        target_areas = target_w * target_h
        
        # Identify small objects
        small_mask = target_areas < (self.small_threshold ** 2)
        
        # Calculate IoU loss
        iou_loss = IoULoss(loss_type='ciou')
        base_loss = iou_loss(pred_boxes, target_boxes)
        
        # Apply scaling for small objects
        loss_weights = torch.ones_like(base_loss)
        loss_weights[small_mask] *= self.scale_factor
        
        weighted_loss = base_loss * loss_weights
        
        return weighted_loss.mean()


class DynamicLossWeighting(nn.Module):
    """Dynamic loss weighting based on training progress and performance."""
    
    def __init__(self, initial_weights: Dict[str, float], 
                 adaptation_rate: float = 0.1):
        """
        Initialize Dynamic Loss Weighting.
        
        Args:
            initial_weights: Initial loss weights
            adaptation_rate: Rate of weight adaptation
        """
        super().__init__()
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight, dtype=torch.float32))
            for name, weight in initial_weights.items()
        })
        self.adaptation_rate = adaptation_rate
        self.loss_history = {name: [] for name in initial_weights.keys()}
        
    def update_weights(self, loss_values: Dict[str, float]):
        """
        Update loss weights based on recent performance.
        
        Args:
            loss_values: Current loss values for each component
        """
        # Store loss history
        for name, value in loss_values.items():
            if name in self.loss_history:
                self.loss_history[name].append(value)
                # Keep only recent history
                if len(self.loss_history[name]) > 10:
                    self.loss_history[name] = self.loss_history[name][-10:]
        
        # Adapt weights based on loss trends
        for name in self.weights.keys():
            if len(self.loss_history[name]) >= 5:
                recent_losses = self.loss_history[name][-5:]
                trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                
                # If loss is increasing, increase weight
                if trend > 0:
                    self.weights[name].data *= (1 + self.adaptation_rate)
                # If loss is decreasing rapidly, decrease weight slightly
                elif trend < -0.01:
                    self.weights[name].data *= (1 - self.adaptation_rate * 0.5)
                
                # Clamp weights to reasonable range
                self.weights[name].data.clamp_(0.1, 5.0)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return {name: weight.item() for name, weight in self.weights.items()}


class AdvancedComboLoss(nn.Module):
    """Advanced combination loss with multiple sophisticated loss functions."""
    
    def __init__(self, config: Dict):
        """
        Initialize Advanced Combo Loss.
        
        Args:
            config: Loss configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Initialize loss components
        self._init_classification_loss()
        self._init_regression_loss()
        self._init_quality_loss()
        self._init_weighting_strategy()
        
        # Loss tracking
        self.loss_history = []
        
    def _init_classification_loss(self):
        """Initialize classification loss components."""
        cls_config = self.config.get('classification', {})
        cls_type = cls_config.get('type', 'adaptive_focal')
        
        if cls_type == 'adaptive_focal':
            self.cls_criterion = AdaptiveFocalLoss(
                alpha=cls_config.get('alpha', 1.0),
                gamma_init=cls_config.get('gamma_init', 2.0),
                gamma_final=cls_config.get('gamma_final', 0.5),
                total_epochs=cls_config.get('total_epochs', 300)
            )
        elif cls_type == 'focal':
            self.cls_criterion = FocalLoss(
                alpha=cls_config.get('alpha', 1.0),
                gamma=cls_config.get('gamma', 2.0)
            )
        elif cls_type == 'varifocal':
            self.cls_criterion = VarifocalLoss(
                alpha=cls_config.get('alpha', 0.75),
                gamma=cls_config.get('gamma', 2.0)
            )
        else:
            self.cls_criterion = nn.CrossEntropyLoss()
    
    def _init_regression_loss(self):
        """Initialize regression loss components."""
        reg_config = self.config.get('regression', {})
        reg_type = reg_config.get('type', 'ciou')
        
        if reg_type in ['iou', 'giou', 'diou', 'ciou']:
            self.reg_criterion = IoULoss(loss_type=reg_type)
        elif reg_type == 'small_object':
            self.reg_criterion = SmallObjectLoss(
                small_threshold=reg_config.get('small_threshold', 32.0),
                scale_factor=reg_config.get('scale_factor', 2.0)
            )
        else:
            self.reg_criterion = nn.SmoothL1Loss()
    
    def _init_quality_loss(self):
        """Initialize quality estimation loss."""
        quality_config = self.config.get('quality', {})
        
        if quality_config.get('enable', False):
            quality_type = quality_config.get('type', 'quality_focal')
            
            if quality_type == 'quality_focal':
                self.quality_criterion = QualityFocalLoss(
                    beta=quality_config.get('beta', 2.0)
                )
            else:
                self.quality_criterion = nn.MSELoss()
        else:
            self.quality_criterion = None
    
    def _init_weighting_strategy(self):
        """Initialize loss weighting strategy."""
        weight_config = self.config.get('weighting', {})
        
        if weight_config.get('dynamic', False):
            initial_weights = {
                'classification': weight_config.get('cls_weight', 1.0),
                'regression': weight_config.get('reg_weight', 1.0),
                'objectness': weight_config.get('obj_weight', 1.0)
            }
            
            if self.quality_criterion is not None:
                initial_weights['quality'] = weight_config.get('quality_weight', 0.5)
            
            self.dynamic_weighting = DynamicLossWeighting(
                initial_weights=initial_weights,
                adaptation_rate=weight_config.get('adaptation_rate', 0.1)
            )
        else:
            self.dynamic_weighting = None
            self.static_weights = {
                'classification': weight_config.get('cls_weight', 1.0),
                'regression': weight_config.get('reg_weight', 1.0),
                'objectness': weight_config.get('obj_weight', 1.0),
                'quality': weight_config.get('quality_weight', 0.5)
            }
    
    def set_epoch(self, epoch: int):
        """Set current epoch for adaptive components."""
        if hasattr(self.cls_criterion, 'set_epoch'):
            self.cls_criterion.set_epoch(epoch)
    
    def forward(self, predictions: Dict, targets: Dict, 
                epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Advanced Combo Loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            epoch: Current epoch (for adaptive components)
            
        Returns:
            Dictionary of loss components
        """
        if epoch is not None:
            self.set_epoch(epoch)
        
        losses = {}
        
        # Classification loss
        if 'cls' in predictions and 'cls' in targets:
            if isinstance(self.cls_criterion, VarifocalLoss):
                # Varifocal loss needs quality scores
                quality_scores = targets.get('quality', torch.ones_like(targets['cls']))
                losses['cls_loss'] = self.cls_criterion(
                    predictions['cls'], targets['cls'], quality_scores
                )
            else:
                losses['cls_loss'] = self.cls_criterion(predictions['cls'], targets['cls'])
        
        # Regression loss
        if 'bbox' in predictions and 'bbox' in targets:
            if isinstance(self.reg_criterion, SmallObjectLoss):
                image_size = targets.get('image_size', (640, 640))
                losses['reg_loss'] = self.reg_criterion(
                    predictions['bbox'], targets['bbox'], image_size
                )
            else:
                losses['reg_loss'] = self.reg_criterion(predictions['bbox'], targets['bbox'])
        
        # Objectness loss
        if 'obj' in predictions and 'obj' in targets:
            losses['obj_loss'] = F.binary_cross_entropy_with_logits(
                predictions['obj'], targets['obj']
            )
        
        # Quality loss
        if (self.quality_criterion is not None and 
            'quality' in predictions and 'quality' in targets):
            if isinstance(self.quality_criterion, QualityFocalLoss):
                losses['quality_loss'] = self.quality_criterion(
                    predictions['cls'], targets['cls'],
                    predictions['quality'], targets['quality']
                )
            else:
                losses['quality_loss'] = self.quality_criterion(
                    predictions['quality'], targets['quality']
                )
        
        # Apply weighting
        if self.dynamic_weighting is not None:
            weights = self.dynamic_weighting.get_weights()
            # Update weights based on current losses - ensure scalar values
            loss_values = {}
            for k, v in losses.items():
                if 'loss' in k and torch.is_tensor(v):
                    if v.numel() == 1:
                        loss_values[k] = v.item()
                    else:
                        loss_values[k] = v.mean().item()
            self.dynamic_weighting.update_weights(loss_values)
        else:
            weights = self.static_weights
        
        # Calculate weighted total loss
        total_loss = 0
        for loss_name, loss_value in losses.items():
            if loss_name.endswith('_loss'):
                weight_key = loss_name.replace('_loss', '')
                weight = weights.get(weight_key, 1.0)
                
                # Ensure loss_value is a scalar
                if torch.is_tensor(loss_value):
                    if loss_value.numel() > 1:
                        loss_value = loss_value.mean()
                
                total_loss += weight * loss_value
        
        losses['total_loss'] = total_loss
        losses['weights'] = weights
        
        # Store loss history
        history_entry = {}
        for k, v in losses.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                history_entry[k] = v.item()
            elif isinstance(v, (int, float)):
                history_entry[k] = float(v)
        
        self.loss_history.append(history_entry)
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
        
        return losses
    
    def get_loss_statistics(self) -> Dict:
        """Get statistics about loss evolution."""
        if not self.loss_history:
            return {}
        
        stats = {}
        for key in self.loss_history[0].keys():
            if key != 'weights':
                values = [h[key] for h in self.loss_history[-20:]]  # Last 20 iterations
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                }
        
        return stats


def create_advanced_loss_config(strategy: str = "balanced") -> Dict:
    """
    Create advanced loss configuration for different strategies.
    
    Args:
        strategy: Loss strategy ('balanced', 'small_objects', 'quality_focused')
        
    Returns:
        Loss configuration dictionary
    """
    if strategy == "balanced":
        return {
            'classification': {
                'type': 'adaptive_focal',
                'alpha': 1.0,
                'gamma_init': 2.0,
                'gamma_final': 0.5,
                'total_epochs': 300
            },
            'regression': {
                'type': 'ciou'
            },
            'quality': {
                'enable': False
            },
            'weighting': {
                'dynamic': True,
                'cls_weight': 1.0,
                'reg_weight': 1.0,
                'obj_weight': 1.0,
                'adaptation_rate': 0.1
            }
        }
    
    elif strategy == "small_objects":
        return {
            'classification': {
                'type': 'focal',
                'alpha': 2.0,  # Higher alpha for rare small objects
                'gamma': 3.0   # Higher gamma for hard examples
            },
            'regression': {
                'type': 'small_object',
                'small_threshold': 32.0,
                'scale_factor': 3.0
            },
            'quality': {
                'enable': True,
                'type': 'quality_focal',
                'beta': 2.0
            },
            'weighting': {
                'dynamic': True,
                'cls_weight': 1.5,
                'reg_weight': 2.0,  # Higher weight for regression
                'obj_weight': 1.0,
                'quality_weight': 1.0,
                'adaptation_rate': 0.15
            }
        }
    
    elif strategy == "quality_focused":
        return {
            'classification': {
                'type': 'varifocal',
                'alpha': 0.75,
                'gamma': 2.0
            },
            'regression': {
                'type': 'ciou'
            },
            'quality': {
                'enable': True,
                'type': 'quality_focal',
                'beta': 2.0
            },
            'weighting': {
                'dynamic': False,
                'cls_weight': 1.0,
                'reg_weight': 1.0,
                'obj_weight': 1.0,
                'quality_weight': 1.5
            }
        }
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_pcb_optimized_loss() -> AdvancedComboLoss:
    """
    Create loss function optimized for PCB defect detection.
    
    Returns:
        Configured AdvancedComboLoss instance
    """
    config = create_advanced_loss_config("small_objects")
    return AdvancedComboLoss(config)