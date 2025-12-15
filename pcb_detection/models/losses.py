"""
Advanced loss functions for PCB defect detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in PCB defect detection."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Predictions from model (before sigmoid/softmax)
            targets: Ground truth labels
            
        Returns:
            Computed focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """IoU-based loss for bounding box regression."""
    
    def __init__(self, loss_type: str = 'iou', eps: float = 1e-6):
        """
        Initialize IoU Loss.
        
        Args:
            loss_type: Type of IoU loss ('iou', 'giou', 'diou', 'ciou')
            eps: Small epsilon to avoid division by zero
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.eps = eps
        
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of IoU Loss.
        
        Args:
            pred_boxes: Predicted bounding boxes [N, 4] (x1, y1, x2, y2)
            target_boxes: Target bounding boxes [N, 4] (x1, y1, x2, y2)
            
        Returns:
            Computed IoU loss
        """
        if self.loss_type == 'iou':
            return self._iou_loss(pred_boxes, target_boxes)
        elif self.loss_type == 'giou':
            return self._giou_loss(pred_boxes, target_boxes)
        elif self.loss_type == 'diou':
            return self._diou_loss(pred_boxes, target_boxes)
        elif self.loss_type == 'ciou':
            return self._ciou_loss(pred_boxes, target_boxes)
        else:
            raise ValueError(f"Unsupported IoU loss type: {self.loss_type}")
            
    def _iou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute standard IoU loss."""
        iou = self._compute_iou(pred_boxes, target_boxes)
        return 1 - iou
        
    def _giou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute Generalized IoU loss."""
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        # Compute enclosing box
        x1_c = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        y1_c = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        x2_c = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        y2_c = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        c_area = (x2_c - x1_c) * (y2_c - y1_c)
        union_area = self._compute_union_area(pred_boxes, target_boxes)
        
        giou = iou - (c_area - union_area) / (c_area + self.eps)
        return 1 - giou
        
    def _diou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute Distance IoU loss."""
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        # Compute center points
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        # Distance between centers
        center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Diagonal of enclosing box
        x1_c = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        y1_c = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        x2_c = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        y2_c = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        diagonal_distance = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2
        
        diou = iou - center_distance / (diagonal_distance + self.eps)
        return 1 - diou
        
    def _ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute Complete IoU loss."""
        diou_loss = self._diou_loss(pred_boxes, target_boxes)
        
        # Compute aspect ratio consistency
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + self.eps)) - torch.atan(pred_w / (pred_h + self.eps)), 2
        )
        
        iou = self._compute_iou(pred_boxes, target_boxes)
        alpha = v / (1 - iou + v + self.eps)
        
        ciou = 1 - diou_loss - alpha * v
        return ciou
        
    def _compute_iou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU between predicted and target boxes."""
        # Intersection area
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union area
        union = self._compute_union_area(pred_boxes, target_boxes)
        
        return intersection / (union + self.eps)
        
    def _compute_union_area(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute union area of predicted and target boxes."""
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Intersection area
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        return pred_area + target_area - intersection


class ClassBalancedLoss(nn.Module):
    """Class-balanced loss for handling class imbalance."""
    
    def __init__(self, samples_per_class: torch.Tensor, beta: float = 0.9999, 
                 loss_type: str = 'focal', gamma: float = 2.0):
        """
        Initialize Class-balanced Loss.
        
        Args:
            samples_per_class: Number of samples per class
            beta: Hyperparameter for re-weighting
            loss_type: Base loss type ('focal', 'ce')
            gamma: Focusing parameter for focal loss
        """
        super().__init__()
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        self.weights = weights / weights.sum() * len(weights)
        
        self.loss_type = loss_type
        if loss_type == 'focal':
            self.criterion = FocalLoss(gamma=gamma, reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of Class-balanced Loss."""
        if self.loss_type == 'focal':
            cb_loss = self.criterion(inputs, targets)
        else:
            cb_loss = self.criterion(inputs, targets)
            
        weights = self.weights[targets].to(inputs.device)
        cb_loss = weights * cb_loss
        
        return cb_loss.mean()


class ComboLoss(nn.Module):
    """Combination of multiple losses for comprehensive training."""
    
    def __init__(self, 
                 cls_loss_weight: float = 1.0,
                 bbox_loss_weight: float = 1.0,
                 obj_loss_weight: float = 1.0,
                 use_focal: bool = True,
                 use_iou: bool = True,
                 iou_type: str = 'ciou'):
        """
        Initialize Combo Loss.
        
        Args:
            cls_loss_weight: Weight for classification loss
            bbox_loss_weight: Weight for bounding box loss
            obj_loss_weight: Weight for objectness loss
            use_focal: Whether to use focal loss for classification
            use_iou: Whether to use IoU loss for bounding box regression
            iou_type: Type of IoU loss to use
        """
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.obj_loss_weight = obj_loss_weight
        
        # Classification loss
        if use_focal:
            self.cls_criterion = FocalLoss(gamma=2.0)
        else:
            self.cls_criterion = nn.CrossEntropyLoss()
            
        # Bounding box loss
        if use_iou:
            self.bbox_criterion = IoULoss(loss_type=iou_type)
        else:
            self.bbox_criterion = nn.SmoothL1Loss()
            
        # Objectness loss
        self.obj_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Forward pass of Combo Loss.
        
        Args:
            predictions: Dictionary containing model predictions
            targets: Dictionary containing ground truth targets
            
        Returns:
            Dictionary containing individual and total losses
        """
        # Classification loss
        cls_loss = self.cls_criterion(predictions['cls'], targets['cls'])
        
        # Bounding box loss
        bbox_loss = self.bbox_criterion(predictions['bbox'], targets['bbox'])
        
        # Objectness loss
        obj_loss = self.obj_criterion(predictions['obj'], targets['obj'])
        
        # Total loss
        total_loss = (self.cls_loss_weight * cls_loss + 
                     self.bbox_loss_weight * bbox_loss + 
                     self.obj_loss_weight * obj_loss)
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss,
            'obj_loss': obj_loss
        }