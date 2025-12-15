"""
Detection head implementation with multi-scale feature processing.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DetectionHead(nn.Module):
    """Detection head for YOLO model with multi-scale feature processing."""
    
    def __init__(self, in_channels_list: List[int], num_classes: int, num_anchors: int = 3):
        """
        Initialize detection head.
        
        Args:
            in_channels_list: List of input channels for each feature level
            num_classes: Number of classes
            num_anchors: Number of anchors per grid cell
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels_list = in_channels_list
        
        # Output channels: (x, y, w, h, objectness, classes)
        self.output_channels = num_anchors * (5 + num_classes)
        
        # Create detection heads for each feature level
        self.heads = nn.ModuleList()
        for in_channels in in_channels_list:
            head = self._make_head(in_channels)
            self.heads.append(head)
            
        # Anchor configurations for different scales
        # These are typical YOLO anchor sizes, can be optimized for PCB defects
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # Small objects (P3)
            [[30, 61], [62, 45], [59, 119]],    # Medium objects (P4)  
            [[116, 90], [156, 198], [373, 326]] # Large objects (P5)
        ], dtype=torch.float32)
        
        # Grid strides for each feature level
        self.strides = torch.tensor([8, 16, 32], dtype=torch.float32)
        
        self._initialize_weights()
        
    def _make_head(self, in_channels: int) -> nn.Module:
        """Create detection head for a single feature level."""
        return nn.Sequential(
            # Feature processing layers
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            
            # Output layer
            nn.Conv2d(in_channels, self.output_channels, 1, 1, 0)
        )
        
    def _initialize_weights(self):
        """Initialize weights for detection heads."""
        for head in self.heads:
            # Initialize output layer with special bias for objectness
            output_layer = head[-1]
            nn.init.normal_(output_layer.weight, 0, 0.01)
            
            # Initialize bias for better convergence
            # Set objectness bias to encourage detection
            with torch.no_grad():
                bias = output_layer.bias.view(self.num_anchors, -1)
                bias[:, 4].fill_(-math.log((1 - 0.01) / 0.01))  # Objectness bias
                bias[:, 5:].fill_(0)  # Class bias
            
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through detection head.
        
        Args:
            features: List of feature tensors from different scales
            
        Returns:
            List of prediction tensors for each scale
        """
        predictions = []
        
        for i, (feature, head) in enumerate(zip(features, self.heads)):
            # Apply detection head
            pred = head(feature)
            
            # Reshape prediction: [B, anchors*(5+classes), H, W] -> [B, anchors, H, W, 5+classes]
            batch_size, _, height, width = pred.shape
            pred = pred.view(batch_size, self.num_anchors, -1, height, width)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Apply activations
            pred = self._apply_activations(pred, i)
            
            predictions.append(pred)
            
        return predictions
        
    def _apply_activations(self, pred: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Apply appropriate activations to predictions."""
        # Split predictions
        xy = pred[..., :2]      # Center coordinates
        wh = pred[..., 2:4]     # Width and height
        obj = pred[..., 4:5]    # Objectness
        cls = pred[..., 5:]     # Class probabilities
        
        # Apply sigmoid to center coordinates and objectness
        xy = torch.sigmoid(xy)
        obj = torch.sigmoid(obj)
        
        # Apply sigmoid to class probabilities (multi-class classification)
        cls = torch.sigmoid(cls)
        
        # Exponential for width and height (with anchor scaling)
        if self.training:
            # During training, keep raw predictions for loss calculation
            wh = wh
        else:
            # During inference, apply exponential and anchor scaling
            anchors = self.anchors[scale_idx].to(pred.device)
            wh = torch.exp(wh) * anchors.view(1, -1, 1, 1, 2)
            
        return torch.cat([xy, wh, obj, cls], dim=-1)
        
    def decode_predictions(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode predictions to absolute coordinates.
        
        Args:
            predictions: List of prediction tensors
            
        Returns:
            Decoded predictions tensor [N, 6] (x, y, w, h, conf, class)
        """
        decoded_preds = []
        
        for i, pred in enumerate(predictions):
            batch_size, num_anchors, height, width, _ = pred.shape
            
            # Create grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=pred.device),
                torch.arange(width, device=pred.device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            grid = grid.unsqueeze(0).unsqueeze(0).expand(batch_size, num_anchors, -1, -1, -1)
            
            # Get stride and anchors for this scale
            stride = self.strides[i].to(pred.device)
            anchors = self.anchors[i].to(pred.device)
            
            # Decode coordinates
            xy = (pred[..., :2] + grid) * stride
            wh = torch.exp(pred[..., 2:4]) * anchors.view(1, -1, 1, 1, 2)
            
            # Get confidence and class scores
            obj_conf = pred[..., 4:5]
            cls_conf = pred[..., 5:]
            
            # Calculate final confidence (objectness * class confidence)
            conf, class_id = torch.max(cls_conf, dim=-1, keepdim=True)
            conf = obj_conf * conf
            
            # Reshape to [batch_size * num_anchors * height * width, 6]
            xy = xy.view(batch_size, -1, 2)
            wh = wh.view(batch_size, -1, 2)
            conf = conf.view(batch_size, -1, 1)
            class_id = class_id.view(batch_size, -1, 1).float()
            
            # Concatenate predictions
            pred_decoded = torch.cat([xy, wh, conf, class_id], dim=-1)
            decoded_preds.append(pred_decoded)
            
        # Concatenate all scales
        return torch.cat(decoded_preds, dim=1)
        
    def compute_loss(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> dict:
        """
        Compute detection loss.
        
        Args:
            predictions: List of prediction tensors
            targets: Ground truth targets
            
        Returns:
            Dictionary containing loss components
        """
        # This is a simplified loss computation
        # In practice, you would implement more sophisticated loss functions
        
        total_loss = 0
        loss_components = {
            'box_loss': 0,
            'obj_loss': 0, 
            'cls_loss': 0
        }
        
        for pred in predictions:
            # Placeholder loss computation
            # Real implementation would match predictions with targets
            # and compute appropriate losses for each component
            
            batch_size = pred.shape[0]
            
            # Dummy loss for demonstration
            box_loss = torch.mean(pred[..., :4] ** 2) * 0.05
            obj_loss = torch.mean(pred[..., 4] ** 2) * 1.0
            cls_loss = torch.mean(pred[..., 5:] ** 2) * 0.5
            
            loss_components['box_loss'] += box_loss
            loss_components['obj_loss'] += obj_loss
            loss_components['cls_loss'] += cls_loss
            
            total_loss += box_loss + obj_loss + cls_loss
            
        loss_components['total_loss'] = total_loss
        return loss_components


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        """
        Initialize FPN.
        
        Args:
            in_channels_list: List of input channels for each level
            out_channels: Output channels for all levels
        """
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1, bias=False)
            )
            
        # Output convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            )
            
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through FPN."""
        # Apply lateral connections
        laterals = []
        for feature, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feature))
            
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample higher level feature
            upsampled = F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[2:], 
                mode='nearest'
            )
            laterals[i] = laterals[i] + upsampled
            
        # Apply output convolutions
        outputs = []
        for lateral, fpn_conv in zip(laterals, self.fpn_convs):
            outputs.append(fpn_conv(lateral))
            
        return outputs


class PANet(nn.Module):
    """Path Aggregation Network for enhanced feature fusion."""
    
    def __init__(self, in_channels: int):
        """
        Initialize PANet.
        
        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        
        # Bottom-up path augmentation
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False)
        ])
        
    def forward(self, fpn_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through PANet."""
        # Start with FPN features
        outputs = [fpn_features[0]]  # P3
        
        # Bottom-up augmentation
        for i, downsample_conv in enumerate(self.downsample_convs):
            # Downsample previous level
            downsampled = downsample_conv(outputs[-1])
            
            # Add with corresponding FPN feature
            enhanced = downsampled + fpn_features[i + 1]
            outputs.append(enhanced)
            
        return outputs