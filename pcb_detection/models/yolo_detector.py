"""
YOLO detector implementation for PCB defect detection.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import cv2

from ..core.interfaces import ModelInterface
from ..core.types import Detection, TrainingConfig, CLASS_MAPPING, CLASS_LABELS
from .attention import AttentionBlock
from .losses import ComboLoss
from .detection_head import DetectionHead
from .postprocessing import PostProcessor


class YOLOBackbone(nn.Module):
    """YOLO backbone network with attention mechanisms."""
    
    def __init__(self, model_variant: str = "yolov8n", attention_type: Optional[str] = None):
        """
        Initialize YOLO backbone.
        
        Args:
            model_variant: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            attention_type: Type of attention mechanism to integrate
        """
        super().__init__()
        self.model_variant = model_variant
        self.attention_type = attention_type
        
        # Define channel configurations for different variants
        self.channel_configs = {
            "yolov8n": [64, 128, 256, 512],
            "yolov8s": [64, 128, 256, 512], 
            "yolov8m": [96, 192, 384, 768],
            "yolov8l": [128, 256, 512, 1024],
            "yolov8x": [160, 320, 640, 1280]
        }
        
        channels = self.channel_configs.get(model_variant, self.channel_configs["yolov8n"])
        
        # Build backbone layers
        self.stem = self._make_stem(3, channels[0])
        self.stage1 = self._make_stage(channels[0], channels[1], 2)
        self.stage2 = self._make_stage(channels[1], channels[2], 2)
        self.stage3 = self._make_stage(channels[2], channels[3], 2)
        
        # Add attention mechanisms if specified
        if attention_type:
            self.attention1 = AttentionBlock(channels[1], attention_type)
            self.attention2 = AttentionBlock(channels[2], attention_type)
            self.attention3 = AttentionBlock(channels[3], attention_type)
        else:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
            self.attention3 = nn.Identity()
            
    def _make_stem(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create stem layer."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 6, 2, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def _make_stage(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Create stage with residual blocks."""
        layers = []
        
        # Downsample layer
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.SiLU(inplace=True))
        
        # Residual blocks
        for _ in range(2):
            layers.append(self._make_residual_block(out_channels))
            
        return nn.Sequential(*layers)
        
    def _make_residual_block(self, channels: int) -> nn.Module:
        """Create residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through backbone."""
        x = self.stem(x)
        
        # Stage 1 - P2
        x1 = self.stage1(x)
        x1 = self.attention1(x1)
        
        # Stage 2 - P3  
        x2 = self.stage2(x1)
        x2 = self.attention2(x2)
        
        # Stage 3 - P4
        x3 = self.stage3(x2)
        x3 = self.attention3(x3)
        
        return [x1, x2, x3]


class YOLONeck(nn.Module):
    """YOLO neck with FPN and PANet, enhanced with attention mechanisms."""
    
    def __init__(self, channels: List[int], attention_type: Optional[str] = None):
        """
        Initialize YOLO neck.
        
        Args:
            channels: Channel dimensions for each feature level
            attention_type: Type of attention mechanism to integrate
        """
        super().__init__()
        self.attention_type = attention_type
        
        # Top-down pathway (FPN)
        self.fpn_conv1 = nn.Conv2d(channels[2], channels[1], 1, bias=False)
        self.fpn_conv2 = nn.Conv2d(channels[1], channels[0], 1, bias=False)
        
        # Bottom-up pathway (PANet)
        self.pan_conv1 = nn.Conv2d(channels[0], channels[1], 3, 2, 1, bias=False)
        self.pan_conv2 = nn.Conv2d(channels[1], channels[2], 3, 2, 1, bias=False)
        
        # Output convolutions
        self.out_conv1 = self._make_conv_block(channels[0], channels[0])
        self.out_conv2 = self._make_conv_block(channels[1], channels[1])
        self.out_conv3 = self._make_conv_block(channels[2], channels[2])
        
        # Add attention mechanisms to neck if specified
        if attention_type:
            self.neck_attention1 = AttentionBlock(channels[0], attention_type)
            self.neck_attention2 = AttentionBlock(channels[1], attention_type)
            self.neck_attention3 = AttentionBlock(channels[2], attention_type)
        else:
            self.neck_attention1 = nn.Identity()
            self.neck_attention2 = nn.Identity()
            self.neck_attention3 = nn.Identity()
        
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through neck."""
        p3, p4, p5 = features
        
        # Top-down pathway
        fpn_p4 = self.fpn_conv1(p5)
        fpn_p4 = F.interpolate(fpn_p4, size=p4.shape[2:], mode='nearest')
        fpn_p4 = fpn_p4 + p4
        
        fpn_p3 = self.fpn_conv2(fpn_p4)
        fpn_p3 = F.interpolate(fpn_p3, size=p3.shape[2:], mode='nearest')
        fpn_p3 = fpn_p3 + p3
        
        # Bottom-up pathway
        pan_p4 = self.pan_conv1(fpn_p3)
        pan_p4 = pan_p4 + fpn_p4
        
        pan_p5 = self.pan_conv2(pan_p4)
        pan_p5 = pan_p5 + p5
        
        # Output features with attention
        out_p3 = self.out_conv1(fpn_p3)
        out_p3 = self.neck_attention1(out_p3)
        
        out_p4 = self.out_conv2(pan_p4)
        out_p4 = self.neck_attention2(out_p4)
        
        out_p5 = self.out_conv3(pan_p5)
        out_p5 = self.neck_attention3(out_p5)
        
        return [out_p3, out_p4, out_p5]


class YOLODetector(ModelInterface):
    """YOLO detector for PCB defect detection."""
    
    def __init__(self, model_config: Dict, num_classes: int = 5):
        """
        Initialize YOLO detector.
        
        Args:
            model_config: Model configuration dictionary
            num_classes: Number of classes (default: 5 for PCB defects)
        """
        self.model_config = model_config
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract configuration
        self.model_variant = model_config.get("variant", "yolov8n")
        self.attention_type = model_config.get("attention_type", None)
        self.input_size = model_config.get("input_size", 640)
        self.conf_threshold = model_config.get("conf_threshold", 0.25)
        self.iou_threshold = model_config.get("iou_threshold", 0.45)
        
        # Build model components
        self._build_model()
        
        # Initialize loss function
        self.criterion = ComboLoss(
            use_focal=model_config.get("use_focal", True),
            use_iou=model_config.get("use_iou", True),
            iou_type=model_config.get("iou_type", "ciou")
        )
        
        # Initialize post-processor
        self.post_processor = PostProcessor(
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            nms_type=model_config.get("nms_type", "standard")
        )
        
    def _build_model(self):
        """Build YOLO model components."""
        # Get channel configuration
        channel_configs = {
            "yolov8n": [64, 128, 256, 512],
            "yolov8s": [64, 128, 256, 512], 
            "yolov8m": [96, 192, 384, 768],
            "yolov8l": [128, 256, 512, 1024],
            "yolov8x": [160, 320, 640, 1280]
        }
        
        channels = channel_configs.get(self.model_variant, channel_configs["yolov8n"])
        
        # Build components
        self.backbone = YOLOBackbone(self.model_variant, self.attention_type)
        self.neck = YOLONeck(channels[1:], self.attention_type)  # Skip first channel for neck
        self.head = DetectionHead(channels[1:], self.num_classes, num_anchors=3)
        
        # Move to device
        self.backbone = self.backbone.to(self.device)
        self.neck = self.neck.to(self.device)
        self.head = self.head.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Extract features
        features = self.backbone(x)
        
        # Feature fusion
        neck_features = self.neck(features)
        
        # Detection head
        predictions = self.head(neck_features)
        
        return predictions
        
    def predict(self, image: np.ndarray) -> List[Detection]:
        """
        Predict detections for a single image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of Detection objects
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.forward(processed_image)
            
        # Decode predictions if needed
        if isinstance(predictions, list):
            # Multi-scale predictions - decode them
            decoded_preds = self.head.decode_predictions(predictions)
        else:
            decoded_preds = predictions
            
        # Ensure we have the right shape for post-processing
        if decoded_preds.dim() == 2:
            decoded_preds = decoded_preds.unsqueeze(0)
            
        # Post-process predictions
        batch_detections = self.post_processor(
            decoded_preds, 
            image_shapes=[image.shape[:2]],
            input_size=self.input_size
        )
        
        return batch_detections[0]  # Return detections for single image
        
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference."""
        # Resize image
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalize and convert to tensor
        tensor = torch.from_numpy(padded).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
        

        
    def train_model(self, config: TrainingConfig) -> Dict[str, Any]:
        """Train the model with given configuration."""
        # This would implement the full training loop
        # For now, return a placeholder
        return {
            "status": "training_not_implemented",
            "message": "Training implementation will be added in training module"
        }
        
    def load_weights(self, path: str) -> None:
        """Load model weights from file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Weight file not found: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load component weights
        if "backbone" in checkpoint:
            self.backbone.load_state_dict(checkpoint["backbone"])
        if "neck" in checkpoint:
            self.neck.load_state_dict(checkpoint["neck"])
        if "head" in checkpoint:
            self.head.load_state_dict(checkpoint["head"])
            
    def save_weights(self, path: str) -> None:
        """Save model weights to file."""
        checkpoint = {
            "backbone": self.backbone.state_dict(),
            "neck": self.neck.state_dict(),
            "head": self.head.state_dict(),
            "config": self.model_config
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
    def compute_loss(self, predictions: torch.Tensor, targets: Dict) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        return self.criterion(predictions, targets)
        
    def set_train_mode(self, training: bool = True):
        """Set training mode for all components."""
        self.backbone.train(training)
        self.neck.train(training)
        self.head.train(training)
        
    def parameters(self):
        """Get all model parameters."""
        params = []
        params.extend(self.backbone.parameters())
        params.extend(self.neck.parameters())
        params.extend(self.head.parameters())
        return params