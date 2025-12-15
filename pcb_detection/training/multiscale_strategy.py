"""
Multi-scale training and testing strategies for PCB defect detection.
Implements various multi-scale techniques to improve detection performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
import random
import math


class MultiScaleConfig:
    """Configuration for multi-scale training and testing."""
    
    # Standard multi-scale configurations
    CONFIGS = {
        "basic": {
            "train_sizes": [480, 512, 544, 576, 608, 640, 672, 704, 736],
            "test_sizes": [640],
            "size_change_freq": 10,  # Change size every N epochs
            "min_size": 320,
            "max_size": 832,
            "size_divisible": 32
        },
        "aggressive": {
            "train_sizes": [416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
            "test_sizes": [640, 704, 768],
            "size_change_freq": 5,
            "min_size": 320,
            "max_size": 896,
            "size_divisible": 32
        },
        "small_objects": {
            "train_sizes": [640, 704, 768, 832, 896, 960],
            "test_sizes": [640, 768, 896],
            "size_change_freq": 8,
            "min_size": 640,
            "max_size": 1024,
            "size_divisible": 32
        },
        "efficient": {
            "train_sizes": [512, 576, 640, 704],
            "test_sizes": [640],
            "size_change_freq": 15,
            "min_size": 416,
            "max_size": 768,
            "size_divisible": 32
        }
    }
    
    @classmethod
    def get_config(cls, config_name: str = "basic") -> Dict:
        """Get multi-scale configuration."""
        return cls.CONFIGS.get(config_name, cls.CONFIGS["basic"])


class MultiScaleTrainer:
    """Multi-scale training strategy implementation."""
    
    def __init__(self, config: Dict):
        """
        Initialize multi-scale trainer.
        
        Args:
            config: Multi-scale configuration
        """
        self.config = config
        self.train_sizes = config["train_sizes"]
        self.size_change_freq = config["size_change_freq"]
        self.min_size = config["min_size"]
        self.max_size = config["max_size"]
        self.size_divisible = config["size_divisible"]
        
        self.current_size = self.train_sizes[len(self.train_sizes) // 2]  # Start with middle size
        self.epoch_count = 0
        
    def get_current_size(self, epoch: Optional[int] = None) -> int:
        """
        Get current training size based on epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Current training image size
        """
        if epoch is not None:
            self.epoch_count = epoch
            
        # Change size based on frequency
        if self.epoch_count % self.size_change_freq == 0:
            self.current_size = random.choice(self.train_sizes)
            
        return self.current_size
    
    def get_random_size(self) -> int:
        """Get random training size."""
        return random.choice(self.train_sizes)
    
    def get_progressive_size(self, epoch: int, total_epochs: int) -> int:
        """
        Get progressively increasing size during training.
        
        Args:
            epoch: Current epoch
            total_epochs: Total training epochs
            
        Returns:
            Progressive training size
        """
        # Start with smaller sizes, gradually increase
        progress = epoch / total_epochs
        size_index = int(progress * (len(self.train_sizes) - 1))
        size_index = min(size_index, len(self.train_sizes) - 1)
        
        return self.train_sizes[size_index]
    
    def resize_batch(self, images: torch.Tensor, targets: List[Dict], 
                    target_size: int) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Resize batch of images and adjust targets accordingly.
        
        Args:
            images: Batch of images [B, C, H, W]
            targets: List of target dictionaries
            target_size: Target image size
            
        Returns:
            Resized images and adjusted targets
        """
        batch_size, channels, orig_h, orig_w = images.shape
        
        # Calculate scale factor
        scale = target_size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        # Make dimensions divisible by size_divisible
        new_h = (new_h // self.size_divisible) * self.size_divisible
        new_w = (new_w // self.size_divisible) * self.size_divisible
        
        # Resize images
        resized_images = F.interpolate(
            images, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        
        # Pad to square if needed
        if new_h != new_w:
            max_size = max(new_h, new_w)
            padded_images = torch.zeros(batch_size, channels, max_size, max_size, 
                                      dtype=images.dtype, device=images.device)
            padded_images[:, :, :new_h, :new_w] = resized_images
            resized_images = padded_images
        
        # Adjust targets
        adjusted_targets = []
        for target in targets:
            adjusted_target = target.copy()
            
            # Scale bounding boxes if present
            if 'boxes' in target:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] *= (new_w / orig_w)  # x coordinates
                boxes[:, [1, 3]] *= (new_h / orig_h)  # y coordinates
                adjusted_target['boxes'] = boxes
            
            # Scale keypoints if present
            if 'keypoints' in target:
                keypoints = target['keypoints'].clone()
                keypoints[:, :, 0] *= (new_w / orig_w)  # x coordinates
                keypoints[:, :, 1] *= (new_h / orig_h)  # y coordinates
                adjusted_target['keypoints'] = keypoints
                
            adjusted_targets.append(adjusted_target)
        
        return resized_images, adjusted_targets


class TestTimeAugmentation:
    """Test Time Augmentation (TTA) for improved inference accuracy."""
    
    def __init__(self, scales: List[float] = None, flips: List[bool] = None,
                 rotations: List[float] = None):
        """
        Initialize TTA.
        
        Args:
            scales: List of scale factors for multi-scale testing
            flips: List of flip operations [horizontal, vertical]
            rotations: List of rotation angles in degrees
        """
        self.scales = scales or [0.8, 1.0, 1.2]
        self.flips = flips or [False, True]  # [no_flip, horizontal_flip]
        self.rotations = rotations or [0]  # No rotation by default
        
    def augment_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """
        Apply TTA to single image.
        
        Args:
            image: Input image [H, W, C]
            
        Returns:
            List of (augmented_image, transform_info) tuples
        """
        augmented_images = []
        
        for scale in self.scales:
            for flip in self.flips:
                for rotation in self.rotations:
                    aug_image = image.copy()
                    transform_info = {
                        'scale': scale,
                        'flip': flip,
                        'rotation': rotation,
                        'original_shape': image.shape[:2]
                    }
                    
                    # Apply scale
                    if scale != 1.0:
                        h, w = image.shape[:2]
                        new_h, new_w = int(h * scale), int(w * scale)
                        aug_image = cv2.resize(aug_image, (new_w, new_h))
                    
                    # Apply flip
                    if flip:
                        aug_image = cv2.flip(aug_image, 1)  # Horizontal flip
                    
                    # Apply rotation
                    if rotation != 0:
                        h, w = aug_image.shape[:2]
                        center = (w // 2, h // 2)
                        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                        aug_image = cv2.warpAffine(aug_image, matrix, (w, h))
                    
                    augmented_images.append((aug_image, transform_info))
        
        return augmented_images
    
    def reverse_transform_detections(self, detections: List, transform_info: Dict,
                                   image_shape: Tuple[int, int]) -> List:
        """
        Reverse transform detections to original image coordinates.
        
        Args:
            detections: List of detections
            transform_info: Transform information
            image_shape: Original image shape (H, W)
            
        Returns:
            Transformed detections
        """
        if not detections:
            return detections
        
        transformed_detections = []
        orig_h, orig_w = image_shape
        
        for detection in detections:
            det = detection.copy()
            
            # Reverse rotation
            if transform_info['rotation'] != 0:
                # This is complex for bounding boxes, simplified implementation
                pass
            
            # Reverse flip
            if transform_info['flip']:
                if hasattr(det, 'bbox'):
                    x, y, w, h = det.bbox
                    det.bbox = (1.0 - x - w, y, w, h)  # Assuming normalized coordinates
            
            # Reverse scale
            if transform_info['scale'] != 1.0:
                scale = transform_info['scale']
                if hasattr(det, 'bbox'):
                    x, y, w, h = det.bbox
                    det.bbox = (x / scale, y / scale, w / scale, h / scale)
            
            transformed_detections.append(det)
        
        return transformed_detections
    
    def merge_detections(self, all_detections: List[List], 
                        iou_threshold: float = 0.5) -> List:
        """
        Merge detections from multiple TTA augmentations.
        
        Args:
            all_detections: List of detection lists from different augmentations
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Merged detections
        """
        if not all_detections:
            return []
        
        # Flatten all detections
        merged_detections = []
        for detections in all_detections:
            merged_detections.extend(detections)
        
        if not merged_detections:
            return []
        
        # Apply NMS to merged detections
        # This is a simplified implementation
        # In practice, you would use proper NMS with confidence weighting
        
        return merged_detections


class MultiScaleTester:
    """Multi-scale testing strategy."""
    
    def __init__(self, test_sizes: List[int], tta_config: Dict = None):
        """
        Initialize multi-scale tester.
        
        Args:
            test_sizes: List of test image sizes
            tta_config: TTA configuration
        """
        self.test_sizes = test_sizes
        self.tta = TestTimeAugmentation(**(tta_config or {}))
        
    def predict_multiscale(self, model, image: np.ndarray, 
                          use_tta: bool = True) -> List:
        """
        Perform multi-scale prediction on single image.
        
        Args:
            model: Detection model
            image: Input image
            use_tta: Whether to use test time augmentation
            
        Returns:
            Merged predictions from all scales
        """
        all_predictions = []
        
        for size in self.test_sizes:
            if use_tta:
                # Apply TTA
                augmented_images = self.tta.augment_image(image)
                
                for aug_image, transform_info in augmented_images:
                    # Resize to test size
                    resized_image = self._resize_image(aug_image, size)
                    
                    # Predict
                    predictions = model.predict(resized_image)
                    
                    # Reverse transform
                    predictions = self.tta.reverse_transform_detections(
                        predictions, transform_info, image.shape[:2]
                    )
                    
                    all_predictions.append(predictions)
            else:
                # Simple multi-scale without TTA
                resized_image = self._resize_image(image, size)
                predictions = model.predict(resized_image)
                all_predictions.append(predictions)
        
        # Merge all predictions
        merged_predictions = self.tta.merge_detections(all_predictions)
        
        return merged_predictions
    
    def _resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        return padded


class AdaptiveMultiScale:
    """Adaptive multi-scale strategy that adjusts based on detection performance."""
    
    def __init__(self, initial_config: str = "basic"):
        """
        Initialize adaptive multi-scale strategy.
        
        Args:
            initial_config: Initial configuration name
        """
        self.config = MultiScaleConfig.get_config(initial_config)
        self.performance_history = []
        self.adaptation_threshold = 0.02  # Minimum improvement threshold
        
    def update_performance(self, epoch: int, metrics: Dict):
        """
        Update performance history and adapt strategy if needed.
        
        Args:
            epoch: Current epoch
            metrics: Performance metrics (mAP, etc.)
        """
        self.performance_history.append({
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config.copy()
        })
        
        # Adapt strategy based on performance
        if len(self.performance_history) >= 5:
            self._adapt_strategy()
    
    def _adapt_strategy(self):
        """Adapt multi-scale strategy based on performance history."""
        recent_performance = self.performance_history[-5:]
        
        # Calculate performance trend
        map_values = [p['metrics'].get('mAP', 0) for p in recent_performance]
        trend = np.polyfit(range(len(map_values)), map_values, 1)[0]
        
        # If performance is stagnating, try more aggressive multi-scale
        if abs(trend) < self.adaptation_threshold:
            if self.config == MultiScaleConfig.get_config("basic"):
                self.config = MultiScaleConfig.get_config("aggressive")
                print("Switching to aggressive multi-scale strategy")
            elif self.config == MultiScaleConfig.get_config("aggressive"):
                self.config = MultiScaleConfig.get_config("small_objects")
                print("Switching to small objects multi-scale strategy")
    
    def get_current_config(self) -> Dict:
        """Get current multi-scale configuration."""
        return self.config


def create_multiscale_trainer(config_name: str = "basic", 
                            adaptive: bool = False) -> Union[MultiScaleTrainer, AdaptiveMultiScale]:
    """
    Create multi-scale trainer with specified configuration.
    
    Args:
        config_name: Configuration name
        adaptive: Whether to use adaptive strategy
        
    Returns:
        Multi-scale trainer instance
    """
    if adaptive:
        return AdaptiveMultiScale(config_name)
    else:
        config = MultiScaleConfig.get_config(config_name)
        return MultiScaleTrainer(config)


def create_multiscale_tester(config_name: str = "basic", 
                           enable_tta: bool = True) -> MultiScaleTester:
    """
    Create multi-scale tester with specified configuration.
    
    Args:
        config_name: Configuration name
        enable_tta: Whether to enable test time augmentation
        
    Returns:
        Multi-scale tester instance
    """
    config = MultiScaleConfig.get_config(config_name)
    
    tta_config = {
        'scales': [0.9, 1.0, 1.1],
        'flips': [False, True],
        'rotations': [0]
    } if enable_tta else {}
    
    return MultiScaleTester(config["test_sizes"], tta_config)