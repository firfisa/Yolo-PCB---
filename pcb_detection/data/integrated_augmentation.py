"""
Integrated advanced data augmentation pipeline for PCB defect detection.
Implements progressive augmentation strategies and training-aware augmentation.
"""

import cv2
import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .advanced_augmentation import (
    MosaicAugmentation, CopyPasteAugmentation, MixUpAugmentation, 
    AlbumentationsAugmentation, ALBUMENTATIONS_AVAILABLE
)


class AugmentationPhase(Enum):
    """Training phases for progressive augmentation."""
    WARMUP = "warmup"
    AGGRESSIVE = "aggressive"
    FINE_TUNE = "fine_tune"
    FINAL = "final"


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    # Basic parameters
    image_size: int = 640
    
    # Mosaic augmentation
    mosaic_prob: float = 0.5
    mosaic_border: Tuple[int, int] = (-320, -320)
    
    # Copy-paste augmentation
    copy_paste_prob: float = 0.3
    max_paste_objects: int = 3
    paste_scale_range: Tuple[float, float] = (0.8, 1.2)
    
    # MixUp augmentation
    mixup_prob: float = 0.2
    mixup_alpha: float = 0.2
    
    # Albumentations
    use_albumentations: bool = True
    albumentations_prob: float = 0.8
    
    # Progressive augmentation
    use_progressive: bool = True
    warmup_epochs: int = 10
    aggressive_epochs: int = 150
    fine_tune_epochs: int = 100
    
    # Defect-specific augmentation
    small_defect_boost: bool = True
    small_defect_threshold: float = 0.02  # Normalized area threshold
    small_defect_copy_prob: float = 0.5


class ProgressiveAugmentationScheduler:
    """Scheduler for progressive augmentation strategies."""
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize progressive augmentation scheduler.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.current_epoch = 0
        self.total_epochs = (config.warmup_epochs + config.aggressive_epochs + 
                           config.fine_tune_epochs)
        
    def set_epoch(self, epoch: int):
        """Set current epoch for phase determination."""
        self.current_epoch = epoch
        
    def get_current_phase(self) -> AugmentationPhase:
        """Get current augmentation phase based on epoch."""
        if self.current_epoch < self.config.warmup_epochs:
            return AugmentationPhase.WARMUP
        elif self.current_epoch < (self.config.warmup_epochs + self.config.aggressive_epochs):
            return AugmentationPhase.AGGRESSIVE
        elif self.current_epoch < self.total_epochs:
            return AugmentationPhase.FINE_TUNE
        else:
            return AugmentationPhase.FINAL
    
    def get_phase_config(self) -> Dict:
        """Get augmentation configuration for current phase."""
        phase = self.get_current_phase()
        base_config = self.config
        
        if phase == AugmentationPhase.WARMUP:
            return {
                'mosaic_prob': base_config.mosaic_prob * 0.3,
                'copy_paste_prob': base_config.copy_paste_prob * 0.5,
                'mixup_prob': base_config.mixup_prob * 0.2,
                'albumentations_prob': base_config.albumentations_prob * 0.5,
                'augmentation_strength': 0.3
            }
        elif phase == AugmentationPhase.AGGRESSIVE:
            return {
                'mosaic_prob': base_config.mosaic_prob,
                'copy_paste_prob': base_config.copy_paste_prob,
                'mixup_prob': base_config.mixup_prob,
                'albumentations_prob': base_config.albumentations_prob,
                'augmentation_strength': 1.0
            }
        elif phase == AugmentationPhase.FINE_TUNE:
            return {
                'mosaic_prob': base_config.mosaic_prob * 0.5,
                'copy_paste_prob': base_config.copy_paste_prob * 0.7,
                'mixup_prob': base_config.mixup_prob * 0.3,
                'albumentations_prob': base_config.albumentations_prob * 0.6,
                'augmentation_strength': 0.5
            }
        else:  # FINAL
            return {
                'mosaic_prob': 0.0,
                'copy_paste_prob': base_config.copy_paste_prob * 0.3,
                'mixup_prob': 0.0,
                'albumentations_prob': base_config.albumentations_prob * 0.3,
                'augmentation_strength': 0.2
            }


class SmallDefectAugmentation:
    """Specialized augmentation for small PCB defects."""
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize small defect augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.small_threshold = config.small_defect_threshold
        self.copy_prob = config.small_defect_copy_prob
        
    def is_small_defect(self, bbox: List[float]) -> bool:
        """
        Check if defect is considered small.
        
        Args:
            bbox: Bounding box [x_center, y_center, width, height] normalized
            
        Returns:
            True if defect is small
        """
        area = bbox[2] * bbox[3]  # width * height
        return area < self.small_threshold
    
    def augment_small_defects(self, image: np.ndarray, 
                            annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply specialized augmentation for small defects.
        
        Args:
            image: Input image
            annotations: List of annotations
            
        Returns:
            Augmented image and annotations
        """
        if not annotations or random.random() > self.copy_prob:
            return image, annotations
        
        augmented_image = image.copy()
        augmented_annotations = annotations.copy()
        
        # Find small defects
        small_defects = [ann for ann in annotations 
                        if self.is_small_defect(ann['bbox'])]
        
        if not small_defects:
            return image, annotations
        
        h, w = image.shape[:2]
        
        # Copy small defects to increase their representation
        for defect in small_defects:
            if random.random() < 0.7:  # 70% chance to copy each small defect
                bbox = defect['bbox']
                
                # Convert to absolute coordinates
                x_center = int(bbox[0] * w)
                y_center = int(bbox[1] * h)
                width = int(bbox[2] * w)
                height = int(bbox[3] * h)
                
                # Extract defect region with some padding
                padding = max(2, min(width, height) // 4)
                x1 = max(0, x_center - width // 2 - padding)
                y1 = max(0, y_center - height // 2 - padding)
                x2 = min(w, x_center + width // 2 + padding)
                y2 = min(h, y_center + height // 2 + padding)
                
                if x2 > x1 and y2 > y1:
                    defect_region = image[y1:y2, x1:x2].copy()
                    
                    # Apply slight variations to the copied defect
                    if random.random() < 0.5:
                        # Slight rotation
                        angle = random.uniform(-15, 15)
                        center = (defect_region.shape[1] // 2, defect_region.shape[0] // 2)
                        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        defect_region = cv2.warpAffine(defect_region, matrix, 
                                                     (defect_region.shape[1], defect_region.shape[0]))
                    
                    if random.random() < 0.3:
                        # Slight scaling
                        scale = random.uniform(0.9, 1.1)
                        new_size = (int(defect_region.shape[1] * scale), 
                                  int(defect_region.shape[0] * scale))
                        defect_region = cv2.resize(defect_region, new_size)
                    
                    # Find new location to paste
                    for _ in range(20):  # Try up to 20 times
                        new_x = random.randint(defect_region.shape[1] // 2, 
                                             w - defect_region.shape[1] // 2)
                        new_y = random.randint(defect_region.shape[0] // 2, 
                                             h - defect_region.shape[0] // 2)
                        
                        # Check for overlap with existing defects
                        overlap = False
                        for existing_ann in augmented_annotations:
                            existing_bbox = existing_ann['bbox']
                            existing_x = int(existing_bbox[0] * w)
                            existing_y = int(existing_bbox[1] * h)
                            existing_w = int(existing_bbox[2] * w)
                            existing_h = int(existing_bbox[3] * h)
                            
                            if (abs(new_x - existing_x) < (defect_region.shape[1] + existing_w) // 2 and
                                abs(new_y - existing_y) < (defect_region.shape[0] + existing_h) // 2):
                                overlap = True
                                break
                        
                        if not overlap:
                            # Paste defect
                            paste_x1 = new_x - defect_region.shape[1] // 2
                            paste_y1 = new_y - defect_region.shape[0] // 2
                            paste_x2 = paste_x1 + defect_region.shape[1]
                            paste_y2 = paste_y1 + defect_region.shape[0]
                            
                            # Ensure paste coordinates are within image bounds
                            paste_x1 = max(0, paste_x1)
                            paste_y1 = max(0, paste_y1)
                            paste_x2 = min(w, paste_x2)
                            paste_y2 = min(h, paste_y2)
                            
                            if paste_x2 > paste_x1 and paste_y2 > paste_y1:
                                # Resize defect region if needed
                                region_h = paste_y2 - paste_y1
                                region_w = paste_x2 - paste_x1
                                if (region_h != defect_region.shape[0] or 
                                    region_w != defect_region.shape[1]):
                                    defect_region = cv2.resize(defect_region, (region_w, region_h))
                                
                                augmented_image[paste_y1:paste_y2, paste_x1:paste_x2] = defect_region
                                
                                # Add new annotation
                                new_annotation = defect.copy()
                                new_annotation['bbox'] = [
                                    new_x / w,  # x_center normalized
                                    new_y / h,  # y_center normalized
                                    region_w / w,  # width normalized
                                    region_h / h   # height normalized
                                ]
                                augmented_annotations.append(new_annotation)
                                break
        
        return augmented_image, augmented_annotations


class AdaptiveAugmentation:
    """Adaptive augmentation that adjusts based on training performance."""
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize adaptive augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.performance_history = []
        self.adaptation_threshold = 0.01  # mAP improvement threshold
        
    def update_performance(self, epoch: int, metrics: Dict):
        """
        Update performance history and adapt augmentation.
        
        Args:
            epoch: Current epoch
            metrics: Performance metrics
        """
        self.performance_history.append({
            'epoch': epoch,
            'map': metrics.get('mAP', 0),
            'loss': metrics.get('loss', float('inf'))
        })
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
    
    def should_increase_augmentation(self) -> bool:
        """Check if augmentation should be increased based on performance."""
        if len(self.performance_history) < 5:
            return False
        
        recent_maps = [h['map'] for h in self.performance_history[-5:]]
        trend = np.polyfit(range(len(recent_maps)), recent_maps, 1)[0]
        
        # If performance is stagnating or decreasing, increase augmentation
        return trend < self.adaptation_threshold
    
    def get_adaptive_multiplier(self) -> float:
        """Get adaptive multiplier for augmentation probabilities."""
        if self.should_increase_augmentation():
            return 1.2  # Increase augmentation by 20%
        else:
            return 1.0  # Keep current level


class IntegratedAugmentationPipeline:
    """Integrated advanced augmentation pipeline with progressive strategies."""
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize integrated augmentation pipeline.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        
        # Initialize components
        self.scheduler = ProgressiveAugmentationScheduler(config)
        self.small_defect_aug = SmallDefectAugmentation(config)
        self.adaptive_aug = AdaptiveAugmentation(config)
        
        # Initialize augmentation modules
        self.mosaic = MosaicAugmentation(config.image_size, config.mosaic_prob)
        self.copy_paste = CopyPasteAugmentation(config.copy_paste_prob, config.max_paste_objects)
        self.mixup = MixUpAugmentation(config.mixup_alpha, config.mixup_prob)
        
        if config.use_albumentations and ALBUMENTATIONS_AVAILABLE:
            self.albumentations = AlbumentationsAugmentation(config.image_size, train=True)
        else:
            self.albumentations = None
            if config.use_albumentations and not ALBUMENTATIONS_AVAILABLE:
                print("Warning: Albumentations not available. Skipping Albumentations augmentation.")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for progressive augmentation."""
        self.scheduler.set_epoch(epoch)
    
    def update_performance(self, epoch: int, metrics: Dict):
        """Update performance for adaptive augmentation."""
        self.adaptive_aug.update_performance(epoch, metrics)
    
    def __call__(self, images: List[np.ndarray], 
                 annotations_list: List[List[Dict]],
                 force_phase: Optional[AugmentationPhase] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply integrated augmentation pipeline.
        
        Args:
            images: List of images (at least 1, preferably 4 for mosaic)
            annotations_list: List of annotation lists
            force_phase: Force specific augmentation phase (for testing)
            
        Returns:
            Augmented image and annotations
        """
        if not images or not annotations_list:
            raise ValueError("At least one image and annotation list required")
        
        # Get current phase configuration
        if force_phase:
            original_epoch = self.scheduler.current_epoch
            # Temporarily set epoch to get desired phase
            phase_epochs = {
                AugmentationPhase.WARMUP: 0,
                AugmentationPhase.AGGRESSIVE: self.config.warmup_epochs + 1,
                AugmentationPhase.FINE_TUNE: self.config.warmup_epochs + self.config.aggressive_epochs + 1,
                AugmentationPhase.FINAL: self.scheduler.total_epochs + 1
            }
            self.scheduler.set_epoch(phase_epochs[force_phase])
            phase_config = self.scheduler.get_phase_config()
            self.scheduler.set_epoch(original_epoch)
        else:
            phase_config = self.scheduler.get_phase_config()
        
        # Apply adaptive multiplier
        adaptive_multiplier = self.adaptive_aug.get_adaptive_multiplier()
        
        # Adjust probabilities based on phase and adaptive feedback
        mosaic_prob = phase_config['mosaic_prob'] * adaptive_multiplier
        copy_paste_prob = phase_config['copy_paste_prob'] * adaptive_multiplier
        mixup_prob = phase_config['mixup_prob'] * adaptive_multiplier
        albumentations_prob = phase_config['albumentations_prob'] * adaptive_multiplier
        
        # Start with first image
        image, annotations = images[0], annotations_list[0]
        
        # Apply mosaic if we have enough images
        if len(images) >= 4 and random.random() < mosaic_prob:
            self.mosaic.prob = mosaic_prob
            image, annotations = self.mosaic(images, annotations_list)
        
        # Apply mixup if we have at least 2 images and didn't apply mosaic
        elif len(images) >= 2 and random.random() < mixup_prob:
            self.mixup.prob = mixup_prob
            image, annotations = self.mixup(
                images[0], annotations_list[0],
                images[1], annotations_list[1]
            )
        
        # Apply small defect augmentation
        if self.config.small_defect_boost:
            image, annotations = self.small_defect_aug.augment_small_defects(image, annotations)
        
        # Apply copy-paste augmentation
        if random.random() < copy_paste_prob:
            self.copy_paste.prob = copy_paste_prob
            image, annotations = self.copy_paste(image, annotations)
        
        # Apply Albumentations
        if self.albumentations and random.random() < albumentations_prob:
            image, annotations = self.albumentations(image, annotations)
        
        return image, annotations
    
    def get_augmentation_stats(self) -> Dict:
        """Get current augmentation statistics."""
        phase = self.scheduler.get_current_phase()
        phase_config = self.scheduler.get_phase_config()
        adaptive_multiplier = self.adaptive_aug.get_adaptive_multiplier()
        
        return {
            'current_phase': phase.value,
            'current_epoch': self.scheduler.current_epoch,
            'phase_config': phase_config,
            'adaptive_multiplier': adaptive_multiplier,
            'should_increase_aug': self.adaptive_aug.should_increase_augmentation(),
            'performance_history_length': len(self.adaptive_aug.performance_history)
        }


def create_pcb_augmentation_config(strategy: str = "balanced") -> AugmentationConfig:
    """
    Create augmentation configuration for different strategies.
    
    Args:
        strategy: Augmentation strategy ('conservative', 'balanced', 'aggressive')
        
    Returns:
        Augmentation configuration
    """
    if strategy == "conservative":
        return AugmentationConfig(
            mosaic_prob=0.3,
            copy_paste_prob=0.2,
            mixup_prob=0.1,
            albumentations_prob=0.6,
            small_defect_boost=True,
            small_defect_copy_prob=0.3,
            warmup_epochs=15,
            aggressive_epochs=100,
            fine_tune_epochs=150
        )
    
    elif strategy == "balanced":
        return AugmentationConfig(
            mosaic_prob=0.5,
            copy_paste_prob=0.3,
            mixup_prob=0.2,
            albumentations_prob=0.8,
            small_defect_boost=True,
            small_defect_copy_prob=0.5,
            warmup_epochs=10,
            aggressive_epochs=150,
            fine_tune_epochs=100
        )
    
    elif strategy == "aggressive":
        return AugmentationConfig(
            mosaic_prob=0.7,
            copy_paste_prob=0.5,
            mixup_prob=0.3,
            albumentations_prob=0.9,
            small_defect_boost=True,
            small_defect_copy_prob=0.7,
            warmup_epochs=5,
            aggressive_epochs=200,
            fine_tune_epochs=50
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_pcb_augmentation_pipeline(strategy: str = "balanced") -> IntegratedAugmentationPipeline:
    """
    Create integrated augmentation pipeline for PCB defect detection.
    
    Args:
        strategy: Augmentation strategy
        
    Returns:
        Configured augmentation pipeline
    """
    config = create_pcb_augmentation_config(strategy)
    return IntegratedAugmentationPipeline(config)