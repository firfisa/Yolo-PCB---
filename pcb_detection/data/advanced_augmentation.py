"""
Advanced data augmentation techniques for PCB defect detection.
"""

import cv2
import numpy as np
import random
from typing import List, Dict, Tuple, Optional

from ..core.types import CLASS_MAPPING

# Optional imports for advanced augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None
    ToTensorV2 = None
    ALBUMENTATIONS_AVAILABLE = False


class MosaicAugmentation:
    """Mosaic augmentation for YOLO training."""
    
    def __init__(self, image_size: int = 640, prob: float = 0.5):
        """
        Initialize Mosaic augmentation.
        
        Args:
            image_size: Target image size
            prob: Probability of applying mosaic
        """
        self.image_size = image_size
        self.prob = prob
        
    def __call__(self, images: List[np.ndarray], 
                 annotations_list: List[List[Dict]]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply mosaic augmentation to 4 images.
        
        Args:
            images: List of 4 images
            annotations_list: List of 4 annotation lists
            
        Returns:
            Tuple of (mosaic_image, combined_annotations)
        """
        if random.random() > self.prob or len(images) < 4:
            # Return first image if not applying mosaic
            return images[0], annotations_list[0]
            
        # Create mosaic
        mosaic_image = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        combined_annotations = []
        
        # Define quadrant positions
        positions = [
            (0, 0),  # Top-left
            (self.image_size // 2, 0),  # Top-right
            (0, self.image_size // 2),  # Bottom-left
            (self.image_size // 2, self.image_size // 2)  # Bottom-right
        ]
        
        quadrant_size = self.image_size // 2
        
        for i, (image, annotations) in enumerate(zip(images[:4], annotations_list[:4])):
            # Resize image to quadrant size
            resized_image = cv2.resize(image, (quadrant_size, quadrant_size))
            
            # Place in mosaic
            x_offset, y_offset = positions[i]
            mosaic_image[y_offset:y_offset + quadrant_size, 
                        x_offset:x_offset + quadrant_size] = resized_image
            
            # Adjust annotations
            scale_x = quadrant_size / image.shape[1]
            scale_y = quadrant_size / image.shape[0]
            
            for ann in annotations:
                bbox = ann['bbox'].copy()  # [x_center, y_center, width, height] normalized
                
                # Convert to absolute coordinates
                abs_x = bbox[0] * image.shape[1]
                abs_y = bbox[1] * image.shape[0]
                abs_w = bbox[2] * image.shape[1]
                abs_h = bbox[3] * image.shape[0]
                
                # Scale and offset
                new_x = (abs_x * scale_x + x_offset) / self.image_size
                new_y = (abs_y * scale_y + y_offset) / self.image_size
                new_w = (abs_w * scale_x) / self.image_size
                new_h = (abs_h * scale_y) / self.image_size
                
                # Check if bbox is still valid
                if (new_x > 0 and new_y > 0 and 
                    new_x + new_w < 1 and new_y + new_h < 1 and
                    new_w > 0.01 and new_h > 0.01):
                    
                    new_ann = ann.copy()
                    new_ann['bbox'] = [new_x, new_y, new_w, new_h]
                    combined_annotations.append(new_ann)
                    
        return mosaic_image, combined_annotations


class CopyPasteAugmentation:
    """Copy-paste augmentation for small defects."""
    
    def __init__(self, prob: float = 0.3, max_paste: int = 3):
        """
        Initialize Copy-paste augmentation.
        
        Args:
            prob: Probability of applying copy-paste
            max_paste: Maximum number of defects to paste
        """
        self.prob = prob
        self.max_paste = max_paste
        
    def __call__(self, image: np.ndarray, 
                 annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply copy-paste augmentation.
        
        Args:
            image: Input image
            annotations: List of annotations
            
        Returns:
            Tuple of (augmented_image, augmented_annotations)
        """
        if random.random() > self.prob or not annotations:
            return image, annotations
            
        augmented_image = image.copy()
        augmented_annotations = annotations.copy()
        
        h, w = image.shape[:2]
        
        # Select defects to copy
        num_paste = random.randint(1, min(self.max_paste, len(annotations)))
        defects_to_copy = random.sample(annotations, num_paste)
        
        for defect in defects_to_copy:
            bbox = defect['bbox']  # [x_center, y_center, width, height] normalized
            
            # Convert to absolute coordinates
            x_center = int(bbox[0] * w)
            y_center = int(bbox[1] * h)
            width = int(bbox[2] * w)
            height = int(bbox[3] * h)
            
            # Extract defect region
            x1 = max(0, x_center - width // 2)
            y1 = max(0, y_center - height // 2)
            x2 = min(w, x_center + width // 2)
            y2 = min(h, y_center + height // 2)
            
            if x2 > x1 and y2 > y1:
                defect_region = image[y1:y2, x1:x2].copy()
                
                # Find new location to paste
                for _ in range(10):  # Try up to 10 times
                    new_x = random.randint(width // 2, w - width // 2)
                    new_y = random.randint(height // 2, h - height // 2)
                    
                    # Check if location is free (no overlap with existing defects)
                    overlap = False
                    for existing_ann in augmented_annotations:
                        existing_bbox = existing_ann['bbox']
                        existing_x = int(existing_bbox[0] * w)
                        existing_y = int(existing_bbox[1] * h)
                        existing_w = int(existing_bbox[2] * w)
                        existing_h = int(existing_bbox[3] * h)
                        
                        if (abs(new_x - existing_x) < (width + existing_w) // 2 and
                            abs(new_y - existing_y) < (height + existing_h) // 2):
                            overlap = True
                            break
                            
                    if not overlap:
                        # Paste defect
                        paste_x1 = new_x - width // 2
                        paste_y1 = new_y - height // 2
                        paste_x2 = paste_x1 + defect_region.shape[1]
                        paste_y2 = paste_y1 + defect_region.shape[0]
                        
                        augmented_image[paste_y1:paste_y2, paste_x1:paste_x2] = defect_region
                        
                        # Add new annotation
                        new_annotation = defect.copy()
                        new_annotation['bbox'] = [
                            new_x / w,  # x_center normalized
                            new_y / h,  # y_center normalized
                            width / w,  # width normalized
                            height / h  # height normalized
                        ]
                        augmented_annotations.append(new_annotation)
                        break
                        
        return augmented_image, augmented_annotations


class MixUpAugmentation:
    """MixUp augmentation for PCB images."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, image1: np.ndarray, annotations1: List[Dict],
                 image2: np.ndarray, annotations2: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply MixUp augmentation to two images.
        
        Args:
            image1: First image
            annotations1: Annotations for first image
            image2: Second image
            annotations2: Annotations for second image
            
        Returns:
            Tuple of (mixed_image, combined_annotations)
        """
        if random.random() > self.prob:
            return image1, annotations1
            
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_image = (lam * image1 + (1 - lam) * image2).astype(np.uint8)
        
        # Combine annotations (keep all, as both images contribute)
        combined_annotations = annotations1 + annotations2
        
        return mixed_image, combined_annotations


class AlbumentationsAugmentation:
    """Advanced augmentation using Albumentations library."""
    
    def __init__(self, image_size: int = 640, train: bool = True):
        """
        Initialize Albumentations augmentation pipeline.
        
        Args:
            image_size: Target image size
            train: Whether this is for training (more aggressive augmentation)
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("Albumentations is required for AlbumentationsAugmentation. "
                            "Install it with: pip install albumentations")
        
        self.image_size = image_size
        
        if train:
            self.transform = A.Compose([
                A.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.PiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RandomGamma(p=0.2),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), 
                               hole_width_range=(8, 32), p=0.2),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
    def __call__(self, image: np.ndarray, 
                 annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply Albumentations augmentation.
        
        Args:
            image: Input image
            annotations: List of annotations
            
        Returns:
            Tuple of (augmented_image, augmented_annotations)
        """
        if not annotations:
            # For empty annotations, use a transform without bbox_params
            simple_transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
            ])
            transformed = simple_transform(image=image)
            return transformed['image'], []
            
        # Prepare bboxes and labels for Albumentations
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x_center, y_center, width, height] normalized
            bboxes.append(bbox)
            class_labels.append(ann['class_id'])
            
        # Apply transformation
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        # Convert back to annotation format
        augmented_annotations = []
        for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
            augmented_annotations.append({
                'class_id': class_id,
                'bbox': list(bbox)
            })
            
        return transformed['image'], augmented_annotations


class PCBAdvancedAugmentation:
    """Combined advanced augmentation pipeline for PCB defect detection."""
    
    def __init__(self, 
                 image_size: int = 640,
                 mosaic_prob: float = 0.5,
                 copy_paste_prob: float = 0.3,
                 mixup_prob: float = 0.2,
                 use_albumentations: bool = True):
        """
        Initialize advanced augmentation pipeline.
        
        Args:
            image_size: Target image size
            mosaic_prob: Probability of mosaic augmentation
            copy_paste_prob: Probability of copy-paste augmentation
            mixup_prob: Probability of mixup augmentation
            use_albumentations: Whether to use Albumentations
        """
        self.image_size = image_size
        
        self.mosaic = MosaicAugmentation(image_size, mosaic_prob)
        self.copy_paste = CopyPasteAugmentation(copy_paste_prob)
        self.mixup = MixUpAugmentation(prob=mixup_prob)
        
        if use_albumentations and ALBUMENTATIONS_AVAILABLE:
            self.albumentations = AlbumentationsAugmentation(image_size, train=True)
        else:
            self.albumentations = None
            if use_albumentations and not ALBUMENTATIONS_AVAILABLE:
                print("Warning: Albumentations not available. Skipping Albumentations augmentation.")
            
    def __call__(self, images: List[np.ndarray], 
                 annotations_list: List[List[Dict]]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply advanced augmentation pipeline.
        
        Args:
            images: List of images (at least 1, preferably 4 for mosaic)
            annotations_list: List of annotation lists
            
        Returns:
            Tuple of (augmented_image, augmented_annotations)
        """
        if len(images) >= 4:
            # Try mosaic first
            image, annotations = self.mosaic(images, annotations_list)
        elif len(images) >= 2:
            # Try mixup
            image, annotations = self.mixup(
                images[0], annotations_list[0],
                images[1], annotations_list[1]
            )
        else:
            image, annotations = images[0], annotations_list[0]
            
        # Apply copy-paste
        image, annotations = self.copy_paste(image, annotations)
        
        # Apply Albumentations if available
        if self.albumentations:
            image, annotations = self.albumentations(image, annotations)
            
        return image, annotations