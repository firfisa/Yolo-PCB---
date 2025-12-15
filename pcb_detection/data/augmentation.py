"""
Data augmentation implementation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import random
import math

from ..core.interfaces import DataAugmentationInterface


class DataAugmentation(DataAugmentationInterface):
    """Data augmentation for PCB images."""
    
    def __init__(self, config: Dict):
        """
        Initialize data augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        
        # Default configuration
        self.rotation_range = config.get('rotation_range', (-15, 15))
        self.scale_range = config.get('scale_range', (0.8, 1.2))
        self.brightness_range = config.get('brightness_range', (-0.2, 0.2))
        self.contrast_range = config.get('contrast_range', (0.8, 1.2))
        self.flip_horizontal = config.get('flip_horizontal', True)
        self.flip_vertical = config.get('flip_vertical', True)
        self.augmentation_prob = config.get('augmentation_prob', 0.5)
        
    def random_rotation(self, image: np.ndarray, 
                       angle_range: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random rotation and return transformation matrix."""
        angle = random.uniform(angle_range[0], angle_range[1])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT_101)
        
        return rotated, rotation_matrix
        
    def random_scaling(self, image: np.ndarray, 
                      scale_range: Tuple[float, float]) -> Tuple[np.ndarray, float]:
        """Apply random scaling."""
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Resize image
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # If scaled image is larger, crop to original size
        if scale_factor > 1.0:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            scaled = scaled[start_y:start_y + h, start_x:start_x + w]
        # If scaled image is smaller, pad to original size
        elif scale_factor < 1.0:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            padded = np.full((h, w, image.shape[2]), 114, dtype=image.dtype)
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled
            scaled = padded
            
        return scaled, scale_factor
        
    def color_jittering(self, image: np.ndarray, 
                       brightness: float, contrast: float) -> np.ndarray:
        """Apply color jittering."""
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Apply brightness adjustment
        img_float = img_float + brightness * 255
        
        # Apply contrast adjustment
        img_float = img_float * contrast
        
        # Clip values to valid range
        img_float = np.clip(img_float, 0, 255)
        
        return img_float.astype(np.uint8)
        
    def random_flip(self, image: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """Apply random horizontal and vertical flips."""
        flip_h = self.flip_horizontal and random.random() < 0.5
        flip_v = self.flip_vertical and random.random() < 0.5
        
        result = image.copy()
        
        if flip_h:
            result = cv2.flip(result, 1)  # Horizontal flip
            
        if flip_v:
            result = cv2.flip(result, 0)  # Vertical flip
            
        return result, flip_h, flip_v
        
    def apply_augmentation(self, image: np.ndarray, 
                          annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Apply augmentation to image and annotations."""
        if random.random() > self.augmentation_prob:
            return image, annotations
            
        augmented_image = image.copy()
        augmented_annotations = [ann.copy() for ann in annotations]
        
        # Track transformations for bbox adjustment
        transformations = []
        
        # Apply random rotation
        if random.random() < 0.3:  # 30% chance
            augmented_image, rotation_matrix = self.random_rotation(
                augmented_image, self.rotation_range)
            transformations.append(('rotation', rotation_matrix))
            
        # Apply random scaling
        if random.random() < 0.3:  # 30% chance
            augmented_image, scale_factor = self.random_scaling(
                augmented_image, self.scale_range)
            transformations.append(('scaling', scale_factor))
            
        # Apply random flips
        if random.random() < 0.5:  # 50% chance
            augmented_image, flip_h, flip_v = self.random_flip(augmented_image)
            if flip_h or flip_v:
                transformations.append(('flip', (flip_h, flip_v)))
                
        # Apply color jittering
        if random.random() < 0.5:  # 50% chance
            brightness = random.uniform(*self.brightness_range)
            contrast = random.uniform(*self.contrast_range)
            augmented_image = self.color_jittering(augmented_image, brightness, contrast)
            
        # Transform bounding boxes
        augmented_annotations = self._transform_annotations(
            augmented_annotations, transformations, augmented_image.shape[:2])
            
        return augmented_image, augmented_annotations
        
    def _transform_annotations(self, annotations: List[Dict], 
                             transformations: List[Tuple], 
                             image_shape: Tuple[int, int]) -> List[Dict]:
        """Transform bounding box annotations based on applied transformations."""
        h, w = image_shape
        
        for annotation in annotations:
            bbox = annotation['bbox']  # [x_center, y_center, width, height] normalized
            
            # Convert to absolute coordinates
            x_center = bbox[0] * w
            y_center = bbox[1] * h
            bbox_width = bbox[2] * w
            bbox_height = bbox[3] * h
            
            # Apply transformations in reverse order
            for transform_type, transform_data in reversed(transformations):
                if transform_type == 'rotation':
                    rotation_matrix = transform_data
                    # Transform center point
                    point = np.array([[x_center, y_center]], dtype=np.float32)
                    point = point.reshape(-1, 1, 2)
                    transformed_point = cv2.transform(point, rotation_matrix)
                    x_center, y_center = transformed_point[0, 0]
                    
                elif transform_type == 'scaling':
                    scale_factor = transform_data
                    # Scaling affects both position and size
                    if scale_factor > 1.0:
                        # Image was cropped, adjust coordinates
                        crop_offset_x = (w * scale_factor - w) // 2
                        crop_offset_y = (h * scale_factor - h) // 2
                        x_center = (x_center * scale_factor - crop_offset_x)
                        y_center = (y_center * scale_factor - crop_offset_y)
                    else:
                        # Image was padded, adjust coordinates
                        pad_offset_x = (w - w * scale_factor) // 2
                        pad_offset_y = (h - h * scale_factor) // 2
                        x_center = x_center * scale_factor + pad_offset_x
                        y_center = y_center * scale_factor + pad_offset_y
                        
                    bbox_width *= scale_factor
                    bbox_height *= scale_factor
                    
                elif transform_type == 'flip':
                    flip_h, flip_v = transform_data
                    if flip_h:
                        x_center = w - x_center
                    if flip_v:
                        y_center = h - y_center
                        
            # Ensure bbox is within image bounds
            x_center = max(bbox_width/2, min(w - bbox_width/2, x_center))
            y_center = max(bbox_height/2, min(h - bbox_height/2, y_center))
            bbox_width = min(bbox_width, w)
            bbox_height = min(bbox_height, h)
            
            # Convert back to normalized coordinates
            annotation['bbox'] = [
                x_center / w,
                y_center / h,
                bbox_width / w,
                bbox_height / h
            ]
            
        # Filter out annotations with invalid bboxes
        valid_annotations = []
        for annotation in annotations:
            bbox = annotation['bbox']
            if (bbox[2] > 0.01 and bbox[3] > 0.01 and  # Minimum size
                0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1):  # Within bounds
                valid_annotations.append(annotation)
                
        return valid_annotations