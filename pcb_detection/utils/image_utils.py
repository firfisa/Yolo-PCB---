"""
Image utility functions.
"""

import cv2
import numpy as np
from typing import Tuple, List


class ImageUtils:
    """Utility functions for image operations."""
    
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            path: Path to image file
            
        Returns:
            Loaded image
        """
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not load image from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    @staticmethod
    def save_image(image: np.ndarray, path: str) -> None:
        """
        Save image to file.
        
        Args:
            image: Image to save
            path: Path to save file
        """
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        
    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to specified size.
        
        Args:
            image: Input image
            size: Target size (width, height)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, size)
        
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
        
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255] range.
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        return (image * 255.0).astype(np.uint8)
        
    @staticmethod
    def convert_bbox_format(bbox: Tuple[float, float, float, float],
                           from_format: str, to_format: str,
                           image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """
        Convert bounding box between different formats.
        
        Args:
            bbox: Bounding box coordinates
            from_format: Source format ('xywh', 'xyxy', 'cxcywh')
            to_format: Target format ('xywh', 'xyxy', 'cxcywh')
            image_size: Image size (width, height)
            
        Returns:
            Converted bounding box
        """
        # Implementation will be added in subsequent tasks
        pass