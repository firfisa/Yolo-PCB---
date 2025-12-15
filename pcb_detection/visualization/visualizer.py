"""
Visualizer implementation for PCB defect detection results.
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2
import os

from ..core.interfaces import VisualizerInterface
from ..core.types import Detection, CLASS_LABELS, CLASS_COLORS, CLASS_MAPPING


class Visualizer(VisualizerInterface):
    """Visualizer for PCB defect detection results."""
    
    def __init__(self, class_names: Optional[List[str]] = None, 
                 colors: Optional[List[tuple]] = None):
        """
        Initialize visualizer.
        
        Args:
            class_names: List of class names (optional, uses default if None)
            colors: List of colors for each class (optional, uses default if None)
        """
        self.class_names = class_names or list(CLASS_MAPPING.values())
        self.colors = colors or list(CLASS_COLORS.values())
        self.class_labels = CLASS_LABELS
        
        # Font settings for text rendering
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.bbox_thickness = 2
        
    def draw_detections(self, image: np.ndarray, 
                       detections: List[Detection]) -> np.ndarray:
        """
        Draw detections on image with bounding boxes and single-letter labels.
        
        Args:
            image: Input image (H, W, C) in BGR format
            detections: List of Detection objects
            
        Returns:
            Image with drawn detections
        """
        if image is None or len(image.shape) != 3:
            raise ValueError("Image must be a 3D numpy array (H, W, C)")
            
        # Create a copy to avoid modifying original
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        for detection in detections:
            # Convert normalized coordinates to pixel coordinates
            x_center, y_center, width, height = detection.bbox
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Get color for this class
            color = self.colors[detection.class_id]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, self.bbox_thickness)
            
            # Prepare label text (single letter + confidence)
            single_letter = self.class_labels[detection.class_id]
            label_text = f"{single_letter}:{detection.confidence:.2f}"
            
            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, self.font, self.font_scale, self.font_thickness
            )
            
            # Position label above bounding box, or inside if too close to top
            label_y = y1 - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5
            label_x = x1
            
            # Ensure label doesn't go outside image bounds
            if label_x + text_width > w:
                label_x = w - text_width
            if label_x < 0:
                label_x = 0
                
            # Draw background rectangle for text
            cv2.rectangle(
                vis_image,
                (label_x, label_y - text_height - baseline),
                (label_x + text_width, label_y + baseline),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                label_text,
                (label_x, label_y),
                self.font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.font_thickness
            )
            
        return vis_image
        
    def create_comparison_image(self, image: np.ndarray,
                               gt_detections: List[Detection],
                               pred_detections: List[Detection]) -> np.ndarray:
        """
        Create side-by-side comparison image with ground truth on left and predictions on right.
        
        Args:
            image: Original image
            gt_detections: Ground truth detections
            pred_detections: Predicted detections
            
        Returns:
            Side-by-side comparison image
        """
        if image is None or len(image.shape) != 3:
            raise ValueError("Image must be a 3D numpy array (H, W, C)")
            
        # Draw detections on separate copies
        gt_image = self.draw_detections(image.copy(), gt_detections)
        pred_image = self.draw_detections(image.copy(), pred_detections)
        
        # Add labels to distinguish GT and predictions
        h, w = image.shape[:2]
        
        # Add "Ground Truth" label to left image
        cv2.putText(
            gt_image,
            "Ground Truth",
            (10, 30),
            self.font,
            0.8,
            (255, 255, 255),  # White text
            2
        )
        
        # Add black background for better visibility
        cv2.rectangle(gt_image, (5, 5), (150, 40), (0, 0, 0), -1)
        cv2.putText(
            gt_image,
            "Ground Truth",
            (10, 30),
            self.font,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Add "Predictions" label to right image
        cv2.rectangle(pred_image, (5, 5), (130, 40), (0, 0, 0), -1)
        cv2.putText(
            pred_image,
            "Predictions",
            (10, 30),
            self.font,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Concatenate images horizontally
        comparison_image = np.hstack([gt_image, pred_image])
        
        return comparison_image
        
    def create_comparison_grid(self, images: List[np.ndarray],
                              gt_detections: List[List[Detection]],
                              pred_detections: List[List[Detection]]) -> np.ndarray:
        """
        Create grid of comparison images for batch visualization.
        
        Args:
            images: List of input images
            gt_detections: List of ground truth detections for each image
            pred_detections: List of predicted detections for each image
            
        Returns:
            Grid layout of comparison images
        """
        if not images:
            raise ValueError("Images list cannot be empty")
            
        if len(images) != len(gt_detections) or len(images) != len(pred_detections):
            raise ValueError("Number of images, GT detections, and predictions must match")
            
        # Create comparison images for each input
        comparison_images = []
        for img, gt_dets, pred_dets in zip(images, gt_detections, pred_detections):
            comp_img = self.create_comparison_image(img, gt_dets, pred_dets)
            comparison_images.append(comp_img)
            
        # Calculate grid dimensions
        num_images = len(comparison_images)
        if num_images == 1:
            return comparison_images[0]
            
        # Calculate optimal grid layout (prefer wider grids)
        cols = min(3, num_images)  # Maximum 3 columns
        rows = (num_images + cols - 1) // cols
        
        # Get dimensions from first image
        img_h, img_w = comparison_images[0].shape[:2]
        
        # Create grid canvas
        grid_h = rows * img_h
        grid_w = cols * img_w
        grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Place images in grid
        for idx, comp_img in enumerate(comparison_images):
            row = idx // cols
            col = idx % cols
            
            y_start = row * img_h
            y_end = y_start + img_h
            x_start = col * img_w
            x_end = x_start + img_w
            
            # Resize image if necessary to fit grid
            if comp_img.shape[:2] != (img_h, img_w):
                comp_img = cv2.resize(comp_img, (img_w, img_h))
                
            grid_image[y_start:y_end, x_start:x_end] = comp_img
            
        return grid_image
        
    def create_batch_visualization(self, images: List[np.ndarray],
                                 gt_detections: List[List[Detection]],
                                 pred_detections: List[List[Detection]],
                                 max_images: int = 9) -> np.ndarray:
        """
        Create batch visualization with a limit on number of images.
        
        Args:
            images: List of input images
            gt_detections: List of ground truth detections
            pred_detections: List of predicted detections
            max_images: Maximum number of images to include in grid
            
        Returns:
            Grid visualization of comparison images
        """
        # Limit number of images
        if len(images) > max_images:
            images = images[:max_images]
            gt_detections = gt_detections[:max_images]
            pred_detections = pred_detections[:max_images]
            
        return self.create_comparison_grid(images, gt_detections, pred_detections)
        
    def save_visualization(self, image: np.ndarray, path: str) -> None:
        """
        Save visualization to file.
        
        Args:
            image: Image to save
            path: Output file path
        """
        if image is None:
            raise ValueError("Image cannot be None")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save image
        success = cv2.imwrite(path, image)
        if not success:
            raise RuntimeError(f"Failed to save image to {path}")
            
    def _optimize_small_region_annotation(self, bbox: Tuple[int, int, int, int], 
                                        image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Optimize label positioning for small regions to minimize text occlusion.
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            image_shape: Image shape (height, width)
            
        Returns:
            Optimal label position (x, y)
        """
        x1, y1, x2, y2 = bbox
        h, w = image_shape
        
        # Calculate bbox dimensions
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # For very small regions, place label outside the bbox
        if bbox_width < 30 or bbox_height < 20:
            # Try to place above first
            if y1 > 25:
                return x1, y1 - 5
            # If not enough space above, try below
            elif y2 + 25 < h:
                return x1, y2 + 20
            # If not enough space above or below, place to the right
            elif x2 + 50 < w:
                return x2 + 5, y1 + 15
            # Last resort: place to the left
            else:
                return max(0, x1 - 50), y1 + 15
        else:
            # For larger regions, place inside at top-left corner
            return x1 + 2, y1 + 15