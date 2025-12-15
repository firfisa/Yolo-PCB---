"""
Post-processing module for YOLO detection results.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from ..core.types import Detection, CLASS_MAPPING


class NMSProcessor:
    """Non-Maximum Suppression processor with multiple algorithms."""
    
    def __init__(self, 
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 max_detections: int = 300,
                 multi_label: bool = True):
        """
        Initialize NMS processor.
        
        Args:
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep
            multi_label: Whether to apply NMS per class or globally
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.multi_label = multi_label
        
    def __call__(self, predictions: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply NMS to predictions.
        
        Args:
            predictions: Tensor of shape [batch_size, num_boxes, 6] 
                        (x, y, w, h, conf, class_id)
                        
        Returns:
            List of filtered predictions for each batch item
        """
        batch_size = predictions.shape[0]
        results = []
        
        for i in range(batch_size):
            pred = predictions[i]
            
            # Filter by confidence
            if len(pred) > 0:
                conf_mask = pred[:, 4] > self.conf_threshold
                pred = pred[conf_mask]
            
            if len(pred) == 0:
                results.append(torch.empty((0, 6), device=predictions.device))
                continue
                
            # Apply NMS
            if self.multi_label:
                filtered_pred = self._multi_class_nms(pred)
            else:
                filtered_pred = self._single_class_nms(pred)
                
            # Limit number of detections
            if len(filtered_pred) > self.max_detections:
                # Sort by confidence and keep top detections
                conf_sorted_idx = torch.argsort(filtered_pred[:, 4], descending=True)
                filtered_pred = filtered_pred[conf_sorted_idx[:self.max_detections]]
                
            results.append(filtered_pred)
            
        return results
        
    def _multi_class_nms(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply NMS per class."""
        keep_indices = []
        
        # Get unique classes
        unique_classes = torch.unique(predictions[:, 5])
        
        for class_id in unique_classes:
            # Filter predictions for this class
            class_mask = predictions[:, 5] == class_id
            class_preds = predictions[class_mask]
            
            if len(class_preds) == 0:
                continue
                
            # Apply NMS for this class
            class_keep = self._nms_single_class(class_preds)
            
            # Get original indices
            original_indices = torch.where(class_mask)[0]
            keep_indices.extend(original_indices[class_keep].tolist())
            
        if len(keep_indices) == 0:
            return torch.empty((0, 6), device=predictions.device)
            
        return predictions[keep_indices]
        
    def _single_class_nms(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply NMS globally across all classes."""
        keep_indices = self._nms_single_class(predictions)
        return predictions[keep_indices]
        
    def _nms_single_class(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply NMS to single class predictions."""
        if len(predictions) == 0:
            return torch.empty(0, dtype=torch.long, device=predictions.device)
            
        # Sort by confidence
        conf_sorted_idx = torch.argsort(predictions[:, 4], descending=True)
        
        keep = []
        while len(conf_sorted_idx) > 0:
            # Keep highest confidence detection
            current_idx = conf_sorted_idx[0]
            keep.append(current_idx)
            
            if len(conf_sorted_idx) == 1:
                break
                
            # Calculate IoU with remaining detections
            current_box = predictions[current_idx:current_idx+1, :4]
            remaining_boxes = predictions[conf_sorted_idx[1:], :4]
            
            ious = self._calculate_iou(current_box, remaining_boxes)
            
            # Keep detections with IoU below threshold
            keep_mask = ious < self.iou_threshold
            conf_sorted_idx = conf_sorted_idx[1:][keep_mask]
            
        return torch.tensor(keep, dtype=torch.long, device=predictions.device)
        
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between boxes in center format (x, y, w, h)."""
        # Convert to corner format
        box1_corners = self._center_to_corners(box1)
        box2_corners = self._center_to_corners(box2)
        
        # Calculate intersection
        inter_min = torch.max(box1_corners[:, :2], box2_corners[:, :2])
        inter_max = torch.min(box1_corners[:, 2:], box2_corners[:, 2:])
        inter_wh = torch.clamp(inter_max - inter_min, min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        
        # Calculate areas
        box1_area = box1[:, 2] * box1[:, 3]
        box2_area = box2[:, 2] * box2[:, 3]
        
        # Calculate union
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        return iou
        
    def _center_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert center format (x, y, w, h) to corner format (x1, y1, x2, y2)."""
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)


class SoftNMSProcessor:
    """Soft Non-Maximum Suppression processor."""
    
    def __init__(self,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 sigma: float = 0.5,
                 method: str = 'gaussian'):
        """
        Initialize Soft NMS processor.
        
        Args:
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for suppression
            sigma: Gaussian parameter for soft suppression
            method: Suppression method ('linear' or 'gaussian')
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.sigma = sigma
        self.method = method
        
    def __call__(self, predictions: torch.Tensor) -> List[torch.Tensor]:
        """Apply Soft NMS to predictions."""
        batch_size = predictions.shape[0]
        results = []
        
        for i in range(batch_size):
            pred = predictions[i]
            
            # Filter by confidence
            conf_mask = pred[:, 4] > self.conf_threshold
            pred = pred[conf_mask]
            
            if len(pred) == 0:
                results.append(torch.empty((0, 6), device=predictions.device))
                continue
                
            # Apply Soft NMS
            filtered_pred = self._soft_nms(pred)
            results.append(filtered_pred)
            
        return results
        
    def _soft_nms(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply Soft NMS algorithm."""
        # Clone predictions to avoid modifying original
        preds = predictions.clone()
        
        # Sort by confidence
        conf_sorted_idx = torch.argsort(preds[:, 4], descending=True)
        preds = preds[conf_sorted_idx]
        
        keep = []
        
        for i in range(len(preds)):
            if preds[i, 4] < self.conf_threshold:
                continue
                
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            if i < len(preds) - 1:
                current_box = preds[i:i+1, :4]
                remaining_boxes = preds[i+1:, :4]
                
                ious = self._calculate_iou(current_box, remaining_boxes)
                
                # Apply soft suppression
                if self.method == 'gaussian':
                    # Gaussian suppression
                    weights = torch.exp(-(ious ** 2) / self.sigma)
                else:
                    # Linear suppression
                    weights = torch.where(
                        ious > self.iou_threshold,
                        1 - ious,
                        torch.ones_like(ious)
                    )
                    
                # Update confidences
                preds[i+1:, 4] *= weights
                
        if len(keep) == 0:
            return torch.empty((0, 6), device=predictions.device)
            
        return preds[keep]
        
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between boxes."""
        # Convert to corner format
        box1_corners = self._center_to_corners(box1)
        box2_corners = self._center_to_corners(box2)
        
        # Calculate intersection
        inter_min = torch.max(box1_corners[:, :2], box2_corners[:, :2])
        inter_max = torch.min(box1_corners[:, 2:], box2_corners[:, 2:])
        inter_wh = torch.clamp(inter_max - inter_min, min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        
        # Calculate areas
        box1_area = box1[:, 2] * box1[:, 3]
        box2_area = box2[:, 2] * box2[:, 3]
        
        # Calculate union
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        return iou
        
    def _center_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert center format to corner format."""
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)


class PostProcessor:
    """Complete post-processing pipeline for YOLO detections."""
    
    def __init__(self,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 max_detections: int = 300,
                 nms_type: str = 'standard',
                 multi_scale: bool = True):
        """
        Initialize post-processor.
        
        Args:
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections
            nms_type: Type of NMS ('standard' or 'soft')
            multi_scale: Whether to handle multi-scale predictions
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.multi_scale = multi_scale
        
        # Initialize NMS processor
        if nms_type == 'soft':
            self.nms_processor = SoftNMSProcessor(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
        else:
            self.nms_processor = NMSProcessor(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections
            )
            
    def __call__(self, 
                 predictions: torch.Tensor,
                 image_shapes: Optional[List[Tuple[int, int]]] = None,
                 input_size: int = 640) -> List[List[Detection]]:
        """
        Process predictions to final detections.
        
        Args:
            predictions: Raw model predictions
            image_shapes: Original image shapes for coordinate scaling
            input_size: Model input size for coordinate scaling
            
        Returns:
            List of detection lists for each batch item
        """
        # Apply NMS
        filtered_predictions = self.nms_processor(predictions)
        
        # Convert to Detection objects
        results = []
        for i, pred in enumerate(filtered_predictions):
            detections = []
            
            if len(pred) > 0:
                # Get image shape for coordinate scaling
                if image_shapes is not None:
                    img_h, img_w = image_shapes[i]
                else:
                    img_h, img_w = input_size, input_size
                    
                # Convert predictions to Detection objects
                for detection in pred:
                    x, y, w, h, conf, class_id = detection.cpu().numpy()
                    
                    # Scale coordinates to original image size
                    scale_x = img_w / input_size
                    scale_y = img_h / input_size
                    
                    # Convert to normalized coordinates
                    norm_x = (x * scale_x) / img_w
                    norm_y = (y * scale_y) / img_h
                    norm_w = (w * scale_x) / img_w
                    norm_h = (h * scale_y) / img_h
                    
                    # Create Detection object
                    det = Detection(
                        bbox=(norm_x, norm_y, norm_w, norm_h),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=CLASS_MAPPING[int(class_id)]
                    )
                    detections.append(det)
                    
            results.append(detections)
            
        return results
        
    def filter_by_confidence(self, 
                           predictions: torch.Tensor, 
                           threshold: float) -> torch.Tensor:
        """Filter predictions by confidence threshold."""
        conf_mask = predictions[..., 4] > threshold
        return predictions[conf_mask]
        
    def scale_coordinates(self,
                         boxes: torch.Tensor,
                         input_size: int,
                         target_size: Tuple[int, int]) -> torch.Tensor:
        """Scale bounding box coordinates."""
        target_h, target_w = target_size
        
        # Calculate scaling factors
        scale_x = target_w / input_size
        scale_y = target_h / input_size
        
        # Scale coordinates
        scaled_boxes = boxes.clone()
        scaled_boxes[:, 0] *= scale_x  # x
        scaled_boxes[:, 1] *= scale_y  # y
        scaled_boxes[:, 2] *= scale_x  # w
        scaled_boxes[:, 3] *= scale_y  # h
        
        return scaled_boxes
        
    def convert_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert center format (x, y, w, h) to corner format (x1, y1, x2, y2)."""
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
        
    def convert_to_center(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert corner format (x1, y1, x2, y2) to center format (x, y, w, h)."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        return torch.stack([x_center, y_center, width, height], dim=1)


class MultiScalePostProcessor:
    """Post-processor for multi-scale YOLO predictions."""
    
    def __init__(self, 
                 strides: List[int] = [8, 16, 32],
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize multi-scale post-processor.
        
        Args:
            strides: Stride values for each scale
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.strides = strides
        self.post_processor = PostProcessor(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
    def __call__(self, 
                 multi_scale_predictions: List[torch.Tensor],
                 image_shapes: Optional[List[Tuple[int, int]]] = None) -> List[List[Detection]]:
        """
        Process multi-scale predictions.
        
        Args:
            multi_scale_predictions: List of predictions from different scales
            image_shapes: Original image shapes
            
        Returns:
            List of detection lists for each batch item
        """
        # Concatenate predictions from all scales
        all_predictions = []
        
        for scale_idx, predictions in enumerate(multi_scale_predictions):
            # Decode predictions for this scale
            decoded = self._decode_scale_predictions(predictions, scale_idx)
            all_predictions.append(decoded)
            
        # Concatenate all scales
        combined_predictions = torch.cat(all_predictions, dim=1)
        
        # Apply post-processing
        return self.post_processor(combined_predictions, image_shapes)
        
    def _decode_scale_predictions(self, 
                                predictions: torch.Tensor, 
                                scale_idx: int) -> torch.Tensor:
        """Decode predictions for a specific scale."""
        batch_size, num_anchors, height, width, num_outputs = predictions.shape
        stride = self.strides[scale_idx]
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=predictions.device),
            torch.arange(width, device=predictions.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).unsqueeze(0).expand(batch_size, num_anchors, -1, -1, -1)
        
        # Decode coordinates
        xy = (predictions[..., :2] + grid) * stride
        wh = predictions[..., 2:4]  # Keep as is for now
        conf = predictions[..., 4:5]
        cls = predictions[..., 5:]
        
        # Get class predictions
        class_conf, class_id = torch.max(cls, dim=-1, keepdim=True)
        final_conf = conf * class_conf
        
        # Reshape to [batch_size, num_detections, 6]
        decoded = torch.cat([xy, wh, final_conf, class_id.float()], dim=-1)
        decoded = decoded.view(batch_size, -1, 6)
        
        return decoded