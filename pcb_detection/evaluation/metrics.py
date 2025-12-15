"""
Metrics calculation utilities.
"""

from typing import List, Dict, Tuple
import numpy as np

from ..core.types import Detection


class MetricsCalculator:
    """Utility class for calculating evaluation metrics."""
    
    @staticmethod
    def calculate_iou(box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            box1: First bounding box (x, y, w, h) normalized coordinates
            box2: Second bounding box (x, y, w, h) normalized coordinates
            
        Returns:
            IoU value between 0 and 1
        """
        # Convert from (x_center, y_center, width, height) to (x1, y1, x2, y2)
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        # Convert center coordinates to corner coordinates
        x1_1 = x1_1 - w1 / 2
        y1_1 = y1_1 - h1 / 2
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        
        x1_2 = x1_2 - w2 / 2
        y1_2 = y1_2 - h2 / 2
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
        
    @staticmethod
    def calculate_precision_recall(predictions: List[Detection],
                                  ground_truths: List[Detection],
                                  iou_threshold: float = 0.5) -> Tuple[float, float]:
        """
        Calculate precision and recall for a single image.
        
        Args:
            predictions: Predicted detections
            ground_truths: Ground truth detections
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (precision, recall)
        """
        if len(predictions) == 0:
            return 0.0, 0.0 if len(ground_truths) == 0 else 0.0
        
        if len(ground_truths) == 0:
            return 0.0, 0.0
        
        # Sort predictions by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        # Track which ground truths have been matched
        gt_matched = [False] * len(ground_truths)
        tp = 0  # True positives
        
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx] or pred.class_id != gt.class_id:
                    continue
                
                iou = MetricsCalculator.calculate_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is above threshold
            if best_iou >= iou_threshold and best_gt_idx != -1:
                gt_matched[best_gt_idx] = True
                tp += 1
        
        precision = tp / len(predictions) if len(predictions) > 0 else 0.0
        recall = tp / len(ground_truths) if len(ground_truths) > 0 else 0.0
        
        return precision, recall
    
    @staticmethod
    def calculate_ap_single_class(predictions: List[Detection],
                                 ground_truths: List[Detection],
                                 class_id: int,
                                 iou_threshold: float = 0.5) -> float:
        """
        Calculate Average Precision for a single class.
        
        Args:
            predictions: All predicted detections
            ground_truths: All ground truth detections
            class_id: Class ID to calculate AP for
            iou_threshold: IoU threshold for matching
            
        Returns:
            Average Precision for the class
        """
        # Filter predictions and ground truths for this class
        class_preds = [p for p in predictions if p.class_id == class_id]
        class_gts = [gt for gt in ground_truths if gt.class_id == class_id]
        
        if len(class_gts) == 0:
            return 0.0
        
        if len(class_preds) == 0:
            return 0.0
        
        # Sort predictions by confidence (descending)
        class_preds = sorted(class_preds, key=lambda x: x.confidence, reverse=True)
        
        # Track which ground truths have been matched
        gt_matched = [False] * len(class_gts)
        
        # Calculate precision and recall at each prediction
        precisions = []
        recalls = []
        
        tp = 0
        fp = 0
        
        for pred in class_preds:
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(class_gts):
                if gt_matched[gt_idx]:
                    continue
                
                iou = MetricsCalculator.calculate_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is above threshold
            if best_iou >= iou_threshold and best_gt_idx != -1:
                gt_matched[best_gt_idx] = True
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / len(class_gts)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find precisions for recalls >= t
            p_interp = 0.0
            for i, r in enumerate(recalls):
                if r >= t:
                    p_interp = max(p_interp, precisions[i])
            ap += p_interp / 11.0
        
        return ap