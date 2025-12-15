"""
Model ensemble implementation for PCB defect detection.
Implements multiple model prediction integration algorithms.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from collections import defaultdict

from ..core.interfaces import ModelInterface
from ..core.types import Detection, CLASS_MAPPING
from .postprocessing import PostProcessor


class EnsembleMethod:
    """Base class for ensemble methods."""
    
    def __init__(self, method: str = "average"):
        """
        Initialize ensemble method.
        
        Args:
            method: Ensemble method ("average", "weighted", "nms", "wbf")
        """
        self.method = method
        
    def combine_predictions(self, predictions_list: List[List[Detection]], 
                          weights: Optional[List[float]] = None) -> List[Detection]:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions_list: List of prediction lists from different models
            weights: Optional weights for each model
            
        Returns:
            Combined predictions
        """
        if self.method == "average":
            return self._average_ensemble(predictions_list, weights)
        elif self.method == "weighted":
            return self._weighted_ensemble(predictions_list, weights)
        elif self.method == "nms":
            return self._nms_ensemble(predictions_list, weights)
        elif self.method == "wbf":
            return self._weighted_boxes_fusion(predictions_list, weights)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def _average_ensemble(self, predictions_list: List[List[Detection]], 
                         weights: Optional[List[float]] = None) -> List[Detection]:
        """Average ensemble method."""
        if not predictions_list:
            return []
        
        # If no weights provided, use equal weights
        if weights is None:
            weights = [1.0 / len(predictions_list)] * len(predictions_list)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Group detections by spatial proximity and class
        grouped_detections = self._group_detections(predictions_list)
        
        # Average grouped detections
        final_detections = []
        for group in grouped_detections:
            if len(group) >= len(predictions_list) // 2:  # Majority voting
                avg_detection = self._average_detections(group, weights)
                final_detections.append(avg_detection)
        
        return final_detections
    
    def _weighted_ensemble(self, predictions_list: List[List[Detection]], 
                          weights: Optional[List[float]] = None) -> List[Detection]:
        """Weighted ensemble method based on model confidence."""
        if not predictions_list:
            return []
        
        # Calculate dynamic weights based on average confidence if not provided
        if weights is None:
            weights = []
            for preds in predictions_list:
                avg_conf = np.mean([det.confidence for det in preds]) if preds else 0.0
                weights.append(avg_conf)
        
        # Normalize weights
        total_weight = sum(weights) if sum(weights) > 0 else 1.0
        weights = [w / total_weight for w in weights]
        
        return self._average_ensemble(predictions_list, weights)
    
    def _nms_ensemble(self, predictions_list: List[List[Detection]], 
                     weights: Optional[List[float]] = None,
                     iou_threshold: float = 0.5) -> List[Detection]:
        """NMS-based ensemble method."""
        # Combine all predictions
        all_detections = []
        for i, preds in enumerate(predictions_list):
            weight = weights[i] if weights else 1.0
            for det in preds:
                # Adjust confidence by model weight
                adjusted_det = Detection(
                    bbox=det.bbox,
                    confidence=det.confidence * weight,
                    class_id=det.class_id,
                    class_name=det.class_name
                )
                all_detections.append(adjusted_det)
        
        # Apply NMS
        post_processor = PostProcessor(
            conf_threshold=0.0,  # Don't filter by confidence here
            iou_threshold=iou_threshold,
            nms_type="standard"
        )
        
        # Convert to format expected by post processor
        if all_detections:
            # Group by class for NMS
            class_detections = defaultdict(list)
            for det in all_detections:
                class_detections[det.class_id].append(det)
            
            final_detections = []
            for class_id, dets in class_detections.items():
                # Apply NMS per class
                nms_dets = self._apply_nms_to_detections(dets, iou_threshold)
                final_detections.extend(nms_dets)
            
            return final_detections
        
        return []
    
    def _weighted_boxes_fusion(self, predictions_list: List[List[Detection]], 
                              weights: Optional[List[float]] = None,
                              iou_threshold: float = 0.55,
                              skip_box_threshold: float = 0.0) -> List[Detection]:
        """Weighted Boxes Fusion (WBF) ensemble method."""
        if not predictions_list:
            return []
        
        if weights is None:
            weights = [1.0] * len(predictions_list)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Group detections by class
        class_detections = defaultdict(list)
        class_weights = defaultdict(list)
        
        for i, preds in enumerate(predictions_list):
            for det in preds:
                class_detections[det.class_id].append(det)
                class_weights[det.class_id].append(weights[i])
        
        # Apply WBF per class
        final_detections = []
        for class_id in class_detections:
            dets = class_detections[class_id]
            class_weights_list = class_weights[class_id]
            
            wbf_dets = self._wbf_single_class(
                dets, class_weights_list, iou_threshold, skip_box_threshold
            )
            final_detections.extend(wbf_dets)
        
        return final_detections
    
    def _group_detections(self, predictions_list: List[List[Detection]], 
                         iou_threshold: float = 0.5) -> List[List[Detection]]:
        """Group detections by spatial proximity and class."""
        all_detections = []
        for preds in predictions_list:
            all_detections.extend(preds)
        
        if not all_detections:
            return []
        
        groups = []
        used = set()
        
        for i, det1 in enumerate(all_detections):
            if i in used:
                continue
            
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(all_detections[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if same class and overlapping
                if (det1.class_id == det2.class_id and 
                    self._calculate_iou(det1.bbox, det2.bbox) > iou_threshold):
                    group.append(det2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _average_detections(self, detections: List[Detection], 
                           weights: List[float]) -> Detection:
        """Average a group of detections."""
        if not detections:
            raise ValueError("Cannot average empty detection list")
        
        # Use equal weights if not enough weights provided
        if len(weights) < len(detections):
            weights = [1.0 / len(detections)] * len(detections)
        
        # Average bbox coordinates
        avg_bbox = [0.0, 0.0, 0.0, 0.0]
        total_weight = 0.0
        avg_confidence = 0.0
        
        for i, det in enumerate(detections):
            weight = weights[min(i, len(weights)-1)]
            total_weight += weight
            
            for j in range(4):
                avg_bbox[j] += det.bbox[j] * weight
            avg_confidence += det.confidence * weight
        
        # Normalize
        if total_weight > 0:
            avg_bbox = [coord / total_weight for coord in avg_bbox]
            avg_confidence /= total_weight
        
        # Use the class from the first detection (they should all be the same)
        return Detection(
            bbox=tuple(avg_bbox),
            confidence=avg_confidence,
            class_id=detections[0].class_id,
            class_name=detections[0].class_name
        )
    
    def _apply_nms_to_detections(self, detections: List[Detection], 
                                iou_threshold: float) -> List[Detection]:
        """Apply NMS to a list of detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._calculate_iou(current.bbox, det.bbox) <= iou_threshold
            ]
        
        return keep
    
    def _wbf_single_class(self, detections: List[Detection], weights: List[float],
                         iou_threshold: float, skip_box_threshold: float) -> List[Detection]:
        """Apply Weighted Boxes Fusion for single class."""
        if not detections:
            return []
        
        # Create clusters of overlapping boxes
        clusters = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            cluster = [(det1, weights[i])]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                if self._calculate_iou(det1.bbox, det2.bbox) > iou_threshold:
                    cluster.append((det2, weights[j]))
                    used.add(j)
            
            clusters.append(cluster)
        
        # Fuse boxes in each cluster
        fused_detections = []
        for cluster in clusters:
            if len(cluster) == 1:
                det, weight = cluster[0]
                if det.confidence > skip_box_threshold:
                    fused_detections.append(det)
            else:
                fused_det = self._fuse_cluster(cluster)
                if fused_det.confidence > skip_box_threshold:
                    fused_detections.append(fused_det)
        
        return fused_detections
    
    def _fuse_cluster(self, cluster: List[Tuple[Detection, float]]) -> Detection:
        """Fuse a cluster of detections using weighted average."""
        total_weight = sum(weight for _, weight in cluster)
        
        # Weighted average of coordinates and confidence
        avg_bbox = [0.0, 0.0, 0.0, 0.0]
        avg_confidence = 0.0
        
        for det, weight in cluster:
            normalized_weight = weight / total_weight
            
            for i in range(4):
                avg_bbox[i] += det.bbox[i] * normalized_weight
            avg_confidence += det.confidence * normalized_weight
        
        # Use class from first detection
        first_det = cluster[0][0]
        
        return Detection(
            bbox=tuple(avg_bbox),
            confidence=avg_confidence,
            class_id=first_det.class_id,
            class_name=first_det.class_name
        )
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        # Convert to corner coordinates
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class ModelEnsemble:
    """Model ensemble manager for PCB defect detection."""
    
    def __init__(self, models: List[ModelInterface], 
                 ensemble_method: str = "weighted",
                 model_weights: Optional[List[float]] = None):
        """
        Initialize model ensemble.
        
        Args:
            models: List of trained models
            ensemble_method: Ensemble method to use
            model_weights: Optional weights for each model
        """
        self.models = models
        self.ensemble_method = EnsembleMethod(ensemble_method)
        self.model_weights = model_weights
        self.logger = logging.getLogger(__name__)
        
        # Validate inputs
        if not models:
            raise ValueError("At least one model is required for ensemble")
        
        if model_weights and len(model_weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, image: np.ndarray) -> List[Detection]:
        """
        Predict using ensemble of models.
        
        Args:
            image: Input image
            
        Returns:
            Ensemble predictions
        """
        # Get predictions from all models
        all_predictions = []
        
        for i, model in enumerate(self.models):
            try:
                predictions = model.predict(image)
                all_predictions.append(predictions)
                self.logger.debug(f"Model {i} produced {len(predictions)} detections")
            except Exception as e:
                self.logger.warning(f"Model {i} failed to predict: {e}")
                all_predictions.append([])  # Empty predictions for failed model
        
        # Combine predictions using ensemble method
        ensemble_predictions = self.ensemble_method.combine_predictions(
            all_predictions, self.model_weights
        )
        
        self.logger.info(f"Ensemble produced {len(ensemble_predictions)} final detections")
        return ensemble_predictions
    
    def predict_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Predict on batch of images using ensemble.
        
        Args:
            images: List of input images
            
        Returns:
            List of ensemble predictions for each image
        """
        batch_predictions = []
        
        for i, image in enumerate(images):
            predictions = self.predict(image)
            batch_predictions.append(predictions)
            self.logger.debug(f"Processed image {i+1}/{len(images)}")
        
        return batch_predictions
    
    def optimize_weights(self, validation_images: List[np.ndarray],
                        validation_targets: List[List[Detection]],
                        method: str = "grid_search") -> List[float]:
        """
        Optimize ensemble weights based on validation data.
        
        Args:
            validation_images: Validation images
            validation_targets: Ground truth detections
            method: Optimization method ("grid_search", "random_search")
            
        Returns:
            Optimized weights
        """
        if method == "grid_search":
            return self._grid_search_weights(validation_images, validation_targets)
        elif method == "random_search":
            return self._random_search_weights(validation_images, validation_targets)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _grid_search_weights(self, validation_images: List[np.ndarray],
                           validation_targets: List[List[Detection]]) -> List[float]:
        """Grid search for optimal weights."""
        from ..evaluation.evaluator import Evaluator
        
        evaluator = Evaluator()
        best_weights = None
        best_map = 0.0
        
        # Generate weight combinations
        num_models = len(self.models)
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Simple grid search (can be optimized for more models)
        if num_models == 2:
            for w1 in weight_options:
                w2 = 1.0 - w1
                weights = [w1, w2]
                map_score = self._evaluate_weights(
                    weights, validation_images, validation_targets, evaluator
                )
                
                if map_score > best_map:
                    best_map = map_score
                    best_weights = weights
        else:
            # For more models, use random sampling from grid
            import itertools
            import random
            
            # Generate all combinations (limited for computational efficiency)
            all_combinations = list(itertools.product(weight_options, repeat=num_models))
            
            # Sample subset if too many combinations
            if len(all_combinations) > 100:
                combinations = random.sample(all_combinations, 100)
            else:
                combinations = all_combinations
            
            for weights in combinations:
                # Normalize weights
                total = sum(weights)
                if total > 0:
                    normalized_weights = [w / total for w in weights]
                    map_score = self._evaluate_weights(
                        normalized_weights, validation_images, validation_targets, evaluator
                    )
                    
                    if map_score > best_map:
                        best_map = map_score
                        best_weights = normalized_weights
        
        if best_weights is None:
            # Fallback to equal weights
            best_weights = [1.0 / num_models] * num_models
        
        self.logger.info(f"Optimized weights: {best_weights}, mAP: {best_map:.4f}")
        return best_weights
    
    def _random_search_weights(self, validation_images: List[np.ndarray],
                             validation_targets: List[List[Detection]],
                             num_trials: int = 50) -> List[float]:
        """Random search for optimal weights."""
        from ..evaluation.evaluator import Evaluator
        import random
        
        evaluator = Evaluator()
        best_weights = None
        best_map = 0.0
        
        num_models = len(self.models)
        
        for _ in range(num_trials):
            # Generate random weights
            weights = [random.random() for _ in range(num_models)]
            
            # Normalize
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                
                map_score = self._evaluate_weights(
                    weights, validation_images, validation_targets, evaluator
                )
                
                if map_score > best_map:
                    best_map = map_score
                    best_weights = weights
        
        if best_weights is None:
            # Fallback to equal weights
            best_weights = [1.0 / num_models] * num_models
        
        self.logger.info(f"Optimized weights: {best_weights}, mAP: {best_map:.4f}")
        return best_weights
    
    def _evaluate_weights(self, weights: List[float],
                         validation_images: List[np.ndarray],
                         validation_targets: List[List[Detection]],
                         evaluator) -> float:
        """Evaluate ensemble with given weights."""
        # Temporarily set weights
        original_weights = self.model_weights
        self.model_weights = weights
        
        try:
            # Get ensemble predictions
            predictions = self.predict_batch(validation_images)
            
            # Calculate mAP
            map_score = evaluator.calculate_map(predictions, validation_targets)
            
            return map_score
        except Exception as e:
            self.logger.warning(f"Error evaluating weights {weights}: {e}")
            return 0.0
        finally:
            # Restore original weights
            self.model_weights = original_weights
    
    def save_ensemble_config(self, path: str) -> None:
        """Save ensemble configuration."""
        config = {
            "ensemble_method": self.ensemble_method.method,
            "model_weights": self.model_weights,
            "num_models": len(self.models)
        }
        
        import json
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_ensemble_config(self, path: str) -> None:
        """Load ensemble configuration."""
        import json
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.ensemble_method = EnsembleMethod(config["ensemble_method"])
        self.model_weights = config["model_weights"]