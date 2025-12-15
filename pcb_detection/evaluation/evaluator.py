"""
Evaluator implementation.
"""

from typing import List, Dict, Union
import numpy as np
import json
import csv
import os
from pathlib import Path

from ..core.interfaces import EvaluatorInterface
from ..core.types import Detection, EvaluationMetrics, CLASS_MAPPING
from .metrics import MetricsCalculator


class Evaluator(EvaluatorInterface):
    """Evaluator for PCB defect detection performance."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for evaluation
        """
        self.iou_threshold = iou_threshold
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
    def calculate_map(self, predictions: List[List[Detection]], 
                     ground_truths: List[List[Detection]]) -> float:
        """
        Calculate mean Average Precision at IoU=0.5.
        
        Args:
            predictions: List of predictions for each image
            ground_truths: List of ground truths for each image
            
        Returns:
            mAP@0.5 value
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")
        
        # Flatten all predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        
        for pred_list, gt_list in zip(predictions, ground_truths):
            all_predictions.extend(pred_list)
            all_ground_truths.extend(gt_list)
        
        # Calculate AP for each class
        class_aps = []
        for class_id in range(5):  # 5 PCB defect classes
            ap = MetricsCalculator.calculate_ap_single_class(
                all_predictions, all_ground_truths, class_id, self.iou_threshold
            )
            class_aps.append(ap)
        
        # Return mean of all class APs
        return np.mean(class_aps) if class_aps else 0.0
        
    def calculate_ap_per_class(self, predictions: List[List[Detection]], 
                              ground_truths: List[List[Detection]]) -> Dict[str, float]:
        """
        Calculate Average Precision per class.
        
        Args:
            predictions: List of predictions for each image
            ground_truths: List of ground truths for each image
            
        Returns:
            Dictionary mapping class names to AP values
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")
        
        # Flatten all predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        
        for pred_list, gt_list in zip(predictions, ground_truths):
            all_predictions.extend(pred_list)
            all_ground_truths.extend(gt_list)
        
        # Calculate AP for each class
        ap_per_class = {}
        for class_id, class_name in CLASS_MAPPING.items():
            ap = MetricsCalculator.calculate_ap_single_class(
                all_predictions, all_ground_truths, class_id, self.iou_threshold
            )
            ap_per_class[class_name] = ap
        
        return ap_per_class
        
    def calculate_map_multi_iou(self, predictions: List[List[Detection]], 
                               ground_truths: List[List[Detection]]) -> Dict[str, float]:
        """
        Calculate mAP at multiple IoU thresholds.
        
        Args:
            predictions: List of predictions for each image
            ground_truths: List of ground truths for each image
            
        Returns:
            Dictionary mapping IoU thresholds to mAP values
        """
        map_results = {}
        
        for iou_thresh in self.iou_thresholds:
            # Temporarily change threshold
            original_threshold = self.iou_threshold
            self.iou_threshold = iou_thresh
            
            # Calculate mAP at this threshold
            map_value = self.calculate_map(predictions, ground_truths)
            map_results[f"mAP@{iou_thresh:.2f}"] = map_value
            
            # Restore original threshold
            self.iou_threshold = original_threshold
        
        return map_results
        
    def generate_metrics_report(self, predictions: List[List[Detection]], 
                               ground_truths: List[List[Detection]]) -> EvaluationMetrics:
        """
        Generate comprehensive evaluation metrics.
        
        Args:
            predictions: List of predictions for each image
            ground_truths: List of ground truths for each image
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")
        
        # Calculate mAP@0.5
        map_50 = self.calculate_map(predictions, ground_truths)
        
        # Calculate AP per class
        ap_per_class = self.calculate_ap_per_class(predictions, ground_truths)
        
        # Calculate overall precision and recall
        all_predictions = []
        all_ground_truths = []
        
        for pred_list, gt_list in zip(predictions, ground_truths):
            all_predictions.extend(pred_list)
            all_ground_truths.extend(gt_list)
        
        precision, recall = MetricsCalculator.calculate_precision_recall(
            all_predictions, all_ground_truths, self.iou_threshold
        )
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            map_50=map_50,
            ap_per_class=ap_per_class,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
        
    def save_results(self, results: Union[EvaluationMetrics, Dict], path: str, 
                    format: str = "json") -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: EvaluationMetrics object or dictionary to save
            path: Output file path
            format: Output format ("json" or "csv")
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Convert EvaluationMetrics to dict if needed
        if isinstance(results, EvaluationMetrics):
            results_dict = {
                "mAP@0.5": results.map_50,
                "precision": results.precision,
                "recall": results.recall,
                "f1_score": results.f1_score,
                **{f"AP_{class_name}": ap for class_name, ap in results.ap_per_class.items()}
            }
        else:
            results_dict = results
        
        if format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(results_dict, f, indent=2)
        elif format.lower() == "csv":
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for key, value in results_dict.items():
                    writer.writerow([key, value])
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")
    
    def evaluate_multi_threshold(self, predictions: List[List[Detection]], 
                                ground_truths: List[List[Detection]]) -> Dict[str, float]:
        """
        Evaluate at multiple IoU thresholds for comprehensive analysis.
        
        Args:
            predictions: List of predictions for each image
            ground_truths: List of ground truths for each image
            
        Returns:
            Dictionary with mAP values at different IoU thresholds
        """
        return self.calculate_map_multi_iou(predictions, ground_truths)