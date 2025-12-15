"""
Report generation utilities for evaluation results.
"""

from typing import List, Dict, Any, Optional
import json
import csv
import os
from datetime import datetime
from pathlib import Path
import numpy as np

from ..core.types import Detection, EvaluationMetrics, CLASS_MAPPING


class ReportGenerator:
    """Generate comprehensive evaluation reports in various formats."""
    
    def __init__(self):
        """Initialize report generator."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_detailed_report(self, 
                               metrics: EvaluationMetrics,
                               predictions: List[List[Detection]],
                               ground_truths: List[List[Detection]],
                               model_name: str = "PCB_Detector",
                               dataset_name: str = "PCB_Dataset") -> Dict[str, Any]:
        """
        Generate a detailed evaluation report.
        
        Args:
            metrics: Evaluation metrics
            predictions: Predictions for each image
            ground_truths: Ground truths for each image
            model_name: Name of the model being evaluated
            dataset_name: Name of the dataset
            
        Returns:
            Comprehensive report dictionary
        """
        # Calculate additional statistics
        total_predictions = sum(len(pred_list) for pred_list in predictions)
        total_ground_truths = sum(len(gt_list) for gt_list in ground_truths)
        
        # Count detections per class
        pred_counts = {class_name: 0 for class_name in CLASS_MAPPING.values()}
        gt_counts = {class_name: 0 for class_name in CLASS_MAPPING.values()}
        
        for pred_list in predictions:
            for pred in pred_list:
                pred_counts[pred.class_name] += 1
        
        for gt_list in ground_truths:
            for gt in gt_list:
                gt_counts[gt.class_name] += 1
        
        report = {
            "metadata": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "evaluation_timestamp": self.timestamp,
                "total_images": len(predictions),
                "total_predictions": total_predictions,
                "total_ground_truths": total_ground_truths
            },
            "overall_metrics": {
                "mAP@0.5": round(metrics.map_50, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1_score": round(metrics.f1_score, 4)
            },
            "per_class_metrics": {
                class_name: {
                    "AP": round(ap, 4),
                    "predictions_count": pred_counts[class_name],
                    "ground_truths_count": gt_counts[class_name]
                }
                for class_name, ap in metrics.ap_per_class.items()
            },
            "class_distribution": {
                "predictions": pred_counts,
                "ground_truths": gt_counts
            }
        }
        
        return report
    
    def save_json_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report as JSON file.
        
        Args:
            report: Report dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def save_csv_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report as CSV file.
        
        Args:
            report: Report dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write metadata
            writer.writerow(["Metadata", ""])
            for key, value in report["metadata"].items():
                writer.writerow([key, value])
            writer.writerow([])
            
            # Write overall metrics
            writer.writerow(["Overall Metrics", ""])
            for key, value in report["overall_metrics"].items():
                writer.writerow([key, value])
            writer.writerow([])
            
            # Write per-class metrics
            writer.writerow(["Per-Class Metrics", ""])
            writer.writerow(["Class", "AP", "Predictions", "Ground Truths"])
            for class_name, metrics in report["per_class_metrics"].items():
                writer.writerow([
                    class_name,
                    metrics["AP"],
                    metrics["predictions_count"],
                    metrics["ground_truths_count"]
                ])
    
    def save_summary_table(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save a summary table in CSV format.
        
        Args:
            report: Report dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Model", "Dataset", "mAP@0.5", "Precision", "Recall", "F1-Score",
                "Mouse_bite_AP", "Open_circuit_AP", "Short_AP", "Spur_AP", "Spurious_copper_AP"
            ])
            
            # Data row
            row = [
                report["metadata"]["model_name"],
                report["metadata"]["dataset_name"],
                report["overall_metrics"]["mAP@0.5"],
                report["overall_metrics"]["precision"],
                report["overall_metrics"]["recall"],
                report["overall_metrics"]["f1_score"]
            ]
            
            # Add per-class APs
            for class_name in CLASS_MAPPING.values():
                row.append(report["per_class_metrics"][class_name]["AP"])
            
            writer.writerow(row)
    
    def export_predictions(self, 
                          predictions: List[List[Detection]], 
                          image_names: Optional[List[str]] = None,
                          output_path: str = "predictions.json") -> None:
        """
        Export predictions in structured format.
        
        Args:
            predictions: Predictions for each image
            image_names: Optional list of image names
            output_path: Output file path
        """
        if image_names is None:
            image_names = [f"image_{i:04d}" for i in range(len(predictions))]
        
        if len(image_names) != len(predictions):
            raise ValueError("Number of image names must match number of prediction lists")
        
        export_data = {
            "metadata": {
                "export_timestamp": self.timestamp,
                "total_images": len(predictions),
                "total_detections": sum(len(pred_list) for pred_list in predictions)
            },
            "predictions": []
        }
        
        for img_name, pred_list in zip(image_names, predictions):
            image_data = {
                "image_name": img_name,
                "detections": []
            }
            
            for detection in pred_list:
                det_data = {
                    "bbox": list(detection.bbox),
                    "confidence": round(detection.confidence, 4),
                    "class_id": detection.class_id,
                    "class_name": detection.class_name
                }
                image_data["detections"].append(det_data)
            
            export_data["predictions"].append(image_data)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def load_predictions(self, input_path: str) -> List[List[Detection]]:
        """
        Load predictions from exported JSON file.
        
        Args:
            input_path: Input file path
            
        Returns:
            List of prediction lists for each image
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        predictions = []
        for image_data in data["predictions"]:
            image_predictions = []
            for det_data in image_data["detections"]:
                detection = Detection(
                    bbox=tuple(det_data["bbox"]),
                    confidence=det_data["confidence"],
                    class_id=det_data["class_id"],
                    class_name=det_data["class_name"]
                )
                image_predictions.append(detection)
            predictions.append(image_predictions)
        
        return predictions
    
    def generate_comparison_report(self, 
                                 baseline_metrics: EvaluationMetrics,
                                 current_metrics: EvaluationMetrics,
                                 baseline_name: str = "Baseline",
                                 current_name: str = "Current") -> Dict[str, Any]:
        """
        Generate a comparison report between two models.
        
        Args:
            baseline_metrics: Baseline model metrics
            current_metrics: Current model metrics
            baseline_name: Name of baseline model
            current_name: Name of current model
            
        Returns:
            Comparison report dictionary
        """
        def calculate_improvement(baseline: float, current: float) -> Dict[str, float]:
            """Calculate absolute and relative improvement."""
            abs_improvement = current - baseline
            rel_improvement = (abs_improvement / baseline * 100) if baseline > 0 else 0.0
            return {
                "absolute": round(abs_improvement, 4),
                "relative_percent": round(rel_improvement, 2)
            }
        
        comparison = {
            "metadata": {
                "comparison_timestamp": self.timestamp,
                "baseline_model": baseline_name,
                "current_model": current_name
            },
            "overall_comparison": {
                "mAP@0.5": {
                    "baseline": round(baseline_metrics.map_50, 4),
                    "current": round(current_metrics.map_50, 4),
                    "improvement": calculate_improvement(baseline_metrics.map_50, current_metrics.map_50)
                },
                "precision": {
                    "baseline": round(baseline_metrics.precision, 4),
                    "current": round(current_metrics.precision, 4),
                    "improvement": calculate_improvement(baseline_metrics.precision, current_metrics.precision)
                },
                "recall": {
                    "baseline": round(baseline_metrics.recall, 4),
                    "current": round(current_metrics.recall, 4),
                    "improvement": calculate_improvement(baseline_metrics.recall, current_metrics.recall)
                },
                "f1_score": {
                    "baseline": round(baseline_metrics.f1_score, 4),
                    "current": round(current_metrics.f1_score, 4),
                    "improvement": calculate_improvement(baseline_metrics.f1_score, current_metrics.f1_score)
                }
            },
            "per_class_comparison": {}
        }
        
        # Compare per-class metrics
        for class_name in CLASS_MAPPING.values():
            baseline_ap = baseline_metrics.ap_per_class.get(class_name, 0.0)
            current_ap = current_metrics.ap_per_class.get(class_name, 0.0)
            
            comparison["per_class_comparison"][class_name] = {
                "baseline": round(baseline_ap, 4),
                "current": round(current_ap, 4),
                "improvement": calculate_improvement(baseline_ap, current_ap)
            }
        
        return comparison