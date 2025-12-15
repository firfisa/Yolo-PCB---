"""
Utility functions for comprehensive evaluation workflows.
"""

from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path

from ..core.types import Detection, EvaluationMetrics
from .evaluator import Evaluator
from .report_generator import ReportGenerator


class EvaluationPipeline:
    """Complete evaluation pipeline for PCB defect detection."""
    
    def __init__(self, iou_threshold: float = 0.5, output_dir: str = "evaluation_results"):
        """
        Initialize evaluation pipeline.
        
        Args:
            iou_threshold: IoU threshold for evaluation
            output_dir: Directory to save evaluation results
        """
        self.evaluator = Evaluator(iou_threshold=iou_threshold)
        self.report_generator = ReportGenerator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_complete_evaluation(self,
                              predictions: List[List[Detection]],
                              ground_truths: List[List[Detection]],
                              model_name: str = "PCB_Detector",
                              dataset_name: str = "PCB_Dataset",
                              image_names: Optional[List[str]] = None,
                              save_predictions: bool = True) -> Dict[str, any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            predictions: Predictions for each image
            ground_truths: Ground truths for each image
            model_name: Name of the model being evaluated
            dataset_name: Name of the dataset
            image_names: Optional list of image names
            save_predictions: Whether to save predictions to file
            
        Returns:
            Dictionary containing all evaluation results and file paths
        """
        # Generate evaluation metrics
        metrics = self.evaluator.generate_metrics_report(predictions, ground_truths)
        
        # Generate detailed report
        detailed_report = self.report_generator.generate_detailed_report(
            metrics, predictions, ground_truths, model_name, dataset_name
        )
        
        # Generate multi-threshold evaluation
        multi_threshold_results = self.evaluator.evaluate_multi_threshold(predictions, ground_truths)
        detailed_report["multi_threshold_map"] = multi_threshold_results
        
        # Save results in multiple formats
        timestamp = self.report_generator.timestamp
        base_filename = f"{model_name}_{dataset_name}_{timestamp}"
        
        # Save JSON report
        json_path = self.output_dir / f"{base_filename}_report.json"
        self.report_generator.save_json_report(detailed_report, str(json_path))
        
        # Save CSV report
        csv_path = self.output_dir / f"{base_filename}_report.csv"
        self.report_generator.save_csv_report(detailed_report, str(csv_path))
        
        # Save summary table
        summary_path = self.output_dir / f"{base_filename}_summary.csv"
        self.report_generator.save_summary_table(detailed_report, str(summary_path))
        
        # Save predictions if requested
        predictions_path = None
        if save_predictions:
            predictions_path = self.output_dir / f"{base_filename}_predictions.json"
            self.report_generator.export_predictions(predictions, image_names, str(predictions_path))
        
        # Save basic metrics using evaluator
        metrics_path = self.output_dir / f"{base_filename}_metrics.json"
        self.evaluator.save_results(metrics, str(metrics_path), "json")
        
        return {
            "metrics": metrics,
            "detailed_report": detailed_report,
            "file_paths": {
                "json_report": str(json_path),
                "csv_report": str(csv_path),
                "summary_table": str(summary_path),
                "predictions": str(predictions_path) if predictions_path else None,
                "metrics": str(metrics_path)
            }
        }
    
    def compare_models(self,
                      baseline_predictions: List[List[Detection]],
                      current_predictions: List[List[Detection]],
                      ground_truths: List[List[Detection]],
                      baseline_name: str = "Baseline",
                      current_name: str = "Current",
                      dataset_name: str = "PCB_Dataset") -> Dict[str, any]:
        """
        Compare two models and generate comparison report.
        
        Args:
            baseline_predictions: Baseline model predictions
            current_predictions: Current model predictions
            ground_truths: Ground truth annotations
            baseline_name: Name of baseline model
            current_name: Name of current model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing comparison results and file paths
        """
        # Evaluate both models
        baseline_metrics = self.evaluator.generate_metrics_report(baseline_predictions, ground_truths)
        current_metrics = self.evaluator.generate_metrics_report(current_predictions, ground_truths)
        
        # Generate comparison report
        comparison_report = self.report_generator.generate_comparison_report(
            baseline_metrics, current_metrics, baseline_name, current_name
        )
        
        # Save comparison report
        timestamp = self.report_generator.timestamp
        comparison_filename = f"comparison_{baseline_name}_vs_{current_name}_{dataset_name}_{timestamp}"
        
        json_path = self.output_dir / f"{comparison_filename}.json"
        self.report_generator.save_json_report(comparison_report, str(json_path))
        
        csv_path = self.output_dir / f"{comparison_filename}.csv"
        self.report_generator.save_csv_report(comparison_report, str(csv_path))
        
        return {
            "baseline_metrics": baseline_metrics,
            "current_metrics": current_metrics,
            "comparison_report": comparison_report,
            "file_paths": {
                "json_comparison": str(json_path),
                "csv_comparison": str(csv_path)
            }
        }
    
    def load_and_evaluate(self, predictions_file: str, ground_truths_file: str) -> Dict[str, any]:
        """
        Load predictions and ground truths from files and evaluate.
        
        Args:
            predictions_file: Path to predictions JSON file
            ground_truths_file: Path to ground truths JSON file
            
        Returns:
            Evaluation results
        """
        predictions = self.report_generator.load_predictions(predictions_file)
        ground_truths = self.report_generator.load_predictions(ground_truths_file)
        
        return self.run_complete_evaluation(predictions, ground_truths)


def quick_evaluate(predictions: List[List[Detection]], 
                  ground_truths: List[List[Detection]],
                  iou_threshold: float = 0.5) -> EvaluationMetrics:
    """
    Quick evaluation function for simple use cases.
    
    Args:
        predictions: Predictions for each image
        ground_truths: Ground truths for each image
        iou_threshold: IoU threshold for evaluation
        
    Returns:
        EvaluationMetrics object
    """
    evaluator = Evaluator(iou_threshold=iou_threshold)
    return evaluator.generate_metrics_report(predictions, ground_truths)


def batch_evaluate_thresholds(predictions: List[List[Detection]], 
                             ground_truths: List[List[Detection]],
                             thresholds: List[float] = None) -> Dict[str, float]:
    """
    Evaluate at multiple IoU thresholds.
    
    Args:
        predictions: Predictions for each image
        ground_truths: Ground truths for each image
        thresholds: List of IoU thresholds to evaluate
        
    Returns:
        Dictionary mapping thresholds to mAP values
    """
    if thresholds is None:
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    results = {}
    for threshold in thresholds:
        evaluator = Evaluator(iou_threshold=threshold)
        map_value = evaluator.calculate_map(predictions, ground_truths)
        results[f"mAP@{threshold:.2f}"] = map_value
    
    return results