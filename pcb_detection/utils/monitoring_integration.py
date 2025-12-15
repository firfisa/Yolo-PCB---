"""
Integration utilities for performance monitoring with PCB detection system.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import logging

from ..core.interfaces import ModelInterface
from ..core.types import Detection
from .performance_monitor import PerformanceMonitor, measure_inference
from ..evaluation.evaluator import Evaluator


class MonitoredModel:
    """Wrapper for models with integrated performance monitoring."""
    
    def __init__(self, model: ModelInterface, model_name: str, 
                 monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize monitored model wrapper.
        
        Args:
            model: The model to wrap
            model_name: Name for performance tracking
            monitor: Optional performance monitor (uses global if None)
        """
        self.model = model
        self.model_name = model_name
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
    def predict(self, image: np.ndarray) -> List[Detection]:
        """Predict with performance monitoring."""
        image_size = image.shape[:2] if image is not None else None
        
        if self.monitor:
            with self.monitor.measure_inference(
                model_name=self.model_name,
                batch_size=1,
                image_size=image_size
            ):
                return self.model.predict(image)
        else:
            with measure_inference(
                model_name=self.model_name,
                batch_size=1,
                image_size=image_size
            ):
                return self.model.predict(image)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """Predict batch with performance monitoring."""
        batch_size = len(images)
        image_size = images[0].shape[:2] if images and images[0] is not None else None
        
        if self.monitor:
            with self.monitor.measure_inference(
                model_name=self.model_name,
                batch_size=batch_size,
                image_size=image_size
            ):
                return [self.model.predict(img) for img in images]
        else:
            with measure_inference(
                model_name=self.model_name,
                batch_size=batch_size,
                image_size=image_size
            ):
                return [self.model.predict(img) for img in images]
    
    def __getattr__(self, name):
        """Delegate other attributes to the wrapped model."""
        return getattr(self.model, name)


class MonitoredEvaluator:
    """Evaluator with integrated performance monitoring."""
    
    def __init__(self, evaluator: Evaluator, 
                 monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize monitored evaluator.
        
        Args:
            evaluator: The evaluator to wrap
            monitor: Optional performance monitor
        """
        self.evaluator = evaluator
        self.monitor = monitor or PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model_performance(self, model: ModelInterface, 
                                  test_images: List[np.ndarray],
                                  ground_truths: List[List[Detection]],
                                  model_name: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate model with comprehensive performance monitoring.
        
        Args:
            model: Model to evaluate
            test_images: Test images
            ground_truths: Ground truth detections
            model_name: Name for tracking
            
        Returns:
            Combined evaluation and performance metrics
        """
        self.logger.info(f"Starting evaluation of {model_name}")
        
        # Wrap model with monitoring
        monitored_model = MonitoredModel(model, model_name, self.monitor)
        
        # Start monitoring
        if not self.monitor.is_monitoring:
            self.monitor.start_monitoring()
        
        try:
            # Get predictions with performance monitoring
            predictions = monitored_model.predict_batch(test_images)
            
            # Calculate evaluation metrics
            eval_metrics = self.evaluator.generate_metrics_report(
                predictions, ground_truths
            )
            
            # Get performance summary
            perf_summary = self.monitor.get_performance_summary()
            
            # Combine results
            combined_results = {
                'model_name': model_name,
                'evaluation_metrics': {
                    'mAP@0.5': eval_metrics.map_50,
                    'precision': eval_metrics.precision,
                    'recall': eval_metrics.recall,
                    'f1_score': eval_metrics.f1_score,
                    'ap_per_class': eval_metrics.ap_per_class
                },
                'performance_metrics': perf_summary,
                'test_set_size': len(test_images),
                'total_detections': sum(len(preds) for preds in predictions)
            }
            
            self.logger.info(f"Evaluation completed for {model_name}")
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for {model_name}: {e}")
            raise
    
    def compare_models(self, models: List[ModelInterface],
                      model_names: List[str],
                      test_images: List[np.ndarray],
                      ground_truths: List[List[Detection]]) -> Dict[str, Any]:
        """
        Compare multiple models with performance monitoring.
        
        Args:
            models: List of models to compare
            model_names: Names for each model
            test_images: Test images
            ground_truths: Ground truth detections
            
        Returns:
            Comparison results with performance metrics
        """
        if len(models) != len(model_names):
            raise ValueError("Number of models must match number of names")
        
        self.logger.info(f"Comparing {len(models)} models")
        
        comparison_results = {
            'models': {},
            'comparison_summary': {},
            'test_set_info': {
                'num_images': len(test_images),
                'total_ground_truths': sum(len(gt) for gt in ground_truths)
            }
        }
        
        # Evaluate each model
        for model, name in zip(models, model_names):
            try:
                results = self.evaluate_model_performance(
                    model, test_images, ground_truths, name
                )
                comparison_results['models'][name] = results
            except Exception as e:
                self.logger.error(f"Failed to evaluate model {name}: {e}")
                comparison_results['models'][name] = {'error': str(e)}
        
        # Generate comparison summary
        comparison_results['comparison_summary'] = self._generate_comparison_summary(
            comparison_results['models']
        )
        
        return comparison_results
    
    def _generate_comparison_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary from model results."""
        summary = {
            'best_map': {'model': None, 'value': 0.0},
            'fastest_inference': {'model': None, 'value': float('inf')},
            'lowest_memory': {'model': None, 'value': float('inf')},
            'highest_throughput': {'model': None, 'value': 0.0},
            'rankings': {}
        }
        
        # Extract metrics for comparison
        model_metrics = {}
        for model_name, results in model_results.items():
            if 'error' in results:
                continue
                
            eval_metrics = results.get('evaluation_metrics', {})
            perf_metrics = results.get('performance_metrics', {})
            
            model_metrics[model_name] = {
                'map': eval_metrics.get('mAP@0.5', 0.0),
                'inference_time': perf_metrics.get('inference_times', {}).get('mean', float('inf')),
                'memory_usage': perf_metrics.get('memory_usage', {}).get('mean', float('inf')),
                'throughput': perf_metrics.get('throughputs', {}).get('mean', 0.0)
            }
        
        # Find best performers
        for model_name, metrics in model_metrics.items():
            # Best mAP
            if metrics['map'] > summary['best_map']['value']:
                summary['best_map'] = {'model': model_name, 'value': metrics['map']}
            
            # Fastest inference
            if metrics['inference_time'] < summary['fastest_inference']['value']:
                summary['fastest_inference'] = {'model': model_name, 'value': metrics['inference_time']}
            
            # Lowest memory usage
            if metrics['memory_usage'] < summary['lowest_memory']['value']:
                summary['lowest_memory'] = {'model': model_name, 'value': metrics['memory_usage']}
            
            # Highest throughput
            if metrics['throughput'] > summary['highest_throughput']['value']:
                summary['highest_throughput'] = {'model': model_name, 'value': metrics['throughput']}
        
        # Generate rankings
        if model_metrics:
            # Rank by mAP (descending)
            map_ranking = sorted(model_metrics.items(), 
                               key=lambda x: x[1]['map'], reverse=True)
            summary['rankings']['by_accuracy'] = [name for name, _ in map_ranking]
            
            # Rank by inference time (ascending)
            speed_ranking = sorted(model_metrics.items(), 
                                 key=lambda x: x[1]['inference_time'])
            summary['rankings']['by_speed'] = [name for name, _ in speed_ranking]
            
            # Rank by memory usage (ascending)
            memory_ranking = sorted(model_metrics.items(), 
                                  key=lambda x: x[1]['memory_usage'])
            summary['rankings']['by_memory'] = [name for name, _ in memory_ranking]
        
        return summary
    
    def save_comparison_report(self, comparison_results: Dict[str, Any], 
                              output_path: str):
        """Save detailed comparison report."""
        import json
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Generate HTML report
        html_path = Path(output_path).with_suffix('.html')
        html_content = self._generate_comparison_html(comparison_results)
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Saved comparison report to {json_path} and {html_path}")
    
    def _generate_comparison_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML comparison report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PCB Detection Model Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; border: 1px solid #ccc; padding: 15px; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .best { background-color: #d4edda; }
                .metric { margin: 5px 0; }
            </style>
        </head>
        <body>
            <h1>PCB Detection Model Comparison Report</h1>
        """
        
        # Summary section
        summary = results.get('comparison_summary', {})
        if summary:
            html += '<div class="section"><h2>Performance Summary</h2>'
            
            if 'best_map' in summary and summary['best_map']['model']:
                html += f'<div class="metric"><strong>Best Accuracy:</strong> {summary["best_map"]["model"]} (mAP: {summary["best_map"]["value"]:.4f})</div>'
            
            if 'fastest_inference' in summary and summary['fastest_inference']['model']:
                html += f'<div class="metric"><strong>Fastest Inference:</strong> {summary["fastest_inference"]["model"]} ({summary["fastest_inference"]["value"]:.4f}s)</div>'
            
            if 'lowest_memory' in summary and summary['lowest_memory']['model']:
                html += f'<div class="metric"><strong>Lowest Memory:</strong> {summary["lowest_memory"]["model"]} ({summary["lowest_memory"]["value"]:.1f}MB)</div>'
            
            if 'highest_throughput' in summary and summary['highest_throughput']['model']:
                html += f'<div class="metric"><strong>Highest Throughput:</strong> {summary["highest_throughput"]["model"]} ({summary["highest_throughput"]["value"]:.2f} img/s)</div>'
            
            html += '</div>'
        
        # Detailed comparison table
        models = results.get('models', {})
        if models:
            html += '<div class="section"><h2>Detailed Comparison</h2><table>'
            html += '<tr><th>Model</th><th>mAP@0.5</th><th>Precision</th><th>Recall</th><th>F1</th><th>Avg Inference (s)</th><th>Throughput (img/s)</th><th>Memory (MB)</th></tr>'
            
            for model_name, model_data in models.items():
                if 'error' in model_data:
                    html += f'<tr><td>{model_name}</td><td colspan="7">Error: {model_data["error"]}</td></tr>'
                    continue
                
                eval_metrics = model_data.get('evaluation_metrics', {})
                perf_metrics = model_data.get('performance_metrics', {})
                
                map_val = eval_metrics.get('mAP@0.5', 0.0)
                precision = eval_metrics.get('precision', 0.0)
                recall = eval_metrics.get('recall', 0.0)
                f1 = eval_metrics.get('f1_score', 0.0)
                
                inf_time = perf_metrics.get('inference_times', {}).get('mean', 0.0)
                throughput = perf_metrics.get('throughputs', {}).get('mean', 0.0)
                memory = perf_metrics.get('memory_usage', {}).get('mean', 0.0)
                
                # Highlight best values
                best_map = summary.get('best_map', {}).get('model') == model_name
                fastest = summary.get('fastest_inference', {}).get('model') == model_name
                lowest_mem = summary.get('lowest_memory', {}).get('model') == model_name
                highest_thr = summary.get('highest_throughput', {}).get('model') == model_name
                
                html += f"""
                <tr>
                    <td>{model_name}</td>
                    <td{'class="best"' if best_map else ''}>{map_val:.4f}</td>
                    <td>{precision:.4f}</td>
                    <td>{recall:.4f}</td>
                    <td>{f1:.4f}</td>
                    <td{'class="best"' if fastest else ''}>{inf_time:.4f}</td>
                    <td{'class="best"' if highest_thr else ''}>{throughput:.2f}</td>
                    <td{'class="best"' if lowest_mem else ''}>{memory:.1f}</td>
                </tr>
                """
            
            html += '</table></div>'
        
        html += '</body></html>'
        return html


def create_monitored_model(model: ModelInterface, model_name: str, 
                          monitor: Optional[PerformanceMonitor] = None) -> MonitoredModel:
    """
    Create a monitored model wrapper.
    
    Args:
        model: Model to wrap
        model_name: Name for performance tracking
        monitor: Optional performance monitor
        
    Returns:
        MonitoredModel instance
    """
    return MonitoredModel(model, model_name, monitor)


def create_monitored_evaluator(evaluator: Optional[Evaluator] = None,
                              monitor: Optional[PerformanceMonitor] = None) -> MonitoredEvaluator:
    """
    Create a monitored evaluator.
    
    Args:
        evaluator: Optional evaluator (creates new if None)
        monitor: Optional performance monitor
        
    Returns:
        MonitoredEvaluator instance
    """
    if evaluator is None:
        evaluator = Evaluator()
    
    return MonitoredEvaluator(evaluator, monitor)