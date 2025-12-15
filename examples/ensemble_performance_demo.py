"""
Demonstration of model ensemble and performance monitoring functionality.
"""

import numpy as np
import cv2
from pathlib import Path
import logging
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pcb_detection.models.ensemble import ModelEnsemble, EnsembleMethod
from pcb_detection.models.yolo_detector import YOLODetector
from pcb_detection.utils.performance_monitor import PerformanceMonitor, measure_inference
from pcb_detection.utils.monitoring_integration import (
    MonitoredModel, MonitoredEvaluator, create_monitored_model
)
from pcb_detection.evaluation.evaluator import Evaluator
from pcb_detection.core.types import Detection, CLASS_MAPPING

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_model(model_name: str) -> YOLODetector:
    """Create a dummy YOLO model for demonstration."""
    model_config = {
        "variant": "yolov8n",
        "attention_type": None,
        "input_size": 640,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "use_focal": True,
        "use_iou": True,
        "iou_type": "ciou"
    }
    
    model = YOLODetector(model_config, num_classes=5)
    logger.info(f"Created dummy model: {model_name}")
    return model


def generate_dummy_detections(num_detections: int = 3) -> List[Detection]:
    """Generate dummy detections for demonstration."""
    detections = []
    
    for i in range(num_detections):
        # Random bbox coordinates (normalized)
        x = np.random.uniform(0.1, 0.7)
        y = np.random.uniform(0.1, 0.7)
        w = np.random.uniform(0.1, 0.3)
        h = np.random.uniform(0.1, 0.3)
        
        # Random class and confidence
        class_id = np.random.randint(0, 5)
        confidence = np.random.uniform(0.3, 0.9)
        
        detection = Detection(
            bbox=(x, y, w, h),
            confidence=confidence,
            class_id=class_id,
            class_name=CLASS_MAPPING[class_id]
        )
        detections.append(detection)
    
    return detections


def create_dummy_image(size: tuple = (640, 640)) -> np.ndarray:
    """Create a dummy image for demonstration."""
    # Create a random image with some patterns
    image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    
    # Add some geometric patterns to make it look more like a PCB
    cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), 2)
    cv2.circle(image, (300, 300), 50, (255, 0, 0), -1)
    cv2.line(image, (0, 400), (640, 400), (0, 0, 255), 3)
    
    return image


def demonstrate_ensemble():
    """Demonstrate model ensemble functionality."""
    logger.info("=== Model Ensemble Demonstration ===")
    
    # Create multiple dummy models
    models = []
    model_names = ["yolov8n_base", "yolov8n_augmented", "yolov8s_optimized"]
    
    for name in model_names:
        model = create_dummy_model(name)
        models.append(model)
    
    # Create ensemble with different methods
    ensemble_methods = ["average", "weighted", "nms"]
    
    for method in ensemble_methods:
        logger.info(f"\nTesting ensemble method: {method}")
        
        # Create ensemble
        ensemble = ModelEnsemble(
            models=models,
            ensemble_method=method,
            model_weights=[0.4, 0.35, 0.25] if method == "weighted" else None
        )
        
        # Create test image
        test_image = create_dummy_image()
        
        # Mock the predict method to return dummy detections
        for i, model in enumerate(models):
            original_predict = model.predict
            def mock_predict(image, model_idx=i):
                # Simulate different model predictions
                num_dets = np.random.randint(2, 6)
                return generate_dummy_detections(num_dets)
            model.predict = mock_predict
        
        try:
            # Get ensemble predictions
            ensemble_predictions = ensemble.predict(test_image)
            
            logger.info(f"Ensemble ({method}) produced {len(ensemble_predictions)} detections")
            
            for j, det in enumerate(ensemble_predictions[:3]):  # Show first 3
                logger.info(f"  Detection {j+1}: {det.class_name}, "
                           f"conf={det.confidence:.3f}, bbox={det.bbox}")
        
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring functionality."""
    logger.info("\n=== Performance Monitoring Demonstration ===")
    
    # Create performance monitor
    monitor = PerformanceMonitor(
        log_file="performance_demo.json",
        auto_save_interval=30.0
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Create a dummy model
        model = create_dummy_model("demo_model")
        
        # Mock the predict method for demonstration
        def mock_predict(image):
            # Simulate inference time
            import time
            time.sleep(np.random.uniform(0.01, 0.05))  # 10-50ms
            return generate_dummy_detections(np.random.randint(1, 5))
        
        model.predict = mock_predict
        
        # Test single inference with monitoring
        test_image = create_dummy_image()
        
        with measure_inference("demo_model", batch_size=1, image_size=(640, 640)):
            predictions = model.predict(test_image)
        
        logger.info(f"Single inference produced {len(predictions)} detections")
        
        # Test batch inference
        batch_images = [create_dummy_image() for _ in range(5)]
        
        with measure_inference("demo_model", batch_size=5, image_size=(640, 640)):
            batch_predictions = [model.predict(img) for img in batch_images]
        
        logger.info(f"Batch inference processed {len(batch_images)} images")
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        
        if 'inference_times' in summary:
            inf_stats = summary['inference_times']
            logger.info(f"Inference time stats: mean={inf_stats['mean']:.4f}s, "
                       f"std={inf_stats['std']:.4f}s")
        
        if 'throughputs' in summary:
            thr_stats = summary['throughputs']
            logger.info(f"Throughput stats: mean={thr_stats['mean']:.2f} img/s")
        
        # Generate performance report
        report_path = "performance_report.html"
        monitor.generate_report(report_path)
        logger.info(f"Generated performance report: {report_path}")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()


def demonstrate_monitored_evaluation():
    """Demonstrate monitored evaluation functionality."""
    logger.info("\n=== Monitored Evaluation Demonstration ===")
    
    # Create models
    models = []
    model_names = ["baseline_model", "optimized_model"]
    
    for name in model_names:
        model = create_dummy_model(name)
        
        # Mock predict method
        def mock_predict(image, model_name=name):
            # Simulate different performance characteristics
            if "optimized" in model_name:
                import time
                time.sleep(0.02)  # Faster model
                num_dets = np.random.randint(3, 7)  # More detections
            else:
                import time
                time.sleep(0.05)  # Slower model
                num_dets = np.random.randint(1, 4)  # Fewer detections
            
            return generate_dummy_detections(num_dets)
        
        model.predict = mock_predict
        models.append(model)
    
    # Create test data
    test_images = [create_dummy_image() for _ in range(10)]
    ground_truths = [generate_dummy_detections(np.random.randint(2, 5)) 
                    for _ in range(10)]
    
    # Create monitored evaluator
    evaluator = Evaluator()
    monitored_evaluator = MonitoredEvaluator(evaluator)
    
    try:
        # Compare models
        comparison_results = monitored_evaluator.compare_models(
            models, model_names, test_images, ground_truths
        )
        
        # Display results
        logger.info("Model comparison results:")
        
        summary = comparison_results.get('comparison_summary', {})
        if 'best_map' in summary and summary['best_map']['model']:
            logger.info(f"Best accuracy: {summary['best_map']['model']} "
                       f"(mAP: {summary['best_map']['value']:.4f})")
        
        if 'fastest_inference' in summary and summary['fastest_inference']['model']:
            logger.info(f"Fastest inference: {summary['fastest_inference']['model']} "
                       f"({summary['fastest_inference']['value']:.4f}s)")
        
        # Save comparison report
        report_path = "model_comparison_report"
        monitored_evaluator.save_comparison_report(comparison_results, report_path)
        logger.info(f"Saved comparison report: {report_path}.json and {report_path}.html")
        
    except Exception as e:
        logger.error(f"Monitored evaluation failed: {e}")


def main():
    """Run all demonstrations."""
    logger.info("Starting PCB Detection Ensemble and Performance Monitoring Demo")
    
    try:
        # Run demonstrations
        demonstrate_ensemble()
        demonstrate_performance_monitoring()
        demonstrate_monitored_evaluation()
        
        logger.info("\n=== Demo completed successfully! ===")
        logger.info("Check the generated files:")
        logger.info("- performance_demo.json: Performance metrics")
        logger.info("- performance_report.html: Performance report")
        logger.info("- model_comparison_report.json: Model comparison data")
        logger.info("- model_comparison_report.html: Model comparison report")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()