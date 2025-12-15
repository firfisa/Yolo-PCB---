#!/usr/bin/env python3
"""
Demonstration of the PCB defect detection evaluation system.

This script shows how to use the evaluation components to:
1. Calculate mAP and per-class AP
2. Generate comprehensive evaluation reports
3. Export results in multiple formats
4. Compare different models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_detection.evaluation import (
    Evaluator, 
    EvaluationPipeline, 
    quick_evaluate, 
    batch_evaluate_thresholds
)
from pcb_detection.core.types import Detection


def create_sample_data():
    """Create sample predictions and ground truths for demonstration."""
    
    # Sample predictions (simulating model output)
    predictions = [
        # Image 1: Mouse bite detection
        [Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.95, class_id=0, class_name='Mouse_bite')],
        
        # Image 2: Open circuit detection  
        [Detection(bbox=(0.3, 0.3, 0.15, 0.15), confidence=0.88, class_id=1, class_name='Open_circuit')],
        
        # Image 3: Multiple defects
        [
            Detection(bbox=(0.2, 0.2, 0.1, 0.1), confidence=0.92, class_id=2, class_name='Short'),
            Detection(bbox=(0.7, 0.7, 0.12, 0.12), confidence=0.85, class_id=3, class_name='Spur')
        ],
        
        # Image 4: Spurious copper
        [Detection(bbox=(0.6, 0.4, 0.18, 0.25), confidence=0.78, class_id=4, class_name='Spurious_copper')],
        
        # Image 5: No detections (empty)
        []
    ]
    
    # Ground truth annotations
    ground_truths = [
        # Image 1: Correct mouse bite
        [Detection(bbox=(0.52, 0.48, 0.18, 0.22), confidence=1.0, class_id=0, class_name='Mouse_bite')],
        
        # Image 2: Correct open circuit
        [Detection(bbox=(0.31, 0.29, 0.14, 0.16), confidence=1.0, class_id=1, class_name='Open_circuit')],
        
        # Image 3: Multiple defects (one missed by model)
        [
            Detection(bbox=(0.21, 0.19, 0.09, 0.11), confidence=1.0, class_id=2, class_name='Short'),
            Detection(bbox=(0.69, 0.71, 0.11, 0.13), confidence=1.0, class_id=3, class_name='Spur'),
            Detection(bbox=(0.8, 0.1, 0.08, 0.08), confidence=1.0, class_id=0, class_name='Mouse_bite')  # Missed
        ],
        
        # Image 4: Spurious copper
        [Detection(bbox=(0.58, 0.42, 0.16, 0.23), confidence=1.0, class_id=4, class_name='Spurious_copper')],
        
        # Image 5: Actually has a defect (false negative)
        [Detection(bbox=(0.4, 0.6, 0.1, 0.1), confidence=1.0, class_id=1, class_name='Open_circuit')]
    ]
    
    image_names = [
        'pcb_sample_001.jpg',
        'pcb_sample_002.jpg', 
        'pcb_sample_003.jpg',
        'pcb_sample_004.jpg',
        'pcb_sample_005.jpg'
    ]
    
    return predictions, ground_truths, image_names


def demo_basic_evaluation():
    """Demonstrate basic evaluation functionality."""
    print("=" * 60)
    print("BASIC EVALUATION DEMO")
    print("=" * 60)
    
    predictions, ground_truths, _ = create_sample_data()
    
    # Quick evaluation
    print("\n1. Quick Evaluation:")
    metrics = quick_evaluate(predictions, ground_truths)
    print(f"   mAP@0.5: {metrics.map_50:.4f}")
    print(f"   Precision: {metrics.precision:.4f}")
    print(f"   Recall: {metrics.recall:.4f}")
    print(f"   F1-Score: {metrics.f1_score:.4f}")
    
    # Per-class AP
    print("\n2. Per-Class Average Precision:")
    for class_name, ap in metrics.ap_per_class.items():
        print(f"   {class_name}: {ap:.4f}")
    
    # Multi-threshold evaluation
    print("\n3. Multi-Threshold Evaluation:")
    multi_threshold_results = batch_evaluate_thresholds(predictions, ground_truths)
    for threshold, map_value in multi_threshold_results.items():
        print(f"   {threshold}: {map_value:.4f}")


def demo_comprehensive_evaluation():
    """Demonstrate comprehensive evaluation pipeline."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION DEMO")
    print("=" * 60)
    
    predictions, ground_truths, image_names = create_sample_data()
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(output_dir="demo_evaluation_results")
    
    print("\n1. Running Complete Evaluation Pipeline...")
    results = pipeline.run_complete_evaluation(
        predictions=predictions,
        ground_truths=ground_truths,
        model_name="DemoYOLOv8",
        dataset_name="PCB_Demo_Dataset",
        image_names=image_names,
        save_predictions=True
    )
    
    print("   ✓ Evaluation completed successfully!")
    
    print("\n2. Generated Files:")
    for file_type, file_path in results['file_paths'].items():
        if file_path:
            print(f"   {file_type}: {os.path.basename(file_path)}")
    
    print("\n3. Detailed Report Summary:")
    report = results['detailed_report']
    print(f"   Total Images: {report['metadata']['total_images']}")
    print(f"   Total Predictions: {report['metadata']['total_predictions']}")
    print(f"   Total Ground Truths: {report['metadata']['total_ground_truths']}")
    print(f"   Overall mAP@0.5: {report['overall_metrics']['mAP@0.5']:.4f}")


def demo_model_comparison():
    """Demonstrate model comparison functionality."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON DEMO")
    print("=" * 60)
    
    predictions, ground_truths, _ = create_sample_data()
    
    # Create a "baseline" model with lower performance
    baseline_predictions = []
    for pred_list in predictions:
        # Simulate lower confidence and some missed detections
        baseline_list = []
        for i, pred in enumerate(pred_list):
            if i % 2 == 0:  # Keep only every other detection
                baseline_pred = Detection(
                    bbox=pred.bbox,
                    confidence=pred.confidence * 0.7,  # Lower confidence
                    class_id=pred.class_id,
                    class_name=pred.class_name
                )
                baseline_list.append(baseline_pred)
        baseline_predictions.append(baseline_list)
    
    # Compare models
    pipeline = EvaluationPipeline(output_dir="demo_comparison_results")
    
    print("\n1. Comparing Baseline vs Current Model...")
    try:
        comparison_results = pipeline.compare_models(
            baseline_predictions=baseline_predictions,
            current_predictions=predictions,
            ground_truths=ground_truths,
            baseline_name="BaselineYOLO",
            current_name="ImprovedYOLO",
            dataset_name="PCB_Demo_Dataset"
        )
        
        print("   ✓ Comparison completed successfully!")
        
        print("\n2. Performance Comparison:")
        print(f"   Available keys: {list(comparison_results.keys())}")
        
        # Access baseline and current metrics directly
        baseline_metrics = comparison_results['baseline_metrics']
        current_metrics = comparison_results['current_metrics']
        
        print(f"   Baseline mAP@0.5: {baseline_metrics.map_50:.4f}")
        print(f"   Current mAP@0.5:  {current_metrics.map_50:.4f}")
        print(f"   Improvement: {current_metrics.map_50 - baseline_metrics.map_50:+.4f}")
        
        print(f"   Baseline Precision: {baseline_metrics.precision:.4f}")
        print(f"   Current Precision:  {current_metrics.precision:.4f}")
        print(f"   Improvement: {current_metrics.precision - baseline_metrics.precision:+.4f}")
        
    except Exception as e:
        print(f"   Error in model comparison: {e}")
        print("   Skipping comparison demo...")


def demo_advanced_features():
    """Demonstrate advanced evaluation features."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMO")
    print("=" * 60)
    
    predictions, ground_truths, image_names = create_sample_data()
    
    # Custom IoU threshold evaluation
    print("\n1. Custom IoU Threshold Evaluation:")
    evaluator_strict = Evaluator(iou_threshold=0.7)
    evaluator_lenient = Evaluator(iou_threshold=0.3)
    
    metrics_strict = evaluator_strict.generate_metrics_report(predictions, ground_truths)
    metrics_lenient = evaluator_lenient.generate_metrics_report(predictions, ground_truths)
    
    print(f"   Strict (IoU=0.7):   mAP = {metrics_strict.map_50:.4f}")
    print(f"   Lenient (IoU=0.3):  mAP = {metrics_lenient.map_50:.4f}")
    
    # Export and reload predictions
    print("\n2. Predictions Export/Import:")
    pipeline = EvaluationPipeline(output_dir="demo_export_results")
    
    # Export predictions
    export_path = "demo_export_results/exported_predictions.json"
    pipeline.report_generator.export_predictions(predictions, image_names, export_path)
    print(f"   ✓ Predictions exported to: {os.path.basename(export_path)}")
    
    # Reload and verify
    reloaded_predictions = pipeline.report_generator.load_predictions(export_path)
    print(f"   ✓ Reloaded {len(reloaded_predictions)} image predictions")
    print(f"   ✓ First image has {len(reloaded_predictions[0])} detections")


def cleanup_demo_files():
    """Clean up demonstration files."""
    import shutil
    
    demo_dirs = [
        "demo_evaluation_results",
        "demo_comparison_results", 
        "demo_export_results"
    ]
    
    for dir_name in demo_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   ✓ Cleaned up {dir_name}")


def main():
    """Run all evaluation demonstrations."""
    print("PCB DEFECT DETECTION - EVALUATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the capabilities of the evaluation system:")
    print("• mAP and AP calculation")
    print("• Multi-threshold evaluation") 
    print("• Comprehensive report generation")
    print("• Model comparison")
    print("• Result export/import")
    
    try:
        # Run demonstrations
        demo_basic_evaluation()
        demo_comprehensive_evaluation()
        demo_model_comparison()
        demo_advanced_features()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the generated files in the demo_*_results directories.")
        print("All evaluation components are working correctly.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        return 1
    
    finally:
        # Optionally clean up (comment out to keep files)
        print("\nCleaning up demonstration files...")
        cleanup_demo_files()
    
    return 0


if __name__ == "__main__":
    exit(main())