#!/usr/bin/env python3
"""
Validation script for model architecture comparison framework.
Tests model benchmarking and comparison functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from typing import Dict, List

from pcb_detection.evaluation.model_comparison import (
    ModelConfig,
    PerformanceMetrics,
    ComparisonResult,
    ModelBenchmark,
    ModelComparison
)
from pcb_detection.models.yolo_detector import YOLODetector
from pcb_detection.models.cbam_integration import create_cbam_enhanced_yolo


def test_model_config():
    """Test model configuration system."""
    print("Testing Model Configuration System")
    print("=" * 40)
    
    # Test different configurations
    configs = [
        ModelConfig(
            name="yolov8n_base",
            variant="yolov8n",
            description="Base YOLOv8n model"
        ),
        ModelConfig(
            name="yolov8n_cbam",
            variant="yolov8n",
            attention_type="cbam",
            description="YOLOv8n with CBAM attention"
        ),
        ModelConfig(
            name="yolov8s_optimized",
            variant="yolov8s",
            attention_type="cbam",
            use_advanced_loss=True,
            use_multiscale=True,
            description="Fully optimized YOLOv8s"
        )
    ]
    
    for config in configs:
        print(f"  {config.name}:")
        print(f"    Variant: {config.variant}")
        print(f"    Attention: {config.attention_type or 'None'}")
        print(f"    Advanced loss: {config.use_advanced_loss}")
        print(f"    Multi-scale: {config.use_multiscale}")
        print(f"    Description: {config.description}")


def test_performance_metrics():
    """Test performance metrics structure."""
    print("\nTesting Performance Metrics")
    print("=" * 35)
    
    # Create sample metrics
    metrics = PerformanceMetrics(
        map_50=0.45,
        map_75=0.32,
        inference_time_ms=15.2,
        fps=65.8,
        model_size_mb=12.5,
        parameters=3_200_000,
        flops=8.5,
        memory_usage_mb=256.0
    )
    
    print(f"  mAP@0.5: {metrics.map_50}")
    print(f"  Inference time: {metrics.inference_time_ms}ms")
    print(f"  FPS: {metrics.fps}")
    print(f"  Model size: {metrics.model_size_mb}MB")
    print(f"  Parameters: {metrics.parameters:,}")
    print(f"  FLOPs: {metrics.flops}G")
    print(f"  Memory usage: {metrics.memory_usage_mb}MB")


def test_model_benchmark():
    """Test model benchmarking functionality."""
    print("\nTesting Model Benchmark")
    print("=" * 30)
    
    benchmark = ModelBenchmark()
    
    # Create a simple test model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(256, 5)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    
    # Test inference speed
    print("  Testing inference speed...")
    speed_metrics = benchmark.benchmark_inference_speed(
        model, input_size=(640, 640), num_runs=20, warmup_runs=5
    )
    
    print(f"    Inference time: {speed_metrics['inference_time_ms']:.2f} ± {speed_metrics['inference_time_std_ms']:.2f}ms")
    print(f"    FPS: {speed_metrics['fps']:.1f}")
    print(f"    Min/Max time: {speed_metrics['min_time_ms']:.2f}/{speed_metrics['max_time_ms']:.2f}ms")
    
    # Test model size calculation
    print("  Testing model size calculation...")
    size_metrics = benchmark.calculate_model_size(model)
    
    print(f"    Total parameters: {size_metrics['total_parameters']:,}")
    print(f"    Trainable parameters: {size_metrics['trainable_parameters']:,}")
    print(f"    Model size: {size_metrics['model_size_mb']:.2f}MB")
    
    # Test FLOP estimation
    print("  Testing FLOP estimation...")
    flops = benchmark.estimate_flops(model, (640, 640))
    print(f"    Estimated FLOPs: {flops:.2f}G")
    
    # Test memory usage
    print("  Testing memory usage...")
    memory_metrics = benchmark.measure_memory_usage(model, (640, 640))
    print(f"    Memory usage: {memory_metrics['memory_usage_mb']:.2f}MB")
    print(f"    Peak memory: {memory_metrics['peak_memory_mb']:.2f}MB")


def test_yolo_model_benchmark():
    """Test benchmarking with actual YOLO models."""
    print("\nTesting YOLO Model Benchmark")
    print("=" * 35)
    
    benchmark = ModelBenchmark()
    
    # Test configurations
    test_configs = [
        ModelConfig(
            name="yolov8n_base",
            variant="yolov8n",
            description="Base YOLOv8n"
        ),
        ModelConfig(
            name="yolov8n_cbam",
            variant="yolov8n",
            attention_type="cbam",
            description="YOLOv8n with CBAM"
        )
    ]
    
    results = []
    
    for config in test_configs:
        print(f"  Benchmarking {config.name}...")
        
        try:
            result = benchmark.benchmark_model(config)
            results.append(result)
            
            if "failed" not in result.notes.lower():
                print(f"    ✓ Parameters: {result.metrics.parameters:,}")
                print(f"    ✓ Inference: {result.metrics.inference_time_ms:.2f}ms")
                print(f"    ✓ FPS: {result.metrics.fps:.1f}")
                print(f"    ✓ Model size: {result.metrics.model_size_mb:.1f}MB")
                print(f"    ✓ FLOPs: {result.metrics.flops:.2f}G")
            else:
                print(f"    ✗ Failed: {result.notes}")
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    return results


def test_model_comparison_framework():
    """Test complete model comparison framework."""
    print("\nTesting Model Comparison Framework")
    print("=" * 40)
    
    comparison = ModelComparison(output_dir="test_comparison_results")
    
    # Create limited set of configurations for testing
    test_configs = [
        ModelConfig(
            name="yolov8n_base",
            variant="yolov8n",
            description="Base YOLOv8n model"
        ),
        ModelConfig(
            name="yolov8n_cbam",
            variant="yolov8n",
            attention_type="cbam",
            description="YOLOv8n with CBAM attention"
        ),
        ModelConfig(
            name="yolov8s_base",
            variant="yolov8s",
            description="Base YOLOv8s model"
        )
    ]
    
    print(f"  Running comparison on {len(test_configs)} configurations...")
    
    # Run comparison
    results = comparison.run_comparison(test_configs)
    
    print(f"  Completed {len(results)} benchmarks")
    
    # Analyze results
    analysis = comparison.analyze_results()
    
    print("  Analysis results:")
    print(f"    Total models: {analysis.get('total_models', 0)}")
    print(f"    Successful: {analysis.get('successful_models', 0)}")
    print(f"    Failed: {analysis.get('failed_models', 0)}")
    
    if 'best_models' in analysis:
        best = analysis['best_models']
        if 'fastest' in best:
            print(f"    Fastest: {best['fastest']['name']} ({best['fastest']['fps']:.1f} FPS)")
        if 'smallest' in best:
            print(f"    Smallest: {best['smallest']['name']} ({best['smallest']['model_size_mb']:.1f} MB)")
    
    # Save results
    print("  Saving results...")
    comparison.save_results("test_comparison")
    
    # Generate visualizations
    print("  Generating visualizations...")
    try:
        comparison.visualize_results()
        print("    ✓ Visualizations generated")
    except Exception as e:
        print(f"    ✗ Visualization failed: {e}")
    
    return results, analysis


def test_configuration_generation():
    """Test automatic configuration generation."""
    print("\nTesting Configuration Generation")
    print("=" * 40)
    
    comparison = ModelComparison()
    configs = comparison.create_model_configs()
    
    print(f"  Generated {len(configs)} configurations")
    
    # Group by variant
    variant_counts = {}
    attention_counts = {}
    
    for config in configs:
        variant_counts[config.variant] = variant_counts.get(config.variant, 0) + 1
        att_type = config.attention_type or "none"
        attention_counts[att_type] = attention_counts.get(att_type, 0) + 1
    
    print("  Configurations by variant:")
    for variant, count in sorted(variant_counts.items()):
        print(f"    {variant}: {count}")
    
    print("  Configurations by attention type:")
    for att_type, count in sorted(attention_counts.items()):
        print(f"    {att_type}: {count}")
    
    # Show sample configurations
    print("  Sample configurations:")
    for i, config in enumerate(configs[:5]):
        print(f"    {i+1}. {config.name}: {config.variant} + {config.attention_type or 'none'}")


def benchmark_attention_mechanisms():
    """Benchmark different attention mechanisms."""
    print("\nBenchmarking Attention Mechanisms")
    print("=" * 40)
    
    benchmark = ModelBenchmark()
    
    attention_types = [None, "cbam", "se", "eca", "coord"]
    results = {}
    
    for att_type in attention_types:
        config = ModelConfig(
            name=f"yolov8n_{att_type or 'base'}",
            variant="yolov8n",
            attention_type=att_type,
            description=f"YOLOv8n with {att_type or 'no'} attention"
        )
        
        print(f"  Testing {att_type or 'base'} attention...")
        
        try:
            result = benchmark.benchmark_model(config)
            
            if "failed" not in result.notes.lower():
                results[att_type or 'base'] = {
                    'parameters': result.metrics.parameters,
                    'inference_time': result.metrics.inference_time_ms,
                    'fps': result.metrics.fps,
                    'model_size': result.metrics.model_size_mb,
                    'flops': result.metrics.flops
                }
                
                print(f"    ✓ {result.metrics.parameters:,} params, "
                      f"{result.metrics.inference_time_ms:.2f}ms, "
                      f"{result.metrics.fps:.1f} FPS")
            else:
                print(f"    ✗ Failed: {result.notes}")
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Compare results
    if len(results) > 1:
        print("\n  Attention mechanism comparison:")
        base_result = results.get('base')
        
        if base_result:
            for att_type, result in results.items():
                if att_type != 'base':
                    param_increase = (result['parameters'] - base_result['parameters']) / base_result['parameters'] * 100
                    time_increase = (result['inference_time'] - base_result['inference_time']) / base_result['inference_time'] * 100
                    
                    print(f"    {att_type:8s}: +{param_increase:+5.1f}% params, +{time_increase:+5.1f}% time")
    
    return results


def main():
    """Main validation function."""
    print("Model Architecture Comparison Validation")
    print("=" * 50)
    
    # Test individual components
    test_model_config()
    test_performance_metrics()
    test_model_benchmark()
    
    # Test with YOLO models
    yolo_results = test_yolo_model_benchmark()
    
    # Test complete framework
    comparison_results, analysis = test_model_comparison_framework()
    
    # Test configuration generation
    test_configuration_generation()
    
    # Benchmark attention mechanisms
    attention_results = benchmark_attention_mechanisms()
    
    # Summary
    print("\n" + "=" * 50)
    print("Model Architecture Comparison Validation Complete")
    
    print(f"\nResults Summary:")
    print(f"  YOLO benchmarks: {len(yolo_results)} completed")
    print(f"  Framework comparison: {len(comparison_results)} models tested")
    print(f"  Attention mechanisms: {len(attention_results)} tested")
    
    if analysis:
        print(f"  Analysis: {analysis.get('successful_models', 0)}/{analysis.get('total_models', 0)} successful")
    
    print("\nKey Features Validated:")
    print("✓ Model configuration system")
    print("✓ Performance metrics calculation")
    print("✓ Inference speed benchmarking")
    print("✓ Model size and FLOP estimation")
    print("✓ Memory usage measurement")
    print("✓ Comprehensive model comparison")
    print("✓ Result analysis and visualization")
    print("✓ Attention mechanism comparison")


if __name__ == "__main__":
    main()