# Model Ensemble and Performance Monitoring Guide

This guide covers the newly implemented model ensemble and performance monitoring functionality for the PCB defect detection system.

## Model Ensemble

### Overview
The model ensemble functionality allows combining predictions from multiple YOLO models to improve detection accuracy and robustness.

### Supported Ensemble Methods

1. **Average Ensemble**: Averages predictions from multiple models with optional weighting
2. **Weighted Ensemble**: Uses dynamic weights based on model confidence
3. **NMS Ensemble**: Combines all predictions and applies Non-Maximum Suppression
4. **Weighted Boxes Fusion (WBF)**: Advanced fusion method for better box alignment

### Basic Usage

```python
from pcb_detection.models.ensemble import ModelEnsemble
from pcb_detection.models.yolo_detector import YOLODetector

# Create multiple models
models = [
    YOLODetector(config1),
    YOLODetector(config2),
    YOLODetector(config3)
]

# Create ensemble
ensemble = ModelEnsemble(
    models=models,
    ensemble_method="weighted",
    model_weights=[0.4, 0.35, 0.25]  # Optional weights
)

# Make predictions
predictions = ensemble.predict(image)
```

### Weight Optimization

The ensemble can automatically optimize weights based on validation data:

```python
# Optimize weights using validation data
optimized_weights = ensemble.optimize_weights(
    validation_images=val_images,
    validation_targets=val_targets,
    method="grid_search"  # or "random_search"
)

# Update ensemble with optimized weights
ensemble.model_weights = optimized_weights
```

## Performance Monitoring

### Overview
The performance monitoring system tracks inference speed, resource usage, and system metrics during model execution.

### Key Features

- **Real-time Resource Tracking**: CPU, memory, and GPU usage
- **Inference Timing**: Precise measurement of inference times
- **Throughput Calculation**: Images processed per second
- **Automatic Reporting**: HTML and JSON report generation
- **Background Monitoring**: Non-intrusive resource tracking

### Basic Usage

```python
from pcb_detection.utils.performance_monitor import PerformanceMonitor, measure_inference

# Create monitor
monitor = PerformanceMonitor(log_file="performance.json")
monitor.start_monitoring()

# Measure single inference
with measure_inference("yolov8n", batch_size=1):
    predictions = model.predict(image)

# Get performance summary
summary = monitor.get_performance_summary()
print(f"Average inference time: {summary['inference_times']['mean']:.4f}s")

# Generate report
monitor.generate_report("performance_report.html")
monitor.stop_monitoring()
```

### Monitored Model Wrapper

For automatic monitoring, use the MonitoredModel wrapper:

```python
from pcb_detection.utils.monitoring_integration import create_monitored_model

# Wrap model with monitoring
monitored_model = create_monitored_model(model, "yolov8n")

# All predictions are automatically monitored
predictions = monitored_model.predict(image)
```

## Integrated Evaluation

### Model Comparison with Performance Metrics

The MonitoredEvaluator combines accuracy evaluation with performance monitoring:

```python
from pcb_detection.utils.monitoring_integration import MonitoredEvaluator
from pcb_detection.evaluation.evaluator import Evaluator

# Create monitored evaluator
evaluator = MonitoredEvaluator(Evaluator())

# Compare multiple models
comparison_results = evaluator.compare_models(
    models=[model1, model2, model3],
    model_names=["baseline", "optimized", "ensemble"],
    test_images=test_images,
    ground_truths=ground_truths
)

# Save comprehensive report
evaluator.save_comparison_report(comparison_results, "comparison_report")
```

### Performance Metrics Tracked

- **Inference Time**: Time per image/batch
- **Throughput**: Images processed per second
- **CPU Usage**: Percentage during inference
- **Memory Usage**: RAM consumption in MB
- **GPU Usage**: GPU utilization percentage (if available)
- **GPU Memory**: VRAM usage in MB (if available)

## Configuration Examples

### Ensemble Configuration

```python
# High accuracy ensemble (slower)
ensemble_config = {
    "ensemble_method": "wbf",  # Weighted Boxes Fusion
    "model_weights": None,     # Auto-optimize
    "iou_threshold": 0.55
}

# Fast ensemble (faster)
ensemble_config = {
    "ensemble_method": "average",
    "model_weights": [0.33, 0.33, 0.34],  # Equal weights
    "iou_threshold": 0.5
}
```

### Performance Monitor Configuration

```python
# Detailed monitoring
monitor = PerformanceMonitor(
    log_file="detailed_performance.json",
    auto_save_interval=30.0  # Save every 30 seconds
)

# Lightweight monitoring
monitor = PerformanceMonitor(
    log_file=None,  # No auto-save
    auto_save_interval=0.0  # Disabled
)
```

## Best Practices

### Ensemble Usage

1. **Model Diversity**: Use models with different architectures or training strategies
2. **Weight Optimization**: Always optimize weights on validation data
3. **Method Selection**: 
   - Use "average" for speed
   - Use "wbf" for best accuracy
   - Use "nms" for balanced performance

### Performance Monitoring

1. **Background Monitoring**: Start monitoring before inference loops
2. **Batch Processing**: Monitor batch inference for better throughput metrics
3. **Resource Tracking**: Enable GPU monitoring when available
4. **Report Generation**: Generate reports after significant evaluation runs

### Memory Management

1. **Monitor Memory Usage**: Track memory consumption during long runs
2. **Batch Size Optimization**: Use performance metrics to optimize batch sizes
3. **Model Loading**: Monitor memory usage when loading multiple models

## Troubleshooting

### Common Issues

1. **GPU Monitoring Not Available**: Install `GPUtil` package
2. **High Memory Usage**: Reduce batch size or number of ensemble models
3. **Slow Ensemble**: Consider using "average" method instead of "wbf"
4. **Inaccurate Timing**: Ensure proper warm-up before timing measurements

### Performance Optimization

1. **Model Selection**: Use performance metrics to select optimal models
2. **Ensemble Size**: Balance accuracy vs. speed with ensemble size
3. **Resource Allocation**: Monitor resource usage to optimize system configuration

## Example Scripts

See `examples/ensemble_performance_demo.py` for a complete demonstration of all functionality.

## Requirements Validation

This implementation satisfies the following requirements:

- **Requirement 2.5**: Model ensemble functionality for improved accuracy
- **Requirement 5.5**: Performance monitoring for inference speed and resource usage
- **Property 6**: Ensemble prediction consistency validation
- **Property 19**: Performance metrics tracking validation

The implementation provides comprehensive tools for both improving model accuracy through ensembling and monitoring system performance for optimization.