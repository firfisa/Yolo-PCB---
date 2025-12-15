"""
Model architecture comparison framework for PCB defect detection.
Provides comprehensive evaluation and benchmarking of different model configurations.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import csv
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.yolo_detector import YOLODetector
from ..models.cbam_integration import create_cbam_enhanced_yolo
from ..core.types import Detection


@dataclass
class ModelConfig:
    """Configuration for model comparison."""
    name: str
    variant: str  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    attention_type: Optional[str] = None  # cbam, se, eca, coord
    use_advanced_loss: bool = False
    use_multiscale: bool = False
    input_size: int = 640
    description: str = ""


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    # Accuracy metrics
    map_50: float = 0.0
    map_75: float = 0.0
    map_50_95: float = 0.0
    ap_per_class: Dict[str, float] = None
    
    # Speed metrics
    inference_time_ms: float = 0.0
    fps: float = 0.0
    
    # Resource metrics
    model_size_mb: float = 0.0
    parameters: int = 0
    flops: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Training metrics
    training_time_hours: float = 0.0
    convergence_epoch: int = 0
    
    def __post_init__(self):
        if self.ap_per_class is None:
            self.ap_per_class = {}


@dataclass
class ComparisonResult:
    """Result of model comparison."""
    config: ModelConfig
    metrics: PerformanceMetrics
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        config_dict = asdict(self.config)
        metrics_dict = asdict(self.metrics)
        
        return {
            'config': convert_numpy_types(config_dict),
            'metrics': convert_numpy_types(metrics_dict),
            'notes': self.notes
        }


class ModelBenchmark:
    """Benchmark suite for model performance evaluation."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize model benchmark.
        
        Args:
            device: Device to run benchmarks on
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.results = []
        
    def benchmark_inference_speed(self, model, 
                                input_size: Tuple[int, int] = (640, 640),
                                batch_size: int = 1,
                                num_runs: int = 100,
                                warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            model: Model to benchmark (YOLODetector or nn.Module)
            input_size: Input image size
            batch_size: Batch size for inference
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Speed metrics dictionary
        """
        # Handle YOLODetector vs nn.Module
        if hasattr(model, 'set_train_mode'):
            model.set_train_mode(False)  # YOLODetector
        elif hasattr(model, 'eval'):
            model.eval()  # nn.Module
        
        # Move to device if it's an nn.Module
        if hasattr(model, 'to'):
            model = model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, *input_size, device=self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                if hasattr(model, 'forward'):
                    _ = model.forward(dummy_input)
                else:
                    _ = model(dummy_input)
        
        # Synchronize GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                
                if hasattr(model, 'forward'):
                    _ = model.forward(dummy_input)
                else:
                    _ = model(dummy_input)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / mean_time * batch_size
        
        return {
            'inference_time_ms': mean_time,
            'inference_time_std_ms': std_time,
            'fps': fps,
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times)
        }
    
    def calculate_model_size(self, model) -> Dict[str, float]:
        """
        Calculate model size and parameter count.
        
        Args:
            model: Model to analyze (YOLODetector or nn.Module)
            
        Returns:
            Size metrics dictionary
        """
        # Handle YOLODetector vs nn.Module
        if hasattr(model, 'parameters') and callable(model.parameters):
            # YOLODetector with parameters() method
            params = list(model.parameters())
            total_params = sum(p.numel() for p in params)
            trainable_params = sum(p.numel() for p in params if p.requires_grad)
        elif hasattr(model, 'backbone') and hasattr(model, 'neck') and hasattr(model, 'head'):
            # YOLODetector with separate components
            all_params = []
            all_params.extend(model.backbone.parameters())
            all_params.extend(model.neck.parameters())
            all_params.extend(model.head.parameters())
            total_params = sum(p.numel() for p in all_params)
            trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
        else:
            # Standard nn.Module
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size (approximate)
        try:
            if hasattr(model, 'parameters') and callable(model.parameters):
                params = list(model.parameters())
                param_size = sum(p.numel() * p.element_size() for p in params)
            elif hasattr(model, 'backbone'):
                all_params = []
                all_params.extend(model.backbone.parameters())
                all_params.extend(model.neck.parameters())
                all_params.extend(model.head.parameters())
                param_size = sum(p.numel() * p.element_size() for p in all_params)
            else:
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # Try to get buffers
            try:
                if hasattr(model, 'buffers'):
                    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                else:
                    buffer_size = 0
            except:
                buffer_size = 0
                
            model_size_bytes = param_size + buffer_size
            model_size_mb = model_size_bytes / (1024 * 1024)
        except:
            # Fallback calculation
            model_size_mb = total_params * 4 / (1024 * 1024)  # Assume 4 bytes per param
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'param_size_mb': param_size / (1024 * 1024),
            'buffer_size_mb': buffer_size / (1024 * 1024)
        }
    
    def estimate_flops(self, model: nn.Module, 
                      input_size: Tuple[int, int] = (640, 640)) -> float:
        """
        Estimate FLOPs for model (simplified calculation).
        
        Args:
            model: Model to analyze
            input_size: Input image size
            
        Returns:
            Estimated FLOPs in GFLOPs
        """
        # This is a simplified FLOP estimation
        # For more accurate results, use tools like thop or fvcore
        
        total_flops = 0
        input_shape = (1, 3, *input_size)
        
        # Rough estimation based on model parameters and input size
        total_params = sum(p.numel() for p in model.parameters())
        
        # Approximate FLOPs as 2 * params * input_pixels (very rough)
        input_pixels = input_shape[1] * input_shape[2] * input_shape[3]
        estimated_flops = 2 * total_params * input_pixels / input_shape[0]
        
        # Convert to GFLOPs
        gflops = estimated_flops / 1e9
        
        return gflops
    
    def measure_memory_usage(self, model,
                           input_size: Tuple[int, int] = (640, 640),
                           batch_size: int = 1) -> Dict[str, float]:
        """
        Measure GPU memory usage during inference.
        
        Args:
            model: Model to test (YOLODetector or nn.Module)
            input_size: Input image size
            batch_size: Batch size
            
        Returns:
            Memory usage metrics
        """
        if self.device.type != "cuda":
            return {'memory_usage_mb': 0.0, 'peak_memory_mb': 0.0}
        
        # Move to device if possible
        if hasattr(model, 'to'):
            model = model.to(self.device)
        
        # Set eval mode
        if hasattr(model, 'set_train_mode'):
            model.set_train_mode(False)
        elif hasattr(model, 'eval'):
            model.eval()
        
        # Clear cache and measure baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Run inference
        dummy_input = torch.randn(batch_size, 3, *input_size, device=self.device)
        
        with torch.no_grad():
            if hasattr(model, 'forward'):
                _ = model.forward(dummy_input)
            else:
                _ = model(dummy_input)
        
        # Measure memory
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return {
            'memory_usage_mb': current_memory - baseline_memory,
            'peak_memory_mb': peak_memory - baseline_memory,
            'baseline_memory_mb': baseline_memory
        }
    
    def benchmark_model(self, config: ModelConfig) -> ComparisonResult:
        """
        Run comprehensive benchmark on a model configuration.
        
        Args:
            config: Model configuration to benchmark
            
        Returns:
            Comparison result with all metrics
        """
        print(f"Benchmarking {config.name}...")
        
        try:
            # Create model
            if config.attention_type:
                model_config = create_cbam_enhanced_yolo(
                    config.variant, 
                    optimize_for_small_objects=True
                )
                model_config['attention_type'] = config.attention_type
            else:
                model_config = {
                    'variant': config.variant,
                    'input_size': config.input_size,
                    'conf_threshold': 0.25,
                    'iou_threshold': 0.45
                }
            
            model = YOLODetector(model_config)
            
            # Initialize metrics
            metrics = PerformanceMetrics()
            
            # Benchmark inference speed
            speed_metrics = self.benchmark_inference_speed(
                model, (config.input_size, config.input_size)
            )
            metrics.inference_time_ms = speed_metrics['inference_time_ms']
            metrics.fps = speed_metrics['fps']
            
            # Calculate model size
            size_metrics = self.calculate_model_size(model)
            metrics.model_size_mb = size_metrics['model_size_mb']
            metrics.parameters = size_metrics['total_parameters']
            
            # Estimate FLOPs
            metrics.flops = self.estimate_flops(model, (config.input_size, config.input_size))
            
            # Measure memory usage
            memory_metrics = self.measure_memory_usage(
                model, (config.input_size, config.input_size)
            )
            metrics.memory_usage_mb = memory_metrics['memory_usage_mb']
            
            result = ComparisonResult(
                config=config,
                metrics=metrics,
                notes="Benchmark completed successfully"
            )
            
        except Exception as e:
            # Create result with error
            metrics = PerformanceMetrics()
            result = ComparisonResult(
                config=config,
                metrics=metrics,
                notes=f"Benchmark failed: {str(e)}"
            )
        
        self.results.append(result)
        return result


class ModelComparison:
    """Comprehensive model comparison framework."""
    
    def __init__(self, output_dir: str = "model_comparison_results"):
        """
        Initialize model comparison framework.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmark = ModelBenchmark()
        self.results = []
        
    def create_model_configs(self) -> List[ModelConfig]:
        """Create comprehensive list of model configurations to compare."""
        configs = []
        
        # Base YOLO variants
        yolo_variants = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        
        for variant in yolo_variants:
            # Base model
            configs.append(ModelConfig(
                name=f"{variant}_base",
                variant=variant,
                description=f"Base {variant} model without attention"
            ))
            
            # With CBAM attention
            configs.append(ModelConfig(
                name=f"{variant}_cbam",
                variant=variant,
                attention_type="cbam",
                description=f"{variant} with CBAM attention mechanism"
            ))
            
            # With other attention mechanisms (for smaller models)
            if variant in ["yolov8n", "yolov8s"]:
                for att_type in ["se", "eca", "coord"]:
                    configs.append(ModelConfig(
                        name=f"{variant}_{att_type}",
                        variant=variant,
                        attention_type=att_type,
                        description=f"{variant} with {att_type.upper()} attention"
                    ))
        
        # Special configurations
        configs.extend([
            ModelConfig(
                name="yolov8n_optimized",
                variant="yolov8n",
                attention_type="cbam",
                use_advanced_loss=True,
                use_multiscale=True,
                description="YOLOv8n with all optimizations"
            ),
            ModelConfig(
                name="yolov8s_optimized",
                variant="yolov8s",
                attention_type="cbam",
                use_advanced_loss=True,
                use_multiscale=True,
                description="YOLOv8s with all optimizations"
            )
        ])
        
        return configs
    
    def run_comparison(self, configs: Optional[List[ModelConfig]] = None) -> List[ComparisonResult]:
        """
        Run comprehensive model comparison.
        
        Args:
            configs: List of model configurations to compare
            
        Returns:
            List of comparison results
        """
        if configs is None:
            configs = self.create_model_configs()
        
        print(f"Running comparison on {len(configs)} model configurations...")
        
        results = []
        for i, config in enumerate(configs):
            print(f"\nProgress: {i+1}/{len(configs)}")
            result = self.benchmark.benchmark_model(config)
            results.append(result)
            
            # Print quick summary
            if "failed" not in result.notes.lower():
                print(f"  Parameters: {result.metrics.parameters:,}")
                print(f"  Inference: {result.metrics.inference_time_ms:.2f}ms")
                print(f"  FPS: {result.metrics.fps:.1f}")
                print(f"  Model size: {result.metrics.model_size_mb:.1f}MB")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze comparison results and generate insights.
        
        Returns:
            Analysis results dictionary
        """
        if not self.results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in self.results if "failed" not in r.notes.lower()]
        
        if not successful_results:
            return {"error": "No successful benchmarks"}
        
        analysis = {
            'total_models': len(self.results),
            'successful_models': len(successful_results),
            'failed_models': len(self.results) - len(successful_results)
        }
        
        # Performance analysis
        metrics_data = {
            'parameters': [r.metrics.parameters for r in successful_results],
            'inference_time': [r.metrics.inference_time_ms for r in successful_results],
            'fps': [r.metrics.fps for r in successful_results],
            'model_size': [r.metrics.model_size_mb for r in successful_results],
            'flops': [r.metrics.flops for r in successful_results],
            'memory_usage': [r.metrics.memory_usage_mb for r in successful_results]
        }
        
        # Calculate statistics
        for metric, values in metrics_data.items():
            if values:
                analysis[f'{metric}_stats'] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values)
                }
        
        # Find best models by different criteria
        analysis['best_models'] = {}
        
        if successful_results:
            # Fastest inference
            fastest = min(successful_results, key=lambda r: r.metrics.inference_time_ms)
            analysis['best_models']['fastest'] = {
                'name': fastest.config.name,
                'inference_time_ms': fastest.metrics.inference_time_ms,
                'fps': fastest.metrics.fps
            }
            
            # Smallest model
            smallest = min(successful_results, key=lambda r: r.metrics.model_size_mb)
            analysis['best_models']['smallest'] = {
                'name': smallest.config.name,
                'model_size_mb': smallest.metrics.model_size_mb,
                'parameters': smallest.metrics.parameters
            }
            
            # Most efficient (FPS per MB)
            efficient_results = [r for r in successful_results if r.metrics.model_size_mb > 0]
            if efficient_results:
                most_efficient = max(efficient_results, 
                                   key=lambda r: r.metrics.fps / r.metrics.model_size_mb)
                analysis['best_models']['most_efficient'] = {
                    'name': most_efficient.config.name,
                    'efficiency_fps_per_mb': most_efficient.metrics.fps / most_efficient.metrics.model_size_mb,
                    'fps': most_efficient.metrics.fps,
                    'model_size_mb': most_efficient.metrics.model_size_mb
                }
        
        return analysis
    
    def save_results(self, filename: str = "comparison_results"):
        """
        Save comparison results to files.
        
        Args:
            filename: Base filename for results
        """
        def convert_for_json(obj):
            """Convert numpy types and other non-serializable objects for JSON."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif hasattr(obj, '__dict__'):
                return convert_for_json(obj.__dict__)
            return obj
        
        # Save as JSON
        json_data = {
            'results': [convert_for_json(result.to_dict()) for result in self.results],
            'analysis': convert_for_json(self.analyze_results())
        }
        
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        # Save as CSV
        csv_path = self.output_dir / f"{filename}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'Model Name', 'Variant', 'Attention Type', 'Parameters', 
                'Model Size (MB)', 'Inference Time (ms)', 'FPS', 
                'FLOPs (G)', 'Memory Usage (MB)', 'Notes'
            ]
            writer.writerow(header)
            
            # Data rows
            for result in self.results:
                row = [
                    result.config.name,
                    result.config.variant,
                    result.config.attention_type or 'None',
                    result.metrics.parameters,
                    f"{result.metrics.model_size_mb:.2f}",
                    f"{result.metrics.inference_time_ms:.2f}",
                    f"{result.metrics.fps:.1f}",
                    f"{result.metrics.flops:.2f}",
                    f"{result.metrics.memory_usage_mb:.2f}",
                    result.notes
                ]
                writer.writerow(row)
        
        print(f"Results saved to {json_path} and {csv_path}")
    
    def visualize_results(self):
        """Generate visualization plots for comparison results."""
        if not self.results:
            print("No results to visualize")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if "failed" not in r.notes.lower()]
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data
        names = [r.config.name for r in successful_results]
        parameters = [r.metrics.parameters / 1e6 for r in successful_results]  # In millions
        inference_times = [r.metrics.inference_time_ms for r in successful_results]
        fps_values = [r.metrics.fps for r in successful_results]
        model_sizes = [r.metrics.model_size_mb for r in successful_results]
        flops = [r.metrics.flops for r in successful_results]
        memory_usage = [r.metrics.memory_usage_mb for r in successful_results]
        
        # Parameters vs Inference Time
        ax = axes[0, 0]
        scatter = ax.scatter(parameters, inference_times, c=model_sizes, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Parameters vs Inference Time')
        plt.colorbar(scatter, ax=ax, label='Model Size (MB)')
        
        # Model Size vs FPS
        ax = axes[0, 1]
        ax.scatter(model_sizes, fps_values, alpha=0.7)
        ax.set_xlabel('Model Size (MB)')
        ax.set_ylabel('FPS')
        ax.set_title('Model Size vs FPS')
        
        # FLOPs vs Inference Time
        ax = axes[0, 2]
        ax.scatter(flops, inference_times, alpha=0.7)
        ax.set_xlabel('FLOPs (G)')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('FLOPs vs Inference Time')
        
        # Parameters distribution
        ax = axes[1, 0]
        ax.hist(parameters, bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('Count')
        ax.set_title('Parameter Distribution')
        
        # FPS distribution
        ax = axes[1, 1]
        ax.hist(fps_values, bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('FPS')
        ax.set_ylabel('Count')
        ax.set_title('FPS Distribution')
        
        # Efficiency plot (FPS per MB)
        ax = axes[1, 2]
        efficiency = [fps / size if size > 0 else 0 for fps, size in zip(fps_values, model_sizes)]
        ax.bar(range(len(names)), efficiency, alpha=0.7)
        ax.set_xlabel('Model Index')
        ax.set_ylabel('FPS per MB')
        ax.set_title('Model Efficiency (FPS/MB)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "model_comparison_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {plot_path}")


def run_comprehensive_architecture_comparison():
    """Run comprehensive model architecture comparison for all YOLO variants and attention mechanisms."""
    print("Starting Comprehensive Model Architecture Comparison")
    print("=" * 60)
    
    # Initialize comparison framework
    comparison = ModelComparison(output_dir="model_comparison_results")
    
    # Create comprehensive model configurations
    configs = comparison.create_model_configs()
    print(f"Created {len(configs)} model configurations")
    
    # Run full comparison
    print(f"Running comprehensive comparison on all {len(configs)} configurations...")
    print("This may take several minutes...")
    
    results = comparison.run_comparison(configs)
    
    # Analyze results
    analysis = comparison.analyze_results()
    
    # Save results
    comparison.save_results("comprehensive_comparison")
    
    # Generate visualizations
    comparison.visualize_results()
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("Comprehensive Model Architecture Comparison Complete")
    print(f"Total models tested: {analysis.get('total_models', 0)}")
    print(f"Successful benchmarks: {analysis.get('successful_models', 0)}")
    print(f"Failed benchmarks: {analysis.get('failed_models', 0)}")
    
    # Performance analysis by variant
    successful_results = [r for r in results if "failed" not in r.notes.lower()]
    
    if successful_results:
        print("\n" + "Performance Analysis by YOLO Variant:")
        print("-" * 50)
        
        variant_stats = {}
        for result in successful_results:
            variant = result.config.variant
            if variant not in variant_stats:
                variant_stats[variant] = []
            variant_stats[variant].append(result)
        
        for variant, variant_results in sorted(variant_stats.items()):
            avg_params = np.mean([r.metrics.parameters for r in variant_results])
            avg_fps = np.mean([r.metrics.fps for r in variant_results])
            avg_size = np.mean([r.metrics.model_size_mb for r in variant_results])
            
            print(f"{variant:8s}: {len(variant_results):2d} configs, "
                  f"Avg: {avg_params/1e6:5.1f}M params, {avg_fps:5.1f} FPS, {avg_size:5.1f}MB")
    
    # Attention mechanism analysis
    if successful_results:
        print("\n" + "Performance Analysis by Attention Mechanism:")
        print("-" * 50)
        
        attention_stats = {}
        for result in successful_results:
            att_type = result.config.attention_type or "none"
            if att_type not in attention_stats:
                attention_stats[att_type] = []
            attention_stats[att_type].append(result)
        
        for att_type, att_results in sorted(attention_stats.items()):
            avg_params = np.mean([r.metrics.parameters for r in att_results])
            avg_fps = np.mean([r.metrics.fps for r in att_results])
            avg_inference = np.mean([r.metrics.inference_time_ms for r in att_results])
            
            print(f"{att_type:8s}: {len(att_results):2d} configs, "
                  f"Avg: {avg_params/1e6:5.1f}M params, {avg_fps:5.1f} FPS, {avg_inference:5.1f}ms")
    
    # Best models analysis
    if 'best_models' in analysis:
        print("\n" + "Best Model Configurations:")
        print("-" * 40)
        best = analysis['best_models']
        
        if 'fastest' in best:
            print(f"Fastest inference: {best['fastest']['name']}")
            print(f"  - {best['fastest']['fps']:.1f} FPS ({best['fastest']['inference_time_ms']:.2f}ms)")
        
        if 'smallest' in best:
            print(f"Smallest model: {best['smallest']['name']}")
            print(f"  - {best['smallest']['model_size_mb']:.1f}MB ({best['smallest']['parameters']:,} params)")
        
        if 'most_efficient' in best:
            print(f"Most efficient: {best['most_efficient']['name']}")
            print(f"  - {best['most_efficient']['efficiency_fps_per_mb']:.2f} FPS/MB")
    
    # Speed vs Accuracy tradeoff analysis
    if successful_results:
        print("\n" + "Speed vs Size Tradeoff Analysis:")
        print("-" * 40)
        
        # Sort by FPS (descending)
        by_speed = sorted(successful_results, key=lambda r: r.metrics.fps, reverse=True)
        print("Top 3 fastest models:")
        for i, result in enumerate(by_speed[:3]):
            print(f"  {i+1}. {result.config.name}: {result.metrics.fps:.1f} FPS, "
                  f"{result.metrics.model_size_mb:.1f}MB")
        
        # Sort by model size (ascending)
        by_size = sorted(successful_results, key=lambda r: r.metrics.model_size_mb)
        print("\nTop 3 smallest models:")
        for i, result in enumerate(by_size[:3]):
            print(f"  {i+1}. {result.config.name}: {result.metrics.model_size_mb:.1f}MB, "
                  f"{result.metrics.fps:.1f} FPS")
    
    # Recommendations
    print("\n" + "Recommendations:")
    print("-" * 20)
    
    if successful_results:
        # Find balanced model (good FPS and reasonable size)
        efficiency_scores = []
        for result in successful_results:
            # Normalize FPS and inverse of size, then combine
            fps_norm = result.metrics.fps / max(r.metrics.fps for r in successful_results)
            size_norm = min(r.metrics.model_size_mb for r in successful_results) / result.metrics.model_size_mb
            efficiency = (fps_norm + size_norm) / 2
            efficiency_scores.append((efficiency, result))
        
        best_balanced = max(efficiency_scores, key=lambda x: x[0])[1]
        
        print(f"For balanced performance: {best_balanced.config.name}")
        print(f"  - {best_balanced.metrics.fps:.1f} FPS, {best_balanced.metrics.model_size_mb:.1f}MB")
        
        # Find best with attention
        with_attention = [r for r in successful_results if r.config.attention_type]
        if with_attention:
            best_attention = max(with_attention, key=lambda r: r.metrics.fps)
            print(f"For attention-enhanced model: {best_attention.config.name}")
            print(f"  - {best_attention.metrics.fps:.1f} FPS with {best_attention.config.attention_type} attention")
    
    return results, analysis


def run_model_architecture_comparison():
    """Run comprehensive model architecture comparison (wrapper for backward compatibility)."""
    return run_comprehensive_architecture_comparison()


if __name__ == "__main__":
    run_model_architecture_comparison()