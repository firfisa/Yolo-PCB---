"""
Performance monitoring system for PCB defect detection.
Tracks inference speed, resource usage, and system metrics.
"""

import time
import psutil
import threading
import json
import csv
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import logging
from contextlib import contextmanager

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    inference_time: float  # seconds
    cpu_usage: float  # percentage
    memory_usage: float  # MB
    gpu_usage: Optional[float] = None  # percentage
    gpu_memory: Optional[float] = None  # MB
    throughput: Optional[float] = None  # images/second
    batch_size: Optional[int] = None
    model_name: Optional[str] = None
    image_size: Optional[tuple] = None


@dataclass
class SystemInfo:
    """System information data structure."""
    cpu_count: int
    total_memory: float  # GB
    gpu_count: int
    gpu_names: List[str]
    python_version: str
    torch_version: Optional[str] = None


class ResourceTracker:
    """Tracks system resource usage during inference."""
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize resource tracker.
        
        Args:
            sampling_interval: Interval between resource measurements (seconds)
        """
        self.sampling_interval = sampling_interval
        self.is_tracking = False
        self.tracking_thread = None
        self.metrics_buffer = deque(maxlen=1000)  # Keep last 1000 measurements
        self.logger = logging.getLogger(__name__)
        
    def start_tracking(self):
        """Start resource tracking in background thread."""
        if self.is_tracking:
            return
        
        self.is_tracking = True
        self.tracking_thread = threading.Thread(target=self._track_resources)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        self.logger.info("Resource tracking started")
        
    def stop_tracking(self):
        """Stop resource tracking."""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        self.logger.info("Resource tracking stopped")
        
    def _track_resources(self):
        """Background thread function for resource tracking."""
        while self.is_tracking:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                
                # GPU usage if available
                gpu_usage = None
                gpu_memory = None
                
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            gpu_usage = gpu.load * 100
                            gpu_memory = gpu.memoryUsed
                    except Exception as e:
                        self.logger.debug(f"GPU monitoring error: {e}")
                
                # Store metrics
                metrics = {
                    'timestamp': time.time(),
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_mb,
                    'gpu_usage': gpu_usage,
                    'gpu_memory': gpu_memory
                }
                
                self.metrics_buffer.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Resource tracking error: {e}")
            
            time.sleep(self.sampling_interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        if not self.metrics_buffer:
            return {}
        
        return self.metrics_buffer[-1].copy()
    
    def get_average_metrics(self, duration: float = 10.0) -> Dict[str, Any]:
        """
        Get average metrics over specified duration.
        
        Args:
            duration: Duration in seconds to average over
            
        Returns:
            Dictionary with average metrics
        """
        if not self.metrics_buffer:
            return {}
        
        current_time = time.time()
        cutoff_time = current_time - duration
        
        # Filter metrics within duration
        recent_metrics = [
            m for m in self.metrics_buffer 
            if m['timestamp'] >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_metrics = {}
        for key in ['cpu_usage', 'memory_usage', 'gpu_usage', 'gpu_memory']:
            values = [m[key] for m in recent_metrics if m[key] is not None]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'max_{key}'] = np.max(values)
                avg_metrics[f'min_{key}'] = np.min(values)
        
        return avg_metrics


class InferenceTimer:
    """Context manager for timing inference operations."""
    
    def __init__(self, operation_name: str = "inference"):
        """
        Initialize inference timer.
        
        Args:
            operation_name: Name of the operation being timed
        """
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
    def get_duration(self) -> float:
        """Get duration in seconds."""
        return self.duration if self.duration is not None else 0.0


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, log_file: Optional[str] = None, 
                 auto_save_interval: float = 60.0):
        """
        Initialize performance monitor.
        
        Args:
            log_file: Optional file to save performance logs
            auto_save_interval: Interval for auto-saving metrics (seconds)
        """
        self.log_file = log_file
        self.auto_save_interval = auto_save_interval
        self.metrics_history = []
        self.resource_tracker = ResourceTracker()
        self.logger = logging.getLogger(__name__)
        
        # Auto-save timer
        self.auto_save_timer = None
        self.is_monitoring = False
        
        # Performance statistics
        self.stats = defaultdict(list)
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.resource_tracker.start_tracking()
        
        # Start auto-save timer if log file specified
        if self.log_file and self.auto_save_interval > 0:
            self._start_auto_save()
        
        self.logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.resource_tracker.stop_tracking()
        
        # Stop auto-save timer
        if self.auto_save_timer:
            self.auto_save_timer.cancel()
        
        # Save final metrics if log file specified
        if self.log_file:
            self.save_metrics(self.log_file)
        
        self.logger.info("Performance monitoring stopped")
        
    def _start_auto_save(self):
        """Start auto-save timer."""
        if self.auto_save_timer:
            self.auto_save_timer.cancel()
        
        self.auto_save_timer = threading.Timer(
            self.auto_save_interval, self._auto_save_callback
        )
        self.auto_save_timer.start()
        
    def _auto_save_callback(self):
        """Auto-save callback function."""
        try:
            if self.log_file:
                self.save_metrics(self.log_file)
        except Exception as e:
            self.logger.error(f"Auto-save error: {e}")
        
        # Schedule next auto-save
        if self.is_monitoring:
            self._start_auto_save()
    
    @contextmanager
    def measure_inference(self, model_name: str = "unknown", 
                         batch_size: int = 1,
                         image_size: Optional[tuple] = None):
        """
        Context manager for measuring inference performance.
        
        Args:
            model_name: Name of the model being used
            batch_size: Batch size for inference
            image_size: Size of input images
            
        Usage:
            with monitor.measure_inference("yolov8n", batch_size=4):
                predictions = model.predict(images)
        """
        # Get baseline resource usage
        baseline_metrics = self.resource_tracker.get_current_metrics()
        
        # Start timing
        timer = InferenceTimer("inference")
        timer.__enter__()
        
        try:
            yield timer
        finally:
            # Stop timing
            timer.__exit__(None, None, None)
            
            # Get final resource usage
            final_metrics = self.resource_tracker.get_current_metrics()
            
            # Calculate throughput
            inference_time = timer.get_duration()
            throughput = batch_size / inference_time if inference_time > 0 else 0.0
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                inference_time=inference_time,
                cpu_usage=final_metrics.get('cpu_usage', 0.0),
                memory_usage=final_metrics.get('memory_usage', 0.0),
                gpu_usage=final_metrics.get('gpu_usage'),
                gpu_memory=final_metrics.get('gpu_memory'),
                throughput=throughput,
                batch_size=batch_size,
                model_name=model_name,
                image_size=image_size
            )
            
            # Store metrics
            self.record_metrics(metrics)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        
        # Update statistics
        self.stats['inference_times'].append(metrics.inference_time)
        self.stats['throughputs'].append(metrics.throughput or 0.0)
        self.stats['cpu_usage'].append(metrics.cpu_usage)
        self.stats['memory_usage'].append(metrics.memory_usage)
        
        if metrics.gpu_usage is not None:
            self.stats['gpu_usage'].append(metrics.gpu_usage)
        if metrics.gpu_memory is not None:
            self.stats['gpu_memory'].append(metrics.gpu_memory)
        
        self.logger.debug(f"Recorded metrics: {metrics.model_name}, "
                         f"time={metrics.inference_time:.4f}s, "
                         f"throughput={metrics.throughput:.2f} img/s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
        
        summary = {}
        
        # Calculate statistics for each metric
        for metric_name, values in self.stats.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
        
        # Add model-specific statistics
        model_stats = defaultdict(list)
        for metrics in self.metrics_history:
            if metrics.model_name:
                model_stats[metrics.model_name].append(metrics.inference_time)
        
        summary['model_performance'] = {}
        for model_name, times in model_stats.items():
            summary['model_performance'][model_name] = {
                'mean_inference_time': np.mean(times),
                'std_inference_time': np.std(times),
                'count': len(times)
            }
        
        # System information
        summary['system_info'] = self.get_system_info()
        
        return summary
    
    def get_system_info(self) -> SystemInfo:
        """Get system information."""
        import sys
        
        # CPU and memory info
        cpu_count = psutil.cpu_count()
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        # GPU info
        gpu_count = 0
        gpu_names = []
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_count = len(gpus)
                gpu_names = [gpu.name for gpu in gpus]
            except Exception:
                pass
        
        # Python and framework versions
        python_version = sys.version
        torch_version = None
        
        if TORCH_AVAILABLE:
            torch_version = torch.__version__
        
        return SystemInfo(
            cpu_count=cpu_count,
            total_memory=total_memory,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            python_version=python_version,
            torch_version=torch_version
        )
    
    def save_metrics(self, file_path: str, format: str = "json"):
        """
        Save performance metrics to file.
        
        Args:
            file_path: Output file path
            format: Output format ("json" or "csv")
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            self._save_json(file_path)
        elif format.lower() == "csv":
            self._save_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved {len(self.metrics_history)} metrics to {file_path}")
    
    def _save_json(self, file_path: str):
        """Save metrics as JSON."""
        data = {
            'metrics': [asdict(m) for m in self.metrics_history],
            'summary': self.get_performance_summary(),
            'timestamp': time.time()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_csv(self, file_path: str):
        """Save metrics as CSV."""
        if not self.metrics_history:
            return
        
        fieldnames = list(asdict(self.metrics_history[0]).keys())
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics in self.metrics_history:
                writer.writerow(asdict(metrics))
    
    def load_metrics(self, file_path: str):
        """Load metrics from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Load metrics
        self.metrics_history = []
        for metric_data in data.get('metrics', []):
            metrics = PerformanceMetrics(**metric_data)
            self.metrics_history.append(metrics)
        
        # Rebuild statistics
        self.stats = defaultdict(list)
        for metrics in self.metrics_history:
            self.record_metrics(metrics)
        
        self.logger.info(f"Loaded {len(self.metrics_history)} metrics from {file_path}")
    
    def generate_report(self, output_path: str):
        """Generate comprehensive performance report."""
        summary = self.get_performance_summary()
        
        # Create HTML report
        html_content = self._generate_html_report(summary)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated performance report: {output_path}")
    
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML performance report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PCB Detection Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { margin: 10px 0; }
                .section { margin: 20px 0; border: 1px solid #ccc; padding: 15px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>PCB Detection Performance Report</h1>
            <p>Generated at: {timestamp}</p>
        """.format(timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
        
        # System information
        if 'system_info' in summary:
            sys_info = summary['system_info']
            html += f"""
            <div class="section">
                <h2>System Information</h2>
                <p>CPU Cores: {sys_info.get('cpu_count', 'N/A')}</p>
                <p>Total Memory: {sys_info.get('total_memory', 'N/A'):.1f} GB</p>
                <p>GPU Count: {sys_info.get('gpu_count', 0)}</p>
                <p>GPU Names: {', '.join(sys_info.get('gpu_names', []))}</p>
                <p>Python Version: {sys_info.get('python_version', 'N/A')}</p>
                <p>PyTorch Version: {sys_info.get('torch_version', 'N/A')}</p>
            </div>
            """
        
        # Performance statistics
        html += '<div class="section"><h2>Performance Statistics</h2><table>'
        html += '<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>'
        
        for metric_name, stats in summary.items():
            if isinstance(stats, dict) and 'mean' in stats:
                html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{stats['mean']:.4f}</td>
                    <td>{stats['std']:.4f}</td>
                    <td>{stats['min']:.4f}</td>
                    <td>{stats['max']:.4f}</td>
                    <td>{stats['count']}</td>
                </tr>
                """
        
        html += '</table></div>'
        
        # Model performance
        if 'model_performance' in summary:
            html += '<div class="section"><h2>Model Performance</h2><table>'
            html += '<tr><th>Model</th><th>Mean Inference Time (s)</th><th>Std (s)</th><th>Count</th></tr>'
            
            for model_name, stats in summary['model_performance'].items():
                html += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{stats['mean_inference_time']:.4f}</td>
                    <td>{stats['std_inference_time']:.4f}</td>
                    <td>{stats['count']}</td>
                </tr>
                """
            
            html += '</table></div>'
        
        html += '</body></html>'
        return html
    
    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics_history.clear()
        self.stats.clear()
        self.logger.info("Cleared all performance metrics")


# Global performance monitor instance
_global_monitor = None

def get_global_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def start_monitoring(log_file: Optional[str] = None):
    """Start global performance monitoring."""
    monitor = get_global_monitor()
    if log_file:
        monitor.log_file = log_file
    monitor.start_monitoring()

def stop_monitoring():
    """Stop global performance monitoring."""
    monitor = get_global_monitor()
    monitor.stop_monitoring()

@contextmanager
def measure_inference(model_name: str = "unknown", 
                     batch_size: int = 1,
                     image_size: Optional[tuple] = None):
    """Global inference measurement context manager."""
    monitor = get_global_monitor()
    with monitor.measure_inference(model_name, batch_size, image_size) as timer:
        yield timer