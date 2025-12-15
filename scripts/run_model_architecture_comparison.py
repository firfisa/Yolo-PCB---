#!/usr/bin/env python3
"""
Script to run comprehensive model architecture comparison for task 10.5.
Compares different YOLO variants (YOLOv8n/s/m/l/x) and attention mechanisms (CBAM/SE/ECA/CoordAtt).
Analyzes accuracy vs speed tradeoffs and selects optimal configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from typing import Dict, List, Any

from pcb_detection.evaluation.model_comparison import (
    ModelComparison,
    ModelConfig,
    run_comprehensive_architecture_comparison
)


def create_focused_model_configs() -> List[ModelConfig]:
    """Create focused set of model configurations for comprehensive comparison."""
    configs = []
    
    # All YOLO variants
    yolo_variants = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    
    # All attention mechanisms
    attention_types = [None, "cbam", "se", "eca", "coord"]
    
    print("Creating model configurations...")
    
    # Base models (no attention)
    for variant in yolo_variants:
        configs.append(ModelConfig(
            name=f"{variant}_base",
            variant=variant,
            attention_type=None,
            description=f"Base {variant} model without attention"
        ))
    
    # Models with attention (focus on smaller variants for attention comparison)
    focus_variants = ["yolov8n", "yolov8s", "yolov8m"]  # Focus on practical variants
    
    for variant in focus_variants:
        for att_type in ["cbam", "se", "eca", "coord"]:
            configs.append(ModelConfig(
                name=f"{variant}_{att_type}",
                variant=variant,
                attention_type=att_type,
                description=f"{variant} with {att_type.upper()} attention mechanism"
            ))
    
    # Optimized configurations
    configs.extend([
        ModelConfig(
            name="yolov8n_optimized",
            variant="yolov8n",
            attention_type="cbam",
            use_advanced_loss=True,
            use_multiscale=True,
            description="YOLOv8n with CBAM + advanced loss + multiscale"
        ),
        ModelConfig(
            name="yolov8s_optimized",
            variant="yolov8s",
            attention_type="cbam",
            use_advanced_loss=True,
            use_multiscale=True,
            description="YOLOv8s with CBAM + advanced loss + multiscale"
        ),
        ModelConfig(
            name="yolov8m_optimized",
            variant="yolov8m",
            attention_type="se",
            use_advanced_loss=True,
            description="YOLOv8m with SE attention + advanced loss"
        )
    ])
    
    print(f"Created {len(configs)} model configurations")
    return configs


def analyze_variant_performance(results: List) -> Dict[str, Any]:
    """Analyze performance differences between YOLO variants."""
    print("\nAnalyzing YOLO variant performance...")
    
    successful_results = [r for r in results if "failed" not in r.notes.lower()]
    
    if not successful_results:
        return {"error": "No successful results to analyze"}
    
    # Group by variant
    variant_groups = {}
    for result in successful_results:
        variant = result.config.variant
        if variant not in variant_groups:
            variant_groups[variant] = []
        variant_groups[variant].append(result)
    
    variant_analysis = {}
    
    for variant, group_results in variant_groups.items():
        # Calculate statistics
        params = [r.metrics.parameters for r in group_results]
        fps_values = [r.metrics.fps for r in group_results]
        sizes = [r.metrics.model_size_mb for r in group_results]
        inference_times = [r.metrics.inference_time_ms for r in group_results]
        
        variant_analysis[variant] = {
            'count': len(group_results),
            'avg_parameters': sum(params) / len(params),
            'avg_fps': sum(fps_values) / len(fps_values),
            'avg_size_mb': sum(sizes) / len(sizes),
            'avg_inference_ms': sum(inference_times) / len(inference_times),
            'min_fps': min(fps_values),
            'max_fps': max(fps_values),
            'min_size': min(sizes),
            'max_size': max(sizes)
        }
    
    return variant_analysis


def analyze_attention_impact(results: List) -> Dict[str, Any]:
    """Analyze the impact of different attention mechanisms."""
    print("Analyzing attention mechanism impact...")
    
    successful_results = [r for r in results if "failed" not in r.notes.lower()]
    
    # Group by attention type
    attention_groups = {}
    for result in successful_results:
        att_type = result.config.attention_type or "none"
        if att_type not in attention_groups:
            attention_groups[att_type] = []
        attention_groups[att_type].append(result)
    
    attention_analysis = {}
    
    for att_type, group_results in attention_groups.items():
        # Calculate statistics
        fps_values = [r.metrics.fps for r in group_results]
        params = [r.metrics.parameters for r in group_results]
        inference_times = [r.metrics.inference_time_ms for r in group_results]
        
        attention_analysis[att_type] = {
            'count': len(group_results),
            'avg_fps': sum(fps_values) / len(fps_values),
            'avg_parameters': sum(params) / len(params),
            'avg_inference_ms': sum(inference_times) / len(inference_times),
            'fps_range': (min(fps_values), max(fps_values)),
            'param_overhead': 0.0  # Will calculate relative to base
        }
    
    # Calculate parameter overhead relative to base models
    if 'none' in attention_analysis:
        base_params = attention_analysis['none']['avg_parameters']
        for att_type in attention_analysis:
            if att_type != 'none':
                overhead = (attention_analysis[att_type]['avg_parameters'] - base_params) / base_params * 100
                attention_analysis[att_type]['param_overhead'] = overhead
    
    return attention_analysis


def find_optimal_configurations(results: List) -> Dict[str, Any]:
    """Find optimal model configurations for different use cases."""
    print("Finding optimal configurations...")
    
    successful_results = [r for r in results if "failed" not in r.notes.lower()]
    
    if not successful_results:
        return {"error": "No successful results"}
    
    optimal_configs = {}
    
    # Fastest model (highest FPS)
    fastest = max(successful_results, key=lambda r: r.metrics.fps)
    optimal_configs['fastest'] = {
        'name': fastest.config.name,
        'variant': fastest.config.variant,
        'attention': fastest.config.attention_type,
        'fps': fastest.metrics.fps,
        'inference_ms': fastest.metrics.inference_time_ms,
        'size_mb': fastest.metrics.model_size_mb,
        'parameters': fastest.metrics.parameters
    }
    
    # Smallest model (lowest memory footprint)
    smallest = min(successful_results, key=lambda r: r.metrics.model_size_mb)
    optimal_configs['smallest'] = {
        'name': smallest.config.name,
        'variant': smallest.config.variant,
        'attention': smallest.config.attention_type,
        'fps': smallest.metrics.fps,
        'size_mb': smallest.metrics.model_size_mb,
        'parameters': smallest.metrics.parameters
    }
    
    # Most efficient (best FPS per MB)
    most_efficient = max(successful_results, 
                        key=lambda r: r.metrics.fps / r.metrics.model_size_mb if r.metrics.model_size_mb > 0 else 0)
    optimal_configs['most_efficient'] = {
        'name': most_efficient.config.name,
        'variant': most_efficient.config.variant,
        'attention': most_efficient.config.attention_type,
        'fps': most_efficient.metrics.fps,
        'size_mb': most_efficient.metrics.model_size_mb,
        'efficiency': most_efficient.metrics.fps / most_efficient.metrics.model_size_mb
    }
    
    # Best with attention (highest FPS among attention models)
    attention_results = [r for r in successful_results if r.config.attention_type]
    if attention_results:
        best_attention = max(attention_results, key=lambda r: r.metrics.fps)
        optimal_configs['best_with_attention'] = {
            'name': best_attention.config.name,
            'variant': best_attention.config.variant,
            'attention': best_attention.config.attention_type,
            'fps': best_attention.metrics.fps,
            'size_mb': best_attention.metrics.model_size_mb,
            'parameters': best_attention.metrics.parameters
        }
    
    # Balanced model (good tradeoff between speed and size)
    # Calculate efficiency score: normalized FPS + normalized inverse size
    max_fps = max(r.metrics.fps for r in successful_results)
    min_size = min(r.metrics.model_size_mb for r in successful_results)
    
    efficiency_scores = []
    for result in successful_results:
        fps_norm = result.metrics.fps / max_fps
        size_norm = min_size / result.metrics.model_size_mb
        efficiency_score = (fps_norm + size_norm) / 2
        efficiency_scores.append((efficiency_score, result))
    
    balanced = max(efficiency_scores, key=lambda x: x[0])[1]
    optimal_configs['balanced'] = {
        'name': balanced.config.name,
        'variant': balanced.config.variant,
        'attention': balanced.config.attention_type,
        'fps': balanced.metrics.fps,
        'size_mb': balanced.metrics.model_size_mb,
        'efficiency_score': max(efficiency_scores, key=lambda x: x[0])[0]
    }
    
    return optimal_configs


def generate_recommendations(variant_analysis: Dict, attention_analysis: Dict, 
                           optimal_configs: Dict) -> Dict[str, str]:
    """Generate recommendations based on analysis results."""
    recommendations = {}
    
    # Speed-focused recommendation
    if 'fastest' in optimal_configs:
        fastest = optimal_configs['fastest']
        recommendations['speed_focused'] = (
            f"For maximum speed: {fastest['name']} "
            f"({fastest['fps']:.1f} FPS, {fastest['inference_ms']:.1f}ms inference)"
        )
    
    # Memory-constrained recommendation
    if 'smallest' in optimal_configs:
        smallest = optimal_configs['smallest']
        recommendations['memory_constrained'] = (
            f"For memory constraints: {smallest['name']} "
            f"({smallest['size_mb']:.1f}MB, {smallest['fps']:.1f} FPS)"
        )
    
    # Balanced recommendation
    if 'balanced' in optimal_configs:
        balanced = optimal_configs['balanced']
        recommendations['balanced'] = (
            f"For balanced performance: {balanced['name']} "
            f"({balanced['fps']:.1f} FPS, {balanced['size_mb']:.1f}MB)"
        )
    
    # Attention mechanism recommendation
    if 'best_with_attention' in optimal_configs:
        best_att = optimal_configs['best_with_attention']
        recommendations['with_attention'] = (
            f"For attention-enhanced detection: {best_att['name']} "
            f"({best_att['fps']:.1f} FPS with {best_att['attention']} attention)"
        )
    
    # Variant-specific recommendations
    if variant_analysis:
        # Find best performing variant
        best_variant = max(variant_analysis.keys(), 
                          key=lambda v: variant_analysis[v]['avg_fps'])
        recommendations['best_variant'] = (
            f"Best YOLO variant: {best_variant} "
            f"(avg {variant_analysis[best_variant]['avg_fps']:.1f} FPS)"
        )
    
    return recommendations


def save_comprehensive_analysis(results: List, variant_analysis: Dict, 
                              attention_analysis: Dict, optimal_configs: Dict,
                              recommendations: Dict, output_dir: str):
    """Save comprehensive analysis results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Comprehensive analysis report
    analysis_report = {
        'summary': {
            'total_configurations': len(results),
            'successful_benchmarks': len([r for r in results if "failed" not in r.notes.lower()]),
            'failed_benchmarks': len([r for r in results if "failed" in r.notes.lower()]),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'variant_analysis': variant_analysis,
        'attention_analysis': attention_analysis,
        'optimal_configurations': optimal_configs,
        'recommendations': recommendations,
        'detailed_results': [r.to_dict() for r in results]
    }
    
    # Save as JSON
    with open(output_path / "comprehensive_analysis.json", 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)
    
    # Save summary report as text
    with open(output_path / "analysis_summary.txt", 'w') as f:
        f.write("PCB Defect Detection - Model Architecture Comparison\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total configurations tested: {analysis_report['summary']['total_configurations']}\n")
        f.write(f"Successful benchmarks: {analysis_report['summary']['successful_benchmarks']}\n")
        f.write(f"Failed benchmarks: {analysis_report['summary']['failed_benchmarks']}\n\n")
        
        f.write("YOLO Variant Performance:\n")
        f.write("-" * 30 + "\n")
        for variant, stats in variant_analysis.items():
            f.write(f"{variant:8s}: {stats['avg_fps']:6.1f} FPS, "
                   f"{stats['avg_size_mb']:6.1f}MB, "
                   f"{stats['avg_parameters']/1e6:5.1f}M params\n")
        
        f.write("\nAttention Mechanism Impact:\n")
        f.write("-" * 30 + "\n")
        for att_type, stats in attention_analysis.items():
            overhead = f"+{stats['param_overhead']:4.1f}%" if stats['param_overhead'] > 0 else "  base"
            f.write(f"{att_type:8s}: {stats['avg_fps']:6.1f} FPS, "
                   f"{stats['avg_inference_ms']:6.1f}ms, {overhead} params\n")
        
        f.write("\nRecommendations:\n")
        f.write("-" * 20 + "\n")
        for use_case, recommendation in recommendations.items():
            f.write(f"{use_case.replace('_', ' ').title()}: {recommendation}\n")
    
    print(f"Comprehensive analysis saved to {output_path}")


def main():
    """Main function to run comprehensive model architecture comparison."""
    print("PCB Defect Detection - Model Architecture Comparison")
    print("=" * 60)
    print("Task 10.5: Compare YOLO variants and attention mechanisms")
    print("Analyzing accuracy vs speed tradeoffs for optimal configuration selection\n")
    
    start_time = time.time()
    
    # Create output directory
    output_dir = "model_comparison_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Run comprehensive comparison
        print("Phase 1: Running comprehensive model benchmarks...")
        results, basic_analysis = run_comprehensive_architecture_comparison()
        
        print(f"\nPhase 2: Analyzing {len(results)} benchmark results...")
        
        # Detailed analysis
        variant_analysis = analyze_variant_performance(results)
        attention_analysis = analyze_attention_impact(results)
        optimal_configs = find_optimal_configurations(results)
        recommendations = generate_recommendations(
            variant_analysis, attention_analysis, optimal_configs
        )
        
        # Save comprehensive analysis
        print("\nPhase 3: Saving comprehensive analysis...")
        save_comprehensive_analysis(
            results, variant_analysis, attention_analysis, 
            optimal_configs, recommendations, output_dir
        )
        
        # Print final summary
        elapsed_time = time.time() - start_time
        print(f"\n" + "=" * 60)
        print("Model Architecture Comparison Complete!")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Results saved to: {output_dir}/")
        
        # Print key findings
        successful_count = len([r for r in results if "failed" not in r.notes.lower()])
        print(f"\nKey Findings:")
        print(f"- Tested {len(results)} configurations, {successful_count} successful")
        
        if optimal_configs:
            if 'fastest' in optimal_configs:
                fastest = optimal_configs['fastest']
                print(f"- Fastest: {fastest['name']} ({fastest['fps']:.1f} FPS)")
            
            if 'smallest' in optimal_configs:
                smallest = optimal_configs['smallest']
                print(f"- Smallest: {smallest['name']} ({smallest['size_mb']:.1f}MB)")
            
            if 'balanced' in optimal_configs:
                balanced = optimal_configs['balanced']
                print(f"- Balanced: {balanced['name']} (efficiency score: {balanced['efficiency_score']:.3f})")
        
        print(f"\nDetailed analysis available in {output_dir}/comprehensive_analysis.json")
        
        return True
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)