#!/usr/bin/env python3
"""
Validation script for CBAM integration in YOLO models.
Tests attention mechanism integration and performance impact.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from typing import Dict, List
import matplotlib.pyplot as plt

from pcb_detection.models.yolo_detector import YOLODetector
from pcb_detection.models.cbam_integration import (
    create_cbam_enhanced_yolo, 
    CBAMConfig,
    CBAMIntegrator,
    CBAMTrainingOptimizer
)


def test_cbam_integration():
    """Test CBAM integration in YOLO models."""
    print("Testing CBAM Integration in YOLO Models")
    print("=" * 50)
    
    # Test different model variants
    variants = ["yolov8n", "yolov8s", "yolov8m"]
    results = {}
    
    for variant in variants:
        print(f"\nTesting {variant}...")
        
        # Create models with and without CBAM
        base_config = {
            "variant": variant,
            "attention_type": None,
            "input_size": 640,
            "conf_threshold": 0.25,
            "iou_threshold": 0.45
        }
        
        cbam_config = create_cbam_enhanced_yolo(variant, optimize_for_small_objects=True)
        
        try:
            # Initialize models
            base_model = YOLODetector(base_config)
            cbam_model = YOLODetector(cbam_config)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Base model
            start_time = time.time()
            with torch.no_grad():
                base_output = base_model.forward(dummy_input)
            base_time = time.time() - start_time
            
            # CBAM model
            start_time = time.time()
            with torch.no_grad():
                cbam_output = cbam_model.forward(dummy_input)
            cbam_time = time.time() - start_time
            
            # Calculate parameters
            base_params = sum(p.numel() for p in base_model.parameters())
            cbam_params = sum(p.numel() for p in cbam_model.parameters())
            
            # Store results
            results[variant] = {
                "base_params": base_params,
                "cbam_params": cbam_params,
                "param_increase": (cbam_params - base_params) / base_params * 100,
                "base_time": base_time,
                "cbam_time": cbam_time,
                "time_increase": (cbam_time - base_time) / base_time * 100,
                "output_shape": cbam_output.shape if hasattr(cbam_output, 'shape') else "list"
            }
            
            print(f"  ✓ Parameters: {base_params:,} → {cbam_params:,} (+{results[variant]['param_increase']:.1f}%)")
            print(f"  ✓ Inference time: {base_time:.4f}s → {cbam_time:.4f}s (+{results[variant]['time_increase']:.1f}%)")
            
        except Exception as e:
            print(f"  ✗ Error testing {variant}: {e}")
            results[variant] = {"error": str(e)}
    
    return results


def test_cbam_configurations():
    """Test different CBAM configurations."""
    print("\nTesting CBAM Configurations")
    print("=" * 30)
    
    variants = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    
    for variant in variants:
        config = CBAMConfig.get_config(variant)
        small_obj_config = CBAMConfig.get_small_object_config(variant)
        
        print(f"\n{variant}:")
        print(f"  Standard config: {config}")
        print(f"  Small object config: {small_obj_config}")


def test_attention_mechanisms():
    """Test different attention mechanisms."""
    print("\nTesting Attention Mechanisms")
    print("=" * 30)
    
    attention_types = ["cbam", "se", "eca", "coord"]
    input_tensor = torch.randn(1, 256, 32, 32)
    
    for att_type in attention_types:
        try:
            from pcb_detection.models.attention import AttentionBlock
            
            attention = AttentionBlock(256, att_type)
            
            start_time = time.time()
            with torch.no_grad():
                output = attention(input_tensor)
            inference_time = time.time() - start_time
            
            params = sum(p.numel() for p in attention.parameters())
            
            print(f"  {att_type.upper()}:")
            print(f"    Parameters: {params:,}")
            print(f"    Inference time: {inference_time:.4f}s")
            print(f"    Output shape: {output.shape}")
            
        except Exception as e:
            print(f"  {att_type.upper()}: Error - {e}")


def test_training_optimization():
    """Test training optimization for CBAM."""
    print("\nTesting Training Optimization")
    print("=" * 30)
    
    # Create CBAM-enhanced model
    config = create_cbam_enhanced_yolo("yolov8n")
    model = YOLODetector(config)
    
    # Test parameter grouping
    lr_config = CBAMTrainingOptimizer.get_attention_lr_schedule(0.01)
    param_groups = CBAMTrainingOptimizer.create_param_groups(model, lr_config)
    
    print(f"Learning rate config: {lr_config}")
    print(f"Parameter groups: {len(param_groups)}")
    
    for i, group in enumerate(param_groups):
        param_count = sum(p.numel() for p in group['params'])
        print(f"  Group {i} ({group.get('name', 'unnamed')}): {param_count:,} parameters, LR: {group['lr']}")


def visualize_results(results: Dict):
    """Visualize CBAM integration results."""
    print("\nGenerating Visualization...")
    
    variants = list(results.keys())
    param_increases = [results[v].get('param_increase', 0) for v in variants if 'error' not in results[v]]
    time_increases = [results[v].get('time_increase', 0) for v in variants if 'error' not in results[v]]
    valid_variants = [v for v in variants if 'error' not in results[v]]
    
    if not valid_variants:
        print("No valid results to visualize")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameter increase
    ax1.bar(valid_variants, param_increases, color='skyblue', alpha=0.7)
    ax1.set_title('Parameter Increase with CBAM')
    ax1.set_ylabel('Increase (%)')
    ax1.set_xlabel('Model Variant')
    ax1.tick_params(axis='x', rotation=45)
    
    # Time increase
    ax2.bar(valid_variants, time_increases, color='lightcoral', alpha=0.7)
    ax2.set_title('Inference Time Increase with CBAM')
    ax2.set_ylabel('Increase (%)')
    ax2.set_xlabel('Model Variant')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cbam_integration_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to cbam_integration_results.png")


def main():
    """Main validation function."""
    print("CBAM Integration Validation")
    print("=" * 50)
    
    # Test CBAM integration
    results = test_cbam_integration()
    
    # Test configurations
    test_cbam_configurations()
    
    # Test attention mechanisms
    test_attention_mechanisms()
    
    # Test training optimization
    test_training_optimization()
    
    # Visualize results
    try:
        visualize_results(results)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("CBAM Integration Validation Complete")
    
    successful_variants = [v for v in results.keys() if 'error' not in results[v]]
    failed_variants = [v for v in results.keys() if 'error' in results[v]]
    
    print(f"Successful integrations: {len(successful_variants)}")
    print(f"Failed integrations: {len(failed_variants)}")
    
    if successful_variants:
        avg_param_increase = np.mean([results[v]['param_increase'] for v in successful_variants])
        avg_time_increase = np.mean([results[v]['time_increase'] for v in successful_variants])
        
        print(f"Average parameter increase: {avg_param_increase:.1f}%")
        print(f"Average inference time increase: {avg_time_increase:.1f}%")
    
    if failed_variants:
        print(f"Failed variants: {failed_variants}")
        for v in failed_variants:
            print(f"  {v}: {results[v]['error']}")


if __name__ == "__main__":
    main()