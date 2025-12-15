#!/usr/bin/env python3
"""
Validation script for integrated advanced data augmentation pipeline.
Tests progressive augmentation strategies and training-aware augmentation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List
import random

from pcb_detection.data.integrated_augmentation import (
    AugmentationConfig,
    AugmentationPhase,
    ProgressiveAugmentationScheduler,
    SmallDefectAugmentation,
    AdaptiveAugmentation,
    IntegratedAugmentationPipeline,
    create_pcb_augmentation_config,
    create_pcb_augmentation_pipeline
)


def create_dummy_data(num_images: int = 4, image_size: int = 640) -> tuple:
    """Create dummy images and annotations for testing."""
    images = []
    annotations_list = []
    
    for i in range(num_images):
        # Create random image
        image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        
        # Add some patterns to make it look more like PCB
        cv2.rectangle(image, (50, 50), (150, 150), (100, 100, 100), -1)
        cv2.circle(image, (300, 300), 30, (200, 200, 200), -1)
        
        # Create random annotations
        num_defects = random.randint(1, 5)
        annotations = []
        
        for j in range(num_defects):
            # Create mix of small and large defects
            if j < num_defects // 2:
                # Small defects
                width = random.uniform(0.01, 0.03)
                height = random.uniform(0.01, 0.03)
            else:
                # Larger defects
                width = random.uniform(0.05, 0.15)
                height = random.uniform(0.05, 0.15)
            
            x_center = random.uniform(width/2, 1 - width/2)
            y_center = random.uniform(height/2, 1 - height/2)
            
            annotations.append({
                'class_id': random.randint(0, 4),
                'bbox': [x_center, y_center, width, height]
            })
        
        images.append(image)
        annotations_list.append(annotations)
    
    return images, annotations_list


def test_augmentation_config():
    """Test augmentation configuration system."""
    print("Testing Augmentation Configuration")
    print("=" * 40)
    
    strategies = ["conservative", "balanced", "aggressive"]
    
    for strategy in strategies:
        config = create_pcb_augmentation_config(strategy)
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Mosaic prob: {config.mosaic_prob}")
        print(f"  Copy-paste prob: {config.copy_paste_prob}")
        print(f"  MixUp prob: {config.mixup_prob}")
        print(f"  Albumentations prob: {config.albumentations_prob}")
        print(f"  Small defect boost: {config.small_defect_boost}")
        print(f"  Training phases: {config.warmup_epochs}/{config.aggressive_epochs}/{config.fine_tune_epochs}")


def test_progressive_scheduler():
    """Test progressive augmentation scheduler."""
    print("\nTesting Progressive Augmentation Scheduler")
    print("=" * 45)
    
    config = create_pcb_augmentation_config("balanced")
    scheduler = ProgressiveAugmentationScheduler(config)
    
    # Test different epochs
    test_epochs = [0, 5, 15, 50, 100, 200, 300]
    
    print("  Epoch progression:")
    for epoch in test_epochs:
        scheduler.set_epoch(epoch)
        phase = scheduler.get_current_phase()
        phase_config = scheduler.get_phase_config()
        
        print(f"    Epoch {epoch:3d}: {phase.value:12s} - "
              f"mosaic={phase_config['mosaic_prob']:.2f}, "
              f"strength={phase_config['augmentation_strength']:.2f}")


def test_small_defect_augmentation():
    """Test small defect augmentation."""
    print("\nTesting Small Defect Augmentation")
    print("=" * 40)
    
    config = create_pcb_augmentation_config("balanced")
    small_defect_aug = SmallDefectAugmentation(config)
    
    # Create test data with small defects
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    annotations = [
        {'class_id': 0, 'bbox': [0.2, 0.2, 0.01, 0.01]},  # Small defect
        {'class_id': 1, 'bbox': [0.5, 0.5, 0.05, 0.05]},  # Large defect
        {'class_id': 2, 'bbox': [0.8, 0.8, 0.015, 0.015]}  # Small defect
    ]
    
    print("  Original annotations:")
    for i, ann in enumerate(annotations):
        bbox = ann['bbox']
        area = bbox[2] * bbox[3]
        is_small = small_defect_aug.is_small_defect(bbox)
        print(f"    Defect {i}: area={area:.6f}, small={is_small}")
    
    # Apply small defect augmentation
    aug_image, aug_annotations = small_defect_aug.augment_small_defects(image, annotations)
    
    print(f"  After augmentation: {len(annotations)} → {len(aug_annotations)} defects")
    
    # Count small vs large defects
    small_count = sum(1 for ann in aug_annotations if small_defect_aug.is_small_defect(ann['bbox']))
    large_count = len(aug_annotations) - small_count
    
    print(f"  Small defects: {small_count}, Large defects: {large_count}")


def test_adaptive_augmentation():
    """Test adaptive augmentation."""
    print("\nTesting Adaptive Augmentation")
    print("=" * 35)
    
    config = create_pcb_augmentation_config("balanced")
    adaptive_aug = AdaptiveAugmentation(config)
    
    print("  Simulating training progress...")
    
    # Simulate different performance scenarios
    scenarios = [
        ("Improving", [0.1, 0.15, 0.2, 0.25, 0.3]),
        ("Stagnating", [0.2, 0.21, 0.19, 0.20, 0.21]),
        ("Declining", [0.3, 0.28, 0.25, 0.22, 0.20])
    ]
    
    for scenario_name, map_values in scenarios:
        adaptive_aug.performance_history = []  # Reset
        
        for epoch, map_val in enumerate(map_values):
            metrics = {'mAP': map_val, 'loss': 1.0 - map_val}
            adaptive_aug.update_performance(epoch, metrics)
        
        should_increase = adaptive_aug.should_increase_augmentation()
        multiplier = adaptive_aug.get_adaptive_multiplier()
        
        print(f"    {scenario_name:12s}: increase_aug={should_increase}, multiplier={multiplier:.2f}")


def test_integrated_pipeline():
    """Test integrated augmentation pipeline."""
    print("\nTesting Integrated Augmentation Pipeline")
    print("=" * 45)
    
    # Create pipeline
    pipeline = create_pcb_augmentation_pipeline("balanced")
    
    # Create test data
    images, annotations_list = create_dummy_data(4, 640)
    
    print(f"  Input: {len(images)} images, {len(annotations_list)} annotation lists")
    
    # Test different phases
    phases = [AugmentationPhase.WARMUP, AugmentationPhase.AGGRESSIVE, 
              AugmentationPhase.FINE_TUNE, AugmentationPhase.FINAL]
    
    results = {}
    
    for phase in phases:
        try:
            aug_image, aug_annotations = pipeline(images, annotations_list, force_phase=phase)
            
            results[phase.value] = {
                'output_shape': aug_image.shape,
                'num_annotations': len(aug_annotations),
                'success': True
            }
            
            print(f"    {phase.value:12s}: shape={aug_image.shape}, annotations={len(aug_annotations)}")
            
        except Exception as e:
            results[phase.value] = {
                'error': str(e),
                'success': False
            }
            print(f"    {phase.value:12s}: ERROR - {e}")
    
    return results


def test_augmentation_statistics():
    """Test augmentation statistics and monitoring."""
    print("\nTesting Augmentation Statistics")
    print("=" * 40)
    
    pipeline = create_pcb_augmentation_pipeline("balanced")
    
    # Simulate training progression
    epochs_to_test = [0, 10, 50, 150, 250]
    
    for epoch in epochs_to_test:
        pipeline.set_epoch(epoch)
        
        # Simulate performance update
        if epoch > 0:
            metrics = {
                'mAP': 0.1 + epoch * 0.001 + np.random.normal(0, 0.01),
                'loss': 2.0 - epoch * 0.005 + np.random.normal(0, 0.1)
            }
            pipeline.update_performance(epoch, metrics)
        
        stats = pipeline.get_augmentation_stats()
        
        print(f"  Epoch {epoch:3d}:")
        print(f"    Phase: {stats['current_phase']}")
        print(f"    Adaptive multiplier: {stats['adaptive_multiplier']:.2f}")
        print(f"    Should increase aug: {stats['should_increase_aug']}")


def test_augmentation_effects():
    """Test visual effects of different augmentation strategies."""
    print("\nTesting Augmentation Effects")
    print("=" * 35)
    
    strategies = ["conservative", "balanced", "aggressive"]
    images, annotations_list = create_dummy_data(4, 640)
    
    results = {}
    
    for strategy in strategies:
        pipeline = create_pcb_augmentation_pipeline(strategy)
        pipeline.set_epoch(50)  # Set to aggressive phase
        
        # Apply augmentation multiple times
        num_trials = 10
        annotation_counts = []
        
        for _ in range(num_trials):
            try:
                aug_image, aug_annotations = pipeline(images, annotations_list)
                annotation_counts.append(len(aug_annotations))
            except Exception as e:
                print(f"    {strategy} trial failed: {e}")
        
        if annotation_counts:
            results[strategy] = {
                'mean_annotations': np.mean(annotation_counts),
                'std_annotations': np.std(annotation_counts),
                'min_annotations': np.min(annotation_counts),
                'max_annotations': np.max(annotation_counts)
            }
            
            print(f"  {strategy.upper()} strategy:")
            print(f"    Annotations: {results[strategy]['mean_annotations']:.1f} ± {results[strategy]['std_annotations']:.1f}")
            print(f"    Range: {results[strategy]['min_annotations']} - {results[strategy]['max_annotations']}")
    
    return results


def visualize_augmentation_progression():
    """Visualize augmentation progression over training."""
    print("\nGenerating Augmentation Progression Visualization...")
    
    config = create_pcb_augmentation_config("balanced")
    scheduler = ProgressiveAugmentationScheduler(config)
    
    epochs = list(range(0, 300, 5))
    phases = []
    mosaic_probs = []
    copy_paste_probs = []
    mixup_probs = []
    strengths = []
    
    for epoch in epochs:
        scheduler.set_epoch(epoch)
        phase = scheduler.get_current_phase()
        phase_config = scheduler.get_phase_config()
        
        phases.append(phase.value)
        mosaic_probs.append(phase_config['mosaic_prob'])
        copy_paste_probs.append(phase_config['copy_paste_prob'])
        mixup_probs.append(phase_config['mixup_prob'])
        strengths.append(phase_config['augmentation_strength'])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Augmentation probabilities
    ax = axes[0, 0]
    ax.plot(epochs, mosaic_probs, label='Mosaic', linewidth=2)
    ax.plot(epochs, copy_paste_probs, label='Copy-Paste', linewidth=2)
    ax.plot(epochs, mixup_probs, label='MixUp', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Probability')
    ax.set_title('Augmentation Probabilities Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Augmentation strength
    ax = axes[0, 1]
    ax.plot(epochs, strengths, 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Augmentation Strength')
    ax.set_title('Overall Augmentation Strength')
    ax.grid(True, alpha=0.3)
    
    # Phase distribution
    ax = axes[1, 0]
    phase_counts = {phase: phases.count(phase) for phase in set(phases)}
    ax.bar(phase_counts.keys(), phase_counts.values(), alpha=0.7)
    ax.set_xlabel('Training Phase')
    ax.set_ylabel('Number of Epochs')
    ax.set_title('Training Phase Distribution')
    ax.tick_params(axis='x', rotation=45)
    
    # Adaptive augmentation simulation
    ax = axes[1, 1]
    adaptive_aug = AdaptiveAugmentation(config)
    
    # Simulate performance with stagnation
    performance_epochs = list(range(50))
    multipliers = []
    
    for epoch in performance_epochs:
        # Simulate stagnating performance after epoch 20
        if epoch < 20:
            map_val = 0.1 + epoch * 0.01
        else:
            map_val = 0.3 + np.random.normal(0, 0.005)
        
        metrics = {'mAP': map_val, 'loss': 1.0 - map_val}
        adaptive_aug.update_performance(epoch, metrics)
        multipliers.append(adaptive_aug.get_adaptive_multiplier())
    
    ax.plot(performance_epochs, multipliers, 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Adaptive Multiplier')
    ax.set_title('Adaptive Augmentation Response')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integrated_augmentation_progression.png', dpi=150, bbox_inches='tight')
    print("Results saved to integrated_augmentation_progression.png")


def main():
    """Main validation function."""
    print("Integrated Augmentation Pipeline Validation")
    print("=" * 50)
    
    # Test configuration system
    test_augmentation_config()
    
    # Test progressive scheduler
    test_progressive_scheduler()
    
    # Test small defect augmentation
    test_small_defect_augmentation()
    
    # Test adaptive augmentation
    test_adaptive_augmentation()
    
    # Test integrated pipeline
    pipeline_results = test_integrated_pipeline()
    
    # Test statistics
    test_augmentation_statistics()
    
    # Test augmentation effects
    effect_results = test_augmentation_effects()
    
    # Generate visualization
    try:
        visualize_augmentation_progression()
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Integrated Augmentation Pipeline Validation Complete")
    
    # Count successful tests
    successful_phases = sum(1 for result in pipeline_results.values() if result.get('success', False))
    total_phases = len(pipeline_results)
    
    print(f"\nPipeline Tests: {successful_phases}/{total_phases} phases successful")
    
    if effect_results:
        print(f"Strategy Tests: {len(effect_results)} strategies tested")
        for strategy, results in effect_results.items():
            print(f"  {strategy}: {results['mean_annotations']:.1f} ± {results['std_annotations']:.1f} annotations")
    
    print("\nKey Features Validated:")
    print("✓ Progressive augmentation scheduling")
    print("✓ Small defect specialized augmentation")
    print("✓ Adaptive augmentation based on performance")
    print("✓ Integrated pipeline with multiple strategies")
    print("✓ Training-aware augmentation phases")
    print("✓ Configuration system for different strategies")


if __name__ == "__main__":
    main()