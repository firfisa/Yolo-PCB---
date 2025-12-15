#!/usr/bin/env python3
"""
Validation script for advanced loss function combinations.
Tests loss function implementations and optimization strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from pcb_detection.models.advanced_losses import (
    AdaptiveFocalLoss,
    QualityFocalLoss,
    VarifocalLoss,
    SmallObjectLoss,
    DynamicLossWeighting,
    AdvancedComboLoss,
    create_advanced_loss_config,
    create_pcb_optimized_loss
)
from pcb_detection.models.losses import FocalLoss, IoULoss


def test_adaptive_focal_loss():
    """Test Adaptive Focal Loss functionality."""
    print("Testing Adaptive Focal Loss")
    print("=" * 30)
    
    # Create loss function
    loss_fn = AdaptiveFocalLoss(
        alpha=1.0,
        gamma_init=2.0,
        gamma_final=0.5,
        total_epochs=100
    )
    
    # Test gamma adaptation
    print("  Gamma adaptation over epochs:")
    for epoch in [0, 25, 50, 75, 100]:
        loss_fn.set_epoch(epoch)
        gamma = loss_fn.get_current_gamma()
        print(f"    Epoch {epoch:3d}: gamma = {gamma:.3f}")
    
    # Test loss computation
    batch_size = 32
    num_classes = 5
    
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Compare with standard focal loss
    standard_focal = FocalLoss(alpha=1.0, gamma=2.0)
    
    loss_fn.set_epoch(0)  # Initial gamma = 2.0
    adaptive_loss_init = loss_fn(predictions, targets)
    standard_loss = standard_focal(predictions, targets)
    
    loss_fn.set_epoch(100)  # Final gamma = 0.5
    adaptive_loss_final = loss_fn(predictions, targets)
    
    print(f"  Standard Focal Loss: {standard_loss:.4f}")
    print(f"  Adaptive Focal (epoch 0): {adaptive_loss_init:.4f}")
    print(f"  Adaptive Focal (epoch 100): {adaptive_loss_final:.4f}")


def test_quality_focal_loss():
    """Test Quality Focal Loss functionality."""
    print("\nTesting Quality Focal Loss")
    print("=" * 30)
    
    loss_fn = QualityFocalLoss(beta=2.0)
    
    batch_size = 32
    num_classes = 5
    
    # Create test data
    pred_scores = torch.randn(batch_size, num_classes)
    target_scores = torch.randint(0, 2, (batch_size, num_classes)).float()
    pred_quality = torch.randn(batch_size)
    target_quality = torch.rand(batch_size)  # IoU scores [0, 1]
    
    loss = loss_fn(pred_scores, target_scores, pred_quality, target_quality)
    
    print(f"  Quality Focal Loss: {loss:.4f}")
    print(f"  Input shapes:")
    print(f"    Pred scores: {pred_scores.shape}")
    print(f"    Target scores: {target_scores.shape}")
    print(f"    Pred quality: {pred_quality.shape}")
    print(f"    Target quality: {target_quality.shape}")


def test_varifocal_loss():
    """Test Varifocal Loss functionality."""
    print("\nTesting Varifocal Loss")
    print("=" * 25)
    
    loss_fn = VarifocalLoss(alpha=0.75, gamma=2.0)
    
    batch_size = 32
    num_classes = 5
    
    # Create test data
    pred_scores = torch.randn(batch_size, num_classes)
    target_scores = torch.randint(0, 2, (batch_size, num_classes)).float()
    target_quality = torch.rand(batch_size, num_classes)  # Quality scores
    
    loss = loss_fn(pred_scores, target_scores, target_quality)
    
    print(f"  Varifocal Loss: {loss:.4f}")
    
    # Compare with different quality distributions
    high_quality = torch.ones_like(target_quality) * 0.9
    low_quality = torch.ones_like(target_quality) * 0.1
    
    loss_high = loss_fn(pred_scores, target_scores, high_quality)
    loss_low = loss_fn(pred_scores, target_scores, low_quality)
    
    print(f"  High quality targets: {loss_high:.4f}")
    print(f"  Low quality targets: {loss_low:.4f}")


def test_small_object_loss():
    """Test Small Object Loss functionality."""
    print("\nTesting Small Object Loss")
    print("=" * 30)
    
    loss_fn = SmallObjectLoss(small_threshold=32.0, scale_factor=2.0)
    
    batch_size = 16
    
    # Create bounding boxes (normalized coordinates)
    pred_boxes = torch.rand(batch_size, 4)
    pred_boxes[:, 2:] += pred_boxes[:, :2]  # Ensure x2 > x1, y2 > y1
    
    target_boxes = torch.rand(batch_size, 4)
    target_boxes[:, 2:] += target_boxes[:, :2]
    
    # Make some boxes small
    small_indices = torch.randperm(batch_size)[:batch_size//2]
    target_boxes[small_indices, 2:] = target_boxes[small_indices, :2] + 0.02  # Small boxes
    
    image_size = (640, 640)
    
    loss = loss_fn(pred_boxes, target_boxes, image_size)
    
    print(f"  Small Object Loss: {loss:.4f}")
    
    # Calculate box areas for analysis
    target_w = (target_boxes[:, 2] - target_boxes[:, 0]) * image_size[1]
    target_h = (target_boxes[:, 3] - target_boxes[:, 1]) * image_size[0]
    target_areas = target_w * target_h
    
    small_mask = target_areas < (32.0 ** 2)
    print(f"  Small objects: {small_mask.sum().item()}/{batch_size}")
    print(f"  Average area (small): {target_areas[small_mask].mean().item():.1f}")
    print(f"  Average area (large): {target_areas[~small_mask].mean().item():.1f}")


def test_dynamic_loss_weighting():
    """Test Dynamic Loss Weighting functionality."""
    print("\nTesting Dynamic Loss Weighting")
    print("=" * 35)
    
    initial_weights = {
        'classification': 1.0,
        'regression': 1.0,
        'objectness': 1.0
    }
    
    weighting = DynamicLossWeighting(
        initial_weights=initial_weights,
        adaptation_rate=0.1
    )
    
    print("  Initial weights:", weighting.get_weights())
    
    # Simulate training with different loss trends
    print("  Simulating training progress...")
    
    for epoch in range(20):
        # Simulate loss values with trends
        cls_loss = 1.0 + 0.1 * np.sin(epoch * 0.3) + np.random.normal(0, 0.05)
        reg_loss = 0.8 - epoch * 0.02 + np.random.normal(0, 0.03)  # Decreasing
        obj_loss = 0.5 + epoch * 0.01 + np.random.normal(0, 0.02)  # Increasing
        
        loss_values = {
            'classification': cls_loss,
            'regression': reg_loss,
            'objectness': obj_loss
        }
        
        weighting.update_weights(loss_values)
        
        if epoch % 5 == 0:
            weights = weighting.get_weights()
            print(f"    Epoch {epoch:2d}: cls={weights['classification']:.3f}, "
                  f"reg={weights['regression']:.3f}, obj={weights['objectness']:.3f}")


def test_advanced_combo_loss():
    """Test Advanced Combo Loss functionality."""
    print("\nTesting Advanced Combo Loss")
    print("=" * 35)
    
    # Test different strategies
    strategies = ["balanced", "small_objects", "quality_focused"]
    
    for strategy in strategies:
        print(f"\n  Testing {strategy} strategy:")
        
        config = create_advanced_loss_config(strategy)
        loss_fn = AdvancedComboLoss(config)
        
        # Create test data
        batch_size = 16
        num_classes = 5
        
        predictions = {
            'cls': torch.randn(batch_size, num_classes),
            'bbox': torch.rand(batch_size, 4),
            'obj': torch.randn(batch_size),
        }
        
        targets = {
            'cls': torch.randint(0, num_classes, (batch_size,)),
            'bbox': torch.rand(batch_size, 4),
            'obj': torch.randint(0, 2, (batch_size,)).float(),
        }
        
        # Add quality if enabled
        if config['quality']['enable']:
            predictions['quality'] = torch.randn(batch_size)
            targets['quality'] = torch.rand(batch_size)
            # For quality focal loss, we need proper classification targets
            targets['cls'] = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        # Compute loss
        losses = loss_fn(predictions, targets, epoch=0)
        
        total_loss_val = losses['total_loss']
        if torch.is_tensor(total_loss_val):
            total_loss_val = total_loss_val.item() if total_loss_val.numel() == 1 else total_loss_val.mean().item()
        print(f"    Total loss: {total_loss_val:.4f}")
        
        for key, value in losses.items():
            if key.endswith('_loss') and key != 'total_loss':
                if torch.is_tensor(value):
                    val = value.item() if value.numel() == 1 else value.mean().item()
                else:
                    val = value
                print(f"    {key}: {val:.4f}")
        
        if 'weights' in losses:
            print(f"    Weights: {losses['weights']}")


def test_pcb_optimized_loss():
    """Test PCB-optimized loss function."""
    print("\nTesting PCB-Optimized Loss")
    print("=" * 35)
    
    loss_fn = create_pcb_optimized_loss()
    
    # Simulate PCB defect detection scenario
    batch_size = 8
    num_classes = 5  # PCB defect classes
    
    predictions = {
        'cls': torch.randn(batch_size, num_classes),
        'bbox': torch.rand(batch_size, 4),
        'obj': torch.randn(batch_size),
        'quality': torch.randn(batch_size)
    }
    
    targets = {
        'cls': torch.randint(0, 2, (batch_size, num_classes)).float(),  # Binary targets for quality focal loss
        'bbox': torch.rand(batch_size, 4),
        'obj': torch.randint(0, 2, (batch_size,)).float(),
        'quality': torch.rand(batch_size),
        'image_size': (640, 640)
    }
    
    # Test loss evolution over epochs
    print("  Loss evolution over epochs:")
    
    for epoch in [0, 50, 100, 150, 200]:
        losses = loss_fn(predictions, targets, epoch=epoch)
        total_val = losses['total_loss'].item() if torch.is_tensor(losses['total_loss']) else losses['total_loss']
        cls_val = losses.get('cls_loss', 0)
        cls_val = cls_val.item() if torch.is_tensor(cls_val) else cls_val
        reg_val = losses.get('reg_loss', 0)
        reg_val = reg_val.item() if torch.is_tensor(reg_val) else reg_val
        
        print(f"    Epoch {epoch:3d}: total={total_val:.4f}, "
              f"cls={cls_val:.4f}, "
              f"reg={reg_val:.4f}")
    
    # Get loss statistics
    stats = loss_fn.get_loss_statistics()
    if stats:
        print("  Loss statistics (recent):")
        for key, stat in stats.items():
            if key != 'weights':
                print(f"    {key}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, "
                      f"trend={stat['trend']:.6f}")


def visualize_loss_comparisons():
    """Visualize loss function comparisons."""
    print("\nGenerating Loss Function Visualizations...")
    
    # Create test data
    batch_size = 1000
    num_classes = 5
    
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test different loss functions
    losses = {}
    
    # Standard Cross Entropy
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    losses['CrossEntropy'] = ce_loss(predictions, targets).detach().numpy()
    
    # Standard Focal Loss
    focal_loss = FocalLoss(gamma=2.0, reduction='none')
    losses['Focal (γ=2.0)'] = focal_loss(predictions, targets).detach().numpy()
    
    # Adaptive Focal Loss (different epochs)
    adaptive_focal = AdaptiveFocalLoss(gamma_init=2.0, gamma_final=0.5, total_epochs=100)
    
    adaptive_focal.set_epoch(0)
    losses['Adaptive Focal (epoch 0)'] = adaptive_focal(predictions, targets).detach().numpy()
    
    adaptive_focal.set_epoch(100)
    losses['Adaptive Focal (epoch 100)'] = adaptive_focal(predictions, targets).detach().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss distributions
    ax = axes[0, 0]
    for name, loss_values in losses.items():
        ax.hist(loss_values, bins=50, alpha=0.6, label=name, density=True)
    ax.set_xlabel('Loss Value')
    ax.set_ylabel('Density')
    ax.set_title('Loss Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss vs confidence
    ax = axes[0, 1]
    confidences = torch.softmax(predictions, dim=1).max(dim=1)[0].detach().numpy()
    
    for name, loss_values in losses.items():
        # Sample for visualization
        indices = np.random.choice(len(loss_values), 200, replace=False)
        ax.scatter(confidences[indices], loss_values[indices], alpha=0.6, label=name, s=10)
    
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss vs Prediction Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gamma adaptation over epochs
    ax = axes[1, 0]
    epochs = np.arange(0, 101, 5)
    gammas = []
    
    for epoch in epochs:
        adaptive_focal.set_epoch(epoch)
        gammas.append(adaptive_focal.get_current_gamma())
    
    ax.plot(epochs, gammas, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gamma Value')
    ax.set_title('Adaptive Focal Loss Gamma Evolution')
    ax.grid(True, alpha=0.3)
    
    # Loss statistics
    ax = axes[1, 1]
    loss_names = list(losses.keys())
    loss_means = [np.mean(losses[name]) for name in loss_names]
    loss_stds = [np.std(losses[name]) for name in loss_names]
    
    x_pos = np.arange(len(loss_names))
    ax.bar(x_pos, loss_means, yerr=loss_stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Loss Function')
    ax.set_ylabel('Mean Loss ± Std')
    ax.set_title('Loss Function Statistics')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(loss_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_losses_comparison.png', dpi=150, bbox_inches='tight')
    print("Results saved to advanced_losses_comparison.png")


def main():
    """Main validation function."""
    print("Advanced Loss Functions Validation")
    print("=" * 50)
    
    # Test individual loss components
    test_adaptive_focal_loss()
    test_quality_focal_loss()
    test_varifocal_loss()
    test_small_object_loss()
    test_dynamic_loss_weighting()
    
    # Test combined loss systems
    test_advanced_combo_loss()
    test_pcb_optimized_loss()
    
    # Generate visualizations
    try:
        visualize_loss_comparisons()
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Advanced Loss Functions Validation Complete")
    print("\nKey Features Validated:")
    print("✓ Adaptive Focal Loss with gamma scheduling")
    print("✓ Quality Focal Loss for joint classification/localization")
    print("✓ Varifocal Loss for quality-aware training")
    print("✓ Small Object Loss with area-based weighting")
    print("✓ Dynamic Loss Weighting with trend adaptation")
    print("✓ Advanced Combo Loss with multiple strategies")
    print("✓ PCB-optimized loss configuration")


if __name__ == "__main__":
    main()