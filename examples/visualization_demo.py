#!/usr/bin/env python3
"""
Demonstration of PCB defect detection visualization capabilities.

This script shows how to use the visualization module to:
1. Draw detections on images
2. Create side-by-side comparison images
3. Generate batch visualization grids
4. Plot training curves and statistics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pcb_detection.visualization.visualizer import Visualizer
from pcb_detection.visualization.plotting import PlottingUtils
from pcb_detection.core.types import Detection, CLASS_MAPPING


def create_sample_image(width=640, height=480):
    """Create a sample PCB-like image for demonstration."""
    # Create a dark background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (30, 30, 30)  # Dark gray
    
    # Add some PCB-like patterns
    # Circuit traces (green lines)
    cv2.line(image, (50, 100), (590, 100), (0, 150, 0), 3)
    cv2.line(image, (50, 200), (590, 200), (0, 150, 0), 3)
    cv2.line(image, (50, 300), (590, 300), (0, 150, 0), 3)
    cv2.line(image, (100, 50), (100, 430), (0, 150, 0), 3)
    cv2.line(image, (300, 50), (300, 430), (0, 150, 0), 3)
    cv2.line(image, (500, 50), (500, 430), (0, 150, 0), 3)
    
    # Add some components (rectangles)
    cv2.rectangle(image, (150, 150), (250, 180), (100, 100, 100), -1)
    cv2.rectangle(image, (350, 250), (450, 280), (100, 100, 100), -1)
    cv2.rectangle(image, (200, 350), (300, 380), (100, 100, 100), -1)
    
    return image


def create_sample_detections():
    """Create sample detection results for demonstration."""
    detections = [
        Detection(
            bbox=(0.25, 0.3, 0.15, 0.1),  # Mouse bite
            confidence=0.87,
            class_id=0,
            class_name=CLASS_MAPPING[0]
        ),
        Detection(
            bbox=(0.6, 0.4, 0.12, 0.08),  # Open circuit
            confidence=0.92,
            class_id=1,
            class_name=CLASS_MAPPING[1]
        ),
        Detection(
            bbox=(0.4, 0.7, 0.08, 0.06),  # Short
            confidence=0.78,
            class_id=2,
            class_name=CLASS_MAPPING[2]
        ),
        Detection(
            bbox=(0.75, 0.25, 0.1, 0.12),  # Spur
            confidence=0.85,
            class_id=3,
            class_name=CLASS_MAPPING[3]
        ),
        Detection(
            bbox=(0.15, 0.8, 0.09, 0.07),  # Spurious copper
            confidence=0.73,
            class_id=4,
            class_name=CLASS_MAPPING[4]
        )
    ]
    return detections


def demo_basic_visualization():
    """Demonstrate basic detection visualization."""
    print("🎨 Demo 1: Basic Detection Visualization")
    
    # Create sample data
    image = create_sample_image()
    detections = create_sample_detections()
    
    # Create visualizer
    visualizer = Visualizer()
    
    # Draw detections
    vis_image = visualizer.draw_detections(image, detections)
    
    print(f"✓ Drew {len(detections)} detections on image")
    print("  Classes detected:", [d.class_name for d in detections])
    print("  Confidence scores:", [f"{d.confidence:.2f}" for d in detections])
    
    return vis_image


def demo_comparison_visualization():
    """Demonstrate side-by-side comparison visualization."""
    print("\n🔍 Demo 2: Ground Truth vs Predictions Comparison")
    
    # Create sample data
    image = create_sample_image()
    all_detections = create_sample_detections()
    
    # Split into GT and predictions (simulate some differences)
    gt_detections = all_detections[:3]  # First 3 as ground truth
    pred_detections = all_detections[1:4]  # Overlapping but different set
    
    # Add a false positive prediction
    pred_detections.append(Detection(
        bbox=(0.8, 0.6, 0.06, 0.05),
        confidence=0.65,
        class_id=1,
        class_name=CLASS_MAPPING[1]
    ))
    
    # Create visualizer and comparison
    visualizer = Visualizer()
    comparison_image = visualizer.create_comparison_image(
        image, gt_detections, pred_detections
    )
    
    print(f"✓ Created comparison with {len(gt_detections)} GT and {len(pred_detections)} predictions")
    print("  GT classes:", [d.class_name for d in gt_detections])
    print("  Pred classes:", [d.class_name for d in pred_detections])
    
    return comparison_image


def demo_batch_visualization():
    """Demonstrate batch grid visualization."""
    print("\n📊 Demo 3: Batch Grid Visualization")
    
    # Create multiple sample images with different detections
    images = []
    gt_detections_list = []
    pred_detections_list = []
    
    for i in range(4):
        # Create slightly different images
        img = create_sample_image()
        # Add some variation
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)
        
        # Create different detection sets for each image
        all_dets = create_sample_detections()
        gt_dets = all_dets[:2+i]  # Varying number of GT detections
        pred_dets = all_dets[1:3+i]  # Varying number of predictions
        
        gt_detections_list.append(gt_dets)
        pred_detections_list.append(pred_dets)
    
    # Create batch visualization
    visualizer = Visualizer()
    grid_image = visualizer.create_comparison_grid(
        images, gt_detections_list, pred_detections_list
    )
    
    print(f"✓ Created grid with {len(images)} image pairs")
    print("  Grid shape:", grid_image.shape)
    
    return grid_image


def demo_plotting_utilities():
    """Demonstrate plotting utilities."""
    print("\n📈 Demo 4: Plotting Utilities")
    
    # Create sample training data
    epochs = 50
    train_losses = [0.8 - 0.01*i + 0.05*np.sin(i/5) + np.random.normal(0, 0.02) for i in range(epochs)]
    val_losses = [0.9 - 0.008*i + 0.03*np.sin(i/7) + np.random.normal(0, 0.03) for i in range(epochs)]
    
    # Ensure losses are positive
    train_losses = [max(0.1, loss) for loss in train_losses]
    val_losses = [max(0.1, loss) for loss in val_losses]
    
    print("✓ Generated sample training curves")
    
    # Create class distribution data
    class_counts = {
        "Mouse_bite": 245,
        "Open_circuit": 189,
        "Short": 156,
        "Spur": 203,
        "Spurious_copper": 178
    }
    
    print("✓ Generated sample class distribution")
    
    # Create detection statistics
    detections_per_image = np.random.poisson(3, 100).tolist()  # Average 3 detections per image
    confidence_scores = np.random.beta(2, 1, 500).tolist()  # Skewed towards higher confidence
    
    print("✓ Generated sample detection statistics")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'class_counts': class_counts,
        'detections_per_image': detections_per_image,
        'confidence_scores': confidence_scores
    }


def main():
    """Run all visualization demos."""
    print("🚀 PCB Defect Detection Visualization Demo")
    print("=" * 50)
    
    # Demo 1: Basic visualization
    vis_image = demo_basic_visualization()
    
    # Demo 2: Comparison visualization
    comparison_image = demo_comparison_visualization()
    
    # Demo 3: Batch visualization
    grid_image = demo_batch_visualization()
    
    # Demo 4: Plotting utilities
    plot_data = demo_plotting_utilities()
    
    # Save all visualizations
    print("\n💾 Saving demonstration results...")
    
    import os
    os.makedirs("examples/demo_output", exist_ok=True)
    
    visualizer = Visualizer()
    visualizer.save_visualization(vis_image, "examples/demo_output/basic_detections.jpg")
    visualizer.save_visualization(comparison_image, "examples/demo_output/comparison.jpg")
    visualizer.save_visualization(grid_image, "examples/demo_output/batch_grid.jpg")
    
    print("✓ Saved visualization images to examples/demo_output/")
    
    # Demonstrate plotting (but don't show plots in demo)
    plt.ioff()  # Turn off interactive mode
    
    PlottingUtils.plot_training_curves(
        plot_data['train_losses'], 
        plot_data['val_losses'],
        "examples/demo_output/training_curves.png"
    )
    
    PlottingUtils.plot_class_distribution(
        plot_data['class_counts'],
        "examples/demo_output/class_distribution.png"
    )
    
    PlottingUtils.plot_detection_statistics(
        plot_data['detections_per_image'],
        plot_data['confidence_scores'],
        "examples/demo_output/detection_statistics.png"
    )
    
    print("✓ Saved plot images to examples/demo_output/")
    
    print("\n🎉 All visualization demos completed successfully!")
    print("\nGenerated files:")
    print("  - examples/demo_output/basic_detections.jpg")
    print("  - examples/demo_output/comparison.jpg") 
    print("  - examples/demo_output/batch_grid.jpg")
    print("  - examples/demo_output/training_curves.png")
    print("  - examples/demo_output/class_distribution.png")
    print("  - examples/demo_output/detection_statistics.png")


if __name__ == "__main__":
    main()