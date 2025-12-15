#!/usr/bin/env python3
"""
Demonstration script for advanced data augmentation techniques.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pcb_detection.data.advanced_augmentation import (
    MosaicAugmentation,
    CopyPasteAugmentation, 
    MixUpAugmentation,
    AlbumentationsAugmentation,
    PCBAdvancedAugmentation
)


def create_demo_data():
    """Create demo images and annotations for testing."""
    # Create synthetic PCB-like images
    images = []
    annotations_list = []
    
    for i in range(4):
        # Create a synthetic PCB image (green background with copper traces)
        image = np.full((640, 640, 3), [34, 139, 34], dtype=np.uint8)  # Green background
        
        # Add some copper traces (golden lines)
        for j in range(5):
            start_point = (np.random.randint(0, 640), np.random.randint(0, 640))
            end_point = (np.random.randint(0, 640), np.random.randint(0, 640))
            cv2.line(image, start_point, end_point, [255, 215, 0], thickness=3)
        
        # Add some defects (red rectangles)
        annotations = []
        for k in range(np.random.randint(1, 4)):
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.05, 0.15)
            height = np.random.uniform(0.05, 0.15)
            
            # Draw defect on image
            x1 = int((x_center - width/2) * 640)
            y1 = int((y_center - height/2) * 640)
            x2 = int((x_center + width/2) * 640)
            y2 = int((y_center + height/2) * 640)
            cv2.rectangle(image, (x1, y1), (x2, y2), [255, 0, 0], -1)
            
            annotations.append({
                'class_id': np.random.randint(0, 5),
                'bbox': [x_center, y_center, width, height]
            })
        
        images.append(image)
        annotations_list.append(annotations)
    
    return images, annotations_list


def demo_mosaic_augmentation():
    """Demonstrate Mosaic augmentation."""
    print("=== Mosaic Augmentation Demo ===")
    
    images, annotations_list = create_demo_data()
    
    mosaic = MosaicAugmentation(image_size=640, prob=1.0)  # Always apply
    mosaic_image, mosaic_annotations = mosaic(images, annotations_list)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show original images
    for i in range(4):
        row = i // 2
        col = i % 2
        if i < len(images):
            axes[row, col].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'Original Image {i+1}')
            axes[row, col].axis('off')
    
    # Show mosaic result
    axes[0, 2].imshow(cv2.cvtColor(mosaic_image, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Mosaic Result ({len(mosaic_annotations)} defects)')
    axes[0, 2].axis('off')
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('mosaic_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Mosaic created with {len(mosaic_annotations)} defects")


def demo_copy_paste_augmentation():
    """Demonstrate Copy-Paste augmentation."""
    print("\n=== Copy-Paste Augmentation Demo ===")
    
    images, annotations_list = create_demo_data()
    image = images[0]
    annotations = annotations_list[0]
    
    copy_paste = CopyPasteAugmentation(prob=1.0, max_paste=2)  # Always apply
    cp_image, cp_annotations = copy_paste(image, annotations)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original ({len(annotations)} defects)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(cp_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Copy-Paste Result ({len(cp_annotations)} defects)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('copy_paste_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Copy-Paste: {len(annotations)} -> {len(cp_annotations)} defects")


def demo_mixup_augmentation():
    """Demonstrate MixUp augmentation."""
    print("\n=== MixUp Augmentation Demo ===")
    
    images, annotations_list = create_demo_data()
    
    mixup = MixUpAugmentation(alpha=0.2, prob=1.0)  # Always apply
    mixed_image, mixed_annotations = mixup(
        images[0], annotations_list[0],
        images[1], annotations_list[1]
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Image 1 ({len(annotations_list[0])} defects)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Image 2 ({len(annotations_list[1])} defects)')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'MixUp Result ({len(mixed_annotations)} defects)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mixup_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"MixUp: {len(annotations_list[0])} + {len(annotations_list[1])} = {len(mixed_annotations)} defects")


def demo_albumentations_augmentation():
    """Demonstrate Albumentations augmentation."""
    print("\n=== Albumentations Augmentation Demo ===")
    
    images, annotations_list = create_demo_data()
    image = images[0]
    annotations = annotations_list[0]
    
    albu = AlbumentationsAugmentation(image_size=640, train=True)
    albu_image, albu_annotations = albu(image, annotations)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original ({len(annotations)} defects)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(albu_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Albumentations Result ({len(albu_annotations)} defects)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('albumentations_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Albumentations: {len(annotations)} -> {len(albu_annotations)} defects")


def demo_combined_pipeline():
    """Demonstrate combined advanced augmentation pipeline."""
    print("\n=== Combined Advanced Augmentation Pipeline Demo ===")
    
    images, annotations_list = create_demo_data()
    
    # Test different configurations
    configs = [
        ("Basic", {"mosaic_prob": 0.0, "copy_paste_prob": 0.0, "mixup_prob": 0.0, "use_albumentations": False}),
        ("Balanced", {"mosaic_prob": 0.3, "copy_paste_prob": 0.2, "mixup_prob": 0.1, "use_albumentations": True}),
        ("Performance", {"mosaic_prob": 0.5, "copy_paste_prob": 0.3, "mixup_prob": 0.2, "use_albumentations": True})
    ]
    
    fig, axes = plt.subplots(1, len(configs), figsize=(15, 5))
    
    for i, (config_name, config) in enumerate(configs):
        pipeline = PCBAdvancedAugmentation(
            image_size=640,
            mosaic_prob=config["mosaic_prob"],
            copy_paste_prob=config["copy_paste_prob"],
            mixup_prob=config["mixup_prob"],
            use_albumentations=config["use_albumentations"]
        )
        
        result_image, result_annotations = pipeline(images, annotations_list)
        
        axes[i].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'{config_name} Config\n({len(result_annotations)} defects)')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('combined_pipeline_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Combined pipeline demonstration completed")


def main():
    """Run all augmentation demonstrations."""
    print("PCB Advanced Data Augmentation Demonstration")
    print("=" * 50)
    
    try:
        demo_mosaic_augmentation()
        demo_copy_paste_augmentation()
        demo_mixup_augmentation()
        demo_albumentations_augmentation()
        demo_combined_pipeline()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("Check the generated PNG files for visual results.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()