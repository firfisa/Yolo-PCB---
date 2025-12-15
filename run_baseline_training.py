#!/usr/bin/env python3
"""
Simple Baseline Training Runner for PCB Defect Detection.

This script provides a simplified interface to train and evaluate
a baseline YOLO model for establishing performance benchmarks.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pcb_detection.utils.config_interface import get_recommended_config
from pcb_detection.utils.file_utils import FileUtils


def check_data_availability():
    """Check if training and test data are available."""
    train_path = "训练集-PCB_DATASET"
    test_path = "PCB_瑕疵测试集"
    
    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path)
    
    print("Data Availability Check:")
    print(f"  Training data ({train_path}): {'✓' if train_exists else '✗'}")
    print(f"  Test data ({test_path}): {'✓' if test_exists else '✗'}")
    
    if not train_exists:
        print(f"\n⚠ Warning: Training data not found at {train_path}")
        print("Please ensure the training dataset is available.")
        
    if not test_exists:
        print(f"\n⚠ Warning: Test data not found at {test_path}")
        print("Please ensure the test dataset is available.")
        
    return train_exists and test_exists


def create_baseline_config():
    """Create a minimal baseline configuration."""
    config = {
        'model': {
            'name': 'yolov8n',
            'backbone': 'yolov8n',
            'num_classes': 5,
            'input_size': 640,
            'pretrained': True,
            'attention': None,  # No attention for baseline
            'use_fpn': False
        },
        'training': {
            'epochs': 30,  # Reduced for faster baseline
            'batch_size': 16,
            'learning_rate': 0.01,
            'device': 'auto',
            'workers': 4,
            'patience': 10,
            'save_period': 5
        },
        'data': {
            'train_path': "训练集-PCB_DATASET",
            'test_path': "PCB_瑕疵测试集",
            'num_classes': 5,
            'train_split': 0.8,
            'val_split': 0.2
        },
        'augmentation': {
            'basic': {
                'rotation_range': [0, 0],  # No rotation
                'scale_range': [1.0, 1.0],  # No scaling
                'brightness_range': [0, 0],  # No brightness change
                'contrast_range': [1.0, 1.0],  # No contrast change
                'flip_horizontal': False,
                'flip_vertical': False,
                'prob': 0.0  # No augmentation for baseline
            },
            'advanced': {
                'mosaic_prob': 0.0,
                'copy_paste_prob': 0.0,
                'mixup_prob': 0.0,
                'use_albumentations': False
            }
        },
        'loss': {
            'classification': 'ce',  # Standard cross-entropy
            'bbox_regression': 'smooth_l1',  # Standard smooth L1
            'objectness': 'bce',
            'weights': {
                'cls': 1.0,
                'bbox': 1.0,
                'obj': 1.0
            }
        },
        'inference': {
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'use_tta': False
        },
        'evaluation': {
            'iou_threshold': 0.5,
            'save_results': True,
            'results_format': 'json'
        },
        'logging': {
            'level': 'INFO',
            'save_logs': True,
            'log_dir': 'outputs/baseline/logs'
        }
    }
    
    return config


def save_config(config, output_dir):
    """Save the baseline configuration."""
    config_dir = os.path.join(output_dir, 'configs')
    FileUtils.ensure_dir(config_dir)
    
    config_file = os.path.join(config_dir, 'baseline_config.yaml')
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
    print(f"Configuration saved to: {config_file}")
    return config_file


def run_baseline_training():
    """Run baseline training with minimal setup."""
    
    print("PCB Defect Detection - Baseline Training")
    print("=" * 50)
    
    # Check data availability
    if not check_data_availability():
        print("\n❌ Cannot proceed without required data.")
        print("Please ensure both training and test datasets are available.")
        return False
        
    # Create output directory
    output_dir = "outputs/baseline"
    FileUtils.ensure_dir(output_dir)
    FileUtils.ensure_dir(os.path.join(output_dir, 'models'))
    FileUtils.ensure_dir(os.path.join(output_dir, 'results'))
    FileUtils.ensure_dir(os.path.join(output_dir, 'logs'))
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create baseline configuration
    print("\nCreating baseline configuration...")
    config = create_baseline_config()
    config_file = save_config(config, output_dir)
    
    # Display configuration summary
    print("\nBaseline Configuration Summary:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Augmentation: Disabled (baseline)")
    print(f"  Attention: None (baseline)")
    
    # Try to import and run training
    try:
        print("\nStarting baseline training...")
        
        # Import training components
        from pcb_detection.training.trainer import Trainer
        from pcb_detection.data.dataset import PCBDataset
        from pcb_detection.models.yolo_wrapper import YOLOWrapper
        from pcb_detection.evaluation.evaluator import Evaluator
        from pcb_detection.core.types import TrainingConfig
        from torch.utils.data import DataLoader
        
        # Create datasets
        print("Loading datasets...")
        train_dataset = PCBDataset(
            data_path=config['data']['train_path'],
            mode='train',
            augmentation_config=config['augmentation']
        )
        
        val_dataset = PCBDataset(
            data_path=config['data']['train_path'],
            mode='val',
            augmentation_config=None
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['workers'],
            pin_memory=True
        )
        
        # Create model
        model = YOLOWrapper(config['model'])
        
        # Create training config
        training_config = TrainingConfig(
            model_name=config['model']['name'],
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            image_size=config['model']['input_size'],
            augmentation=True,
            patience=config['training']['patience'],
            save_period=config['training']['save_period']
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            checkpoint_dir=os.path.join(output_dir, 'models')
        )
        
        # Train the model
        print("\nTraining baseline model...")
        training_results = trainer.train()
        
        print("✓ Training completed successfully")
        
        # Evaluate the model
        print("\nEvaluating baseline model...")
        
        # Create test dataset
        test_dataset = PCBDataset(
            data_path=config['data']['test_path'],
            mode='test',
            augmentation_config=None
        )
        
        print(f"Test samples: {len(test_dataset)}")
        
        # Initialize evaluator
        evaluator = Evaluator(
            iou_threshold=config['evaluation']['iou_threshold'],
            confidence_threshold=config['inference']['confidence_threshold']
        )
        
        # Use the trained model from trainer
        model = trainer.model
        print("Using trained model for evaluation")
            
        # Run evaluation
        evaluation_results = evaluator.evaluate_model(
            model=model,
            test_dataset=test_dataset,
            save_results=True,
            results_dir=os.path.join(output_dir, 'results')
        )
        
        # Extract key metrics
        map_score = evaluation_results.get('map_50', 0.0)
        
        print("✓ Evaluation completed successfully")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'baseline_verification': {
                'map_50': map_score,
                'expected_range': [0.005, 0.01],
                'within_expected_range': 0.005 <= map_score <= 0.01,
                'status': 'PASS' if 0.005 <= map_score <= 0.01 else 'OUT_OF_RANGE'
            }
        }
        
        results_file = os.path.join(output_dir, 'results', 'baseline_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # Print final results
        print("\n" + "=" * 60)
        print("BASELINE TRAINING RESULTS")
        print("=" * 60)
        print(f"mAP@0.5: {map_score:.6f}")
        print(f"Expected Range: 0.005 - 0.01")
        print(f"Status: {results['baseline_verification']['status']}")
        
        if 'ap_per_class' in evaluation_results:
            print("\nPer-Class Average Precision:")
            for class_name, ap in evaluation_results['ap_per_class'].items():
                print(f"  {class_name}: {ap:.6f}")
                
        print(f"\nResults saved to: {results_file}")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please ensure all required dependencies are installed.")
        return False
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_baseline_training()
    if success:
        print("\n✓ Baseline training completed successfully!")
    else:
        print("\n❌ Baseline training failed!")
        sys.exit(1)