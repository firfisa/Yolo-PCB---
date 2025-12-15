#!/usr/bin/env python3
"""
Baseline YOLO Model Training Script for PCB Defect Detection.

This script trains a baseline YOLO model using minimal configuration
to establish performance baseline (expected mAP: 0.005-0.01).
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcb_detection.utils.config_interface import ConfigInterface, get_recommended_config
from pcb_detection.training.trainer import PCBTrainer
from pcb_detection.data.dataset import PCBDataset
from pcb_detection.evaluation.evaluator import Evaluator
from pcb_detection.utils.file_utils import ensure_dir


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging for baseline training."""
    ensure_dir(log_dir)
    
    # Create logger
    logger = logging.getLogger('baseline_training')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f'baseline_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def validate_data_paths(config: dict, logger: logging.Logger) -> bool:
    """Validate that data paths exist."""
    train_path = config['data']['train_path']
    test_path = config['data']['test_path']
    
    if not os.path.exists(train_path):
        logger.error(f"Training data path not found: {train_path}")
        return False
        
    if not os.path.exists(test_path):
        logger.error(f"Test data path not found: {test_path}")
        return False
        
    logger.info(f"Training data path: {train_path}")
    logger.info(f"Test data path: {test_path}")
    
    return True


def create_baseline_directories(base_dir: str) -> dict:
    """Create directories for baseline training outputs."""
    dirs = {
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs'),
        'results': os.path.join(base_dir, 'results'),
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'configs': os.path.join(base_dir, 'configs')
    }
    
    for dir_path in dirs.values():
        ensure_dir(dir_path)
        
    return dirs


def save_baseline_config(config: dict, config_path: str, logger: logging.Logger):
    """Save the baseline configuration for reproducibility."""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Baseline configuration saved to: {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")


def train_baseline_model(config: dict, output_dirs: dict, logger: logging.Logger) -> dict:
    """Train the baseline YOLO model."""
    
    logger.info("Starting baseline model training...")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    
    try:
        # Initialize trainer
        trainer = PCBTrainer(config)
        
        # Create datasets
        train_dataset = PCBDataset(
            data_path=config['data']['train_path'],
            mode='train',
            augmentation_config=config['augmentation']
        )
        
        val_dataset = PCBDataset(
            data_path=config['data']['train_path'],
            mode='val',
            augmentation_config=None  # No augmentation for validation
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Train the model
        training_results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=output_dirs['models']
        )
        
        logger.info("Baseline training completed successfully")
        return training_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def evaluate_baseline_model(config: dict, model_path: str, output_dirs: dict, 
                          logger: logging.Logger) -> dict:
    """Evaluate the baseline model on test data."""
    
    logger.info("Starting baseline model evaluation...")
    
    try:
        # Initialize evaluator
        evaluator = Evaluator(
            iou_threshold=config['evaluation']['iou_threshold'],
            confidence_threshold=config['inference']['confidence_threshold']
        )
        
        # Create test dataset
        test_dataset = PCBDataset(
            data_path=config['data']['test_path'],
            mode='test',
            augmentation_config=None
        )
        
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Load trained model
        from pcb_detection.models.yolo_detector import YOLODetector
        model = YOLODetector(config['model'])
        model.load_weights(model_path)
        
        # Run evaluation
        evaluation_results = evaluator.evaluate_model(
            model=model,
            test_dataset=test_dataset,
            save_results=True,
            results_dir=output_dirs['results']
        )
        
        # Log key metrics
        map_score = evaluation_results.get('map_50', 0.0)
        logger.info(f"Baseline mAP@0.5: {map_score:.6f}")
        
        # Check if mAP is in expected baseline range
        if 0.005 <= map_score <= 0.01:
            logger.info("✓ Baseline mAP is within expected range (0.005-0.01)")
        elif map_score < 0.005:
            logger.warning(f"⚠ Baseline mAP ({map_score:.6f}) is below expected range")
        else:
            logger.info(f"✓ Baseline mAP ({map_score:.6f}) exceeds expected range (good!)")
            
        # Log per-class AP
        if 'ap_per_class' in evaluation_results:
            logger.info("Per-class Average Precision:")
            for class_name, ap in evaluation_results['ap_per_class'].items():
                logger.info(f"  {class_name}: {ap:.6f}")
                
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def save_baseline_results(training_results: dict, evaluation_results: dict,
                         output_dirs: dict, logger: logging.Logger):
    """Save baseline training and evaluation results."""
    
    # Combine results
    baseline_results = {
        'timestamp': datetime.now().isoformat(),
        'training': training_results,
        'evaluation': evaluation_results,
        'baseline_verification': {
            'map_50': evaluation_results.get('map_50', 0.0),
            'expected_range': [0.005, 0.01],
            'within_range': 0.005 <= evaluation_results.get('map_50', 0.0) <= 0.01
        }
    }
    
    # Save to JSON
    results_file = os.path.join(output_dirs['results'], 'baseline_results.json')
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Baseline results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        
    # Save summary
    summary_file = os.path.join(output_dirs['results'], 'baseline_summary.txt')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PCB Defect Detection - Baseline Model Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {training_results.get('model_name', 'yolov8n')}\n")
            f.write(f"Training Epochs: {training_results.get('epochs_trained', 'N/A')}\n")
            f.write(f"Best Validation Loss: {training_results.get('best_val_loss', 'N/A')}\n\n")
            
            f.write("Evaluation Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"mAP@0.5: {evaluation_results.get('map_50', 0.0):.6f}\n")
            f.write(f"Expected Range: 0.005 - 0.01\n")
            f.write(f"Within Range: {'Yes' if baseline_results['baseline_verification']['within_range'] else 'No'}\n\n")
            
            if 'ap_per_class' in evaluation_results:
                f.write("Per-Class Average Precision:\n")
                f.write("-" * 30 + "\n")
                for class_name, ap in evaluation_results['ap_per_class'].items():
                    f.write(f"{class_name}: {ap:.6f}\n")
                    
        logger.info(f"Baseline summary saved to: {summary_file}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")


def main():
    """Main function for baseline training."""
    parser = argparse.ArgumentParser(description='Train baseline PCB defect detection model')
    parser.add_argument('--output-dir', type=str, default='outputs/baseline',
                       help='Output directory for baseline results')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Custom configuration file (optional)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dirs = create_baseline_directories(args.output_dir)
    
    # Setup logging
    logger = setup_logging(output_dirs['logs'])
    logger.info("Starting PCB Defect Detection Baseline Training")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Get baseline configuration
        if args.config_file and os.path.exists(args.config_file):
            logger.info(f"Loading configuration from: {args.config_file}")
            with open(args.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            logger.info("Using default baseline configuration")
            config = get_recommended_config("baseline")
            
        # Override with command line arguments
        if args.epochs != 50:
            config['training']['epochs'] = args.epochs
        if args.batch_size != 16:
            config['training']['batch_size'] = args.batch_size
        if args.device != 'auto':
            config['training']['device'] = args.device
            
        # Validate data paths
        if not validate_data_paths(config, logger):
            logger.error("Data validation failed. Please check data paths.")
            return 1
            
        # Save configuration
        config_file = os.path.join(output_dirs['configs'], 'baseline_config.yaml')
        save_baseline_config(config, config_file, logger)
        
        # Train baseline model
        training_results = train_baseline_model(config, output_dirs, logger)
        
        # Find best model path
        best_model_path = os.path.join(output_dirs['models'], 'best_model.pt')
        if not os.path.exists(best_model_path):
            # Try alternative paths
            model_files = [f for f in os.listdir(output_dirs['models']) if f.endswith('.pt')]
            if model_files:
                best_model_path = os.path.join(output_dirs['models'], model_files[0])
            else:
                logger.error("No trained model found")
                return 1
                
        # Evaluate baseline model
        evaluation_results = evaluate_baseline_model(
            config, best_model_path, output_dirs, logger
        )
        
        # Save results
        save_baseline_results(training_results, evaluation_results, output_dirs, logger)
        
        # Final summary
        map_score = evaluation_results.get('map_50', 0.0)
        logger.info("=" * 60)
        logger.info("BASELINE TRAINING COMPLETED")
        logger.info(f"Final mAP@0.5: {map_score:.6f}")
        logger.info(f"Expected range: 0.005 - 0.01")
        logger.info(f"Status: {'✓ PASS' if 0.005 <= map_score <= 0.01 else '⚠ OUT OF RANGE'}")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Baseline training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())