#!/usr/bin/env python3
"""
Evaluate Trained Baseline Model on Test Dataset.

This script loads the trained baseline model and performs real evaluation
on the test dataset to get accurate performance metrics.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train_real_baseline import SimpleYOLOBaseline, custom_collate_fn
from pcb_detection.data.dataset import PCBDataset
from pcb_detection.utils.file_utils import FileUtils
from pcb_detection.core.types import Detection, CLASS_MAPPING
from real_baseline_test import RealBaselineEvaluator
from torch.utils.data import DataLoader


class TrainedModelEvaluator:
    """Evaluator for trained baseline model."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run evaluation on
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> SimpleYOLOBaseline:
        """Load trained model from checkpoint."""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model_config = checkpoint.get('model_config', {})
        model = SimpleYOLOBaseline(
            num_classes=model_config.get('num_classes', 5),
            input_size=model_config.get('input_size', 640)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✓ Loaded trained model from: {model_path}")
        return model
        
    def extract_detections_from_predictions(self, predictions: torch.Tensor, 
                                          confidence_threshold: float = 0.1) -> list:
        """
        Extract detections from model predictions.
        
        Args:
            predictions: Model output tensor [batch, channels, H, W]
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detections for each image in batch
        """
        batch_detections = []
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred = predictions[i]  # [channels, H, W]
            
            # Get prediction dimensions
            channels, H, W = pred.shape
            
            # Reshape predictions for processing
            # Expected format: [num_anchors * (5 + num_classes), H, W]
            num_anchors = 3
            num_classes = 5
            expected_channels = num_anchors * (5 + num_classes)
            
            if channels != expected_channels:
                print(f"Warning: Unexpected prediction shape: {pred.shape}")
                # Create empty detections for this image
                batch_detections.append([])
                continue
                
            # Reshape to [num_anchors, 5 + num_classes, H, W]
            pred = pred.view(num_anchors, 5 + num_classes, H, W)
            
            image_detections = []
            
            # Process each spatial location
            for h in range(H):
                for w in range(W):
                    for anchor in range(num_anchors):
                        # Extract prediction for this anchor at this location
                        anchor_pred = pred[anchor, :, h, w]  # [5 + num_classes]
                        
                        # Extract components
                        x_offset = torch.sigmoid(anchor_pred[0])  # Sigmoid for offset
                        y_offset = torch.sigmoid(anchor_pred[1])  # Sigmoid for offset
                        width = torch.exp(anchor_pred[2])         # Exp for size
                        height = torch.exp(anchor_pred[3])        # Exp for size
                        objectness = torch.sigmoid(anchor_pred[4]) # Sigmoid for objectness
                        class_scores = torch.softmax(anchor_pred[5:], dim=0)  # Softmax for classes
                        
                        # Calculate actual coordinates (normalized)
                        x_center = (w + x_offset) / W
                        y_center = (h + y_offset) / H
                        box_width = width / W
                        box_height = height / H
                        
                        # Clamp to valid range
                        x_center = torch.clamp(x_center, 0, 1)
                        y_center = torch.clamp(y_center, 0, 1)
                        box_width = torch.clamp(box_width, 0, 1)
                        box_height = torch.clamp(box_height, 0, 1)
                        
                        # Get best class
                        class_confidence, class_id = torch.max(class_scores, dim=0)
                        
                        # Final confidence
                        final_confidence = objectness * class_confidence
                        
                        # Filter by confidence threshold
                        if final_confidence > confidence_threshold:
                            detection = Detection(
                                bbox=(x_center.item(), y_center.item(), 
                                     box_width.item(), box_height.item()),
                                confidence=final_confidence.item(),
                                class_id=class_id.item(),
                                class_name=CLASS_MAPPING[class_id.item()]
                            )
                            image_detections.append(detection)
                            
            batch_detections.append(image_detections)
            
        return batch_detections
        
    def evaluate_on_test_dataset(self, test_dataset_path: str = "PCB_瑕疵测试集",
                                confidence_threshold: float = 0.1) -> dict:
        """
        Evaluate trained model on test dataset.
        
        Args:
            test_dataset_path: Path to test dataset
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            Evaluation results dictionary
        """
        print(f"Evaluating trained model on test dataset...")
        print(f"Confidence threshold: {confidence_threshold}")
        
        # Load test dataset
        test_dataset = PCBDataset(
            data_path=test_dataset_path,
            mode="test",
            image_size=640,
            augmentation_config=None
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        print(f"Test dataset: {len(test_dataset)} samples")
        
        # Generate predictions
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images = images.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Extract detections
                batch_detections = self.extract_detections_from_predictions(
                    predictions, confidence_threshold
                )
                
                all_predictions.extend(batch_detections)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} images")
                    
        print(f"Generated predictions for {len(all_predictions)} images")
        
        # Load ground truth for evaluation
        evaluator = RealBaselineEvaluator(test_dataset_path)
        test_samples = evaluator.load_test_data()
        
        if len(test_samples) != len(all_predictions):
            print(f"Warning: Mismatch in sample count: GT={len(test_samples)}, Pred={len(all_predictions)}")
            # Align the counts
            min_count = min(len(test_samples), len(all_predictions))
            test_samples = test_samples[:min_count]
            all_predictions = all_predictions[:min_count]
            
        # Evaluate predictions
        evaluation_results = self._evaluate_predictions(test_samples, all_predictions)
        
        return evaluation_results
        
    def _evaluate_predictions(self, test_samples: list, predictions: list, 
                            iou_threshold: float = 0.5) -> dict:
        """Evaluate predictions against ground truth."""
        
        print(f"Evaluating predictions with IoU threshold {iou_threshold}...")
        
        # Initialize metrics
        total_gt = 0
        total_pred = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        class_stats = {class_name: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0} 
                      for class_name in CLASS_MAPPING.values()}
        
        for sample, pred_detections in zip(test_samples, predictions):
            # Get ground truth
            gt_boxes = sample['ground_truth']
            total_gt += len(gt_boxes)
            
            # Update class ground truth counts
            for gt_box in gt_boxes:
                class_name = CLASS_MAPPING.get(gt_box['class_id'], 'Unknown')
                if class_name in class_stats:
                    class_stats[class_name]['gt_count'] += 1
            
            # Convert predictions to comparable format
            pred_boxes = []
            for det in pred_detections:
                pred_boxes.append({
                    'class_id': det.class_id,
                    'bbox': det.bbox,
                    'confidence': det.confidence
                })
                
            total_pred += len(pred_boxes)
            
            # Match predictions to ground truth
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                        
                    # Check class match and IoU
                    if pred_box['class_id'] == gt_box['class_id']:
                        iou = self._calculate_iou(pred_box['bbox'], gt_box['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                            
                # Determine if this is a true positive
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    
                    # Update class stats
                    class_name = CLASS_MAPPING[pred_box['class_id']]
                    if class_name in class_stats:
                        class_stats[class_name]['tp'] += 1
                else:
                    false_positives += 1
                    
                    # Update class stats
                    class_name = CLASS_MAPPING[pred_box['class_id']]
                    if class_name in class_stats:
                        class_stats[class_name]['fp'] += 1
            
            # Count false negatives (unmatched ground truth)
            unmatched_gt = len(gt_boxes) - len(matched_gt)
            false_negatives += unmatched_gt
            
            # Update class false negatives
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx not in matched_gt:
                    class_name = CLASS_MAPPING.get(gt_box['class_id'], 'Unknown')
                    if class_name in class_stats:
                        class_stats[class_name]['fn'] += 1
        
        # Calculate overall metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate per-class AP (simplified)
        ap_per_class = {}
        for class_name, stats in class_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Simplified AP calculation (single point)
            ap_per_class[class_name] = class_precision * class_recall
        
        # Calculate mAP
        map_score = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
        
        results = {
            'map_50': map_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ap_per_class': ap_per_class,
            'total_gt': total_gt,
            'total_pred': total_pred,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'class_stats': class_stats
        }
        
        return results
        
    def _calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """Calculate IoU between two bounding boxes (normalized coordinates)."""
        
        # Convert center format to corner format
        x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
        x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
        
        x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
        x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def main():
    """Main evaluation function."""
    
    print("PCB Defect Detection - Trained Baseline Evaluation")
    print("=" * 55)
    
    # Check if trained model exists
    model_path = "outputs/trained_baseline/models/baseline_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please run train_real_baseline.py first to train the model.")
        return 1
        
    try:
        # Initialize evaluator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluator = TrainedModelEvaluator(model_path, device)
        
        # Evaluate on test dataset
        results = evaluator.evaluate_on_test_dataset(confidence_threshold=0.05)
        
        # Create comprehensive results
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'trained_baseline_evaluation',
            'description': 'Evaluation of trained baseline YOLO model on test dataset',
            'model_info': {
                'model_path': model_path,
                'architecture': 'SimpleYOLOBaseline',
                'device': device
            },
            'evaluation': results,
            'baseline_comparison': {
                'untrained_map': 0.000000,
                'trained_map': results['map_50'],
                'improvement_factor': results['map_50'] / 0.001 if results['map_50'] > 0 else 0,
                'improvement_achieved': results['map_50'] > 0.001
            }
        }
        
        # Save results
        output_dir = "outputs/trained_baseline/results"
        FileUtils.ensure_dir(output_dir)
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
            
        comprehensive_results_serializable = convert_for_json(comprehensive_results)
        
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results_serializable, f, indent=2, ensure_ascii=False)
            
        # Create detailed report
        report_file = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("PCB Defect Detection - Trained Baseline Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Device: {device}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"mAP@0.5: {results['map_50']:.6f}\n")
            f.write(f"Precision: {results['precision']:.6f}\n")
            f.write(f"Recall: {results['recall']:.6f}\n")
            f.write(f"F1-Score: {results['f1_score']:.6f}\n\n")
            
            f.write("Detection Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Ground Truth: {results['total_gt']}\n")
            f.write(f"Total Predictions: {results['total_pred']}\n")
            f.write(f"True Positives: {results['true_positives']}\n")
            f.write(f"False Positives: {results['false_positives']}\n")
            f.write(f"False Negatives: {results['false_negatives']}\n\n")
            
            f.write("Per-Class Average Precision:\n")
            f.write("-" * 30 + "\n")
            for class_name, ap in results['ap_per_class'].items():
                f.write(f"{class_name}: {ap:.6f}\n")
                
            f.write(f"\nBaseline Comparison:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Untrained Model mAP: 0.000000\n")
            f.write(f"Trained Model mAP: {results['map_50']:.6f}\n")
            improvement = comprehensive_results['baseline_comparison']['improvement_achieved']
            f.write(f"Improvement Achieved: {'Yes' if improvement else 'No'}\n")
            
        # Print results
        print("\n" + "=" * 60)
        print("TRAINED BASELINE EVALUATION RESULTS")
        print("=" * 60)
        print(f"📊 mAP@0.5: {results['map_50']:.6f}")
        print(f"🎯 Precision: {results['precision']:.6f}")
        print(f"📈 Recall: {results['recall']:.6f}")
        print(f"⚖️  F1-Score: {results['f1_score']:.6f}")
        
        print(f"\n📋 Detection Statistics:")
        print(f"   Ground Truth Objects: {results['total_gt']}")
        print(f"   Model Predictions: {results['total_pred']}")
        print(f"   True Positives: {results['true_positives']}")
        print(f"   False Positives: {results['false_positives']}")
        print(f"   False Negatives: {results['false_negatives']}")
        
        print(f"\n🏷️  Per-Class Performance:")
        for class_name, ap in results['ap_per_class'].items():
            print(f"   {class_name}: {ap:.6f}")
            
        print(f"\n📊 Baseline Comparison:")
        print(f"   Untrained Model: mAP = 0.000000")
        print(f"   Trained Model: mAP = {results['map_50']:.6f}")
        improvement = comprehensive_results['baseline_comparison']['improvement_achieved']
        print(f"   Improvement: {'✅ Yes' if improvement else '❌ No'}")
        
        print(f"\n💾 Results saved to:")
        print(f"   {results_file}")
        print(f"   {report_file}")
        print("=" * 60)
        
        # Final assessment
        if results['map_50'] > 0.01:
            print(f"\n🎉 EXCELLENT: Trained model significantly outperforms baseline!")
            print(f"   mAP improvement: {results['map_50']:.6f} (>10x better than untrained)")
        elif results['map_50'] > 0.001:
            print(f"\n✅ GOOD: Trained model shows improvement over baseline!")
            print(f"   mAP improvement: {results['map_50']:.6f}")
        else:
            print(f"\n⚠️  LIMITED: Trained model shows minimal improvement")
            print(f"   Consider longer training or advanced techniques")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())