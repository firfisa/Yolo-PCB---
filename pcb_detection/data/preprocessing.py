"""
Dataset splitting and preprocessing utilities.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import random
from collections import defaultdict, Counter
from pathlib import Path
import json
import cv2

from ..core.types import CLASS_MAPPING


class DatasetSplitter:
    """Utility class for splitting datasets while maintaining class balance."""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                 test_ratio: float = 0.15, random_seed: int = 42):
        """
        Initialize dataset splitter.
        
        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            random_seed: Random seed for reproducibility
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")
            
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
    def split_dataset(self, annotations: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset while maintaining class balance.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Tuple of (train_annotations, val_annotations, test_annotations)
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Group annotations by class distribution
        class_groups = self._group_by_class_distribution(annotations)
        
        train_annotations = []
        val_annotations = []
        test_annotations = []
        
        # Split each group proportionally
        for group in class_groups:
            group_size = len(group)
            
            # Calculate split sizes
            train_size = int(group_size * self.train_ratio)
            val_size = int(group_size * self.val_ratio)
            test_size = group_size - train_size - val_size
            
            # Shuffle group
            shuffled_group = group.copy()
            random.shuffle(shuffled_group)
            
            # Split
            train_annotations.extend(shuffled_group[:train_size])
            val_annotations.extend(shuffled_group[train_size:train_size + val_size])
            test_annotations.extend(shuffled_group[train_size + val_size:])
            
        # Final shuffle
        random.shuffle(train_annotations)
        random.shuffle(val_annotations)
        random.shuffle(test_annotations)
        
        return train_annotations, val_annotations, test_annotations
        
    def _group_by_class_distribution(self, annotations: List[Dict]) -> List[List[Dict]]:
        """Group annotations by their class distribution patterns."""
        # Create signature for each annotation based on classes present
        signature_groups = defaultdict(list)
        
        for annotation in annotations:
            # Create a signature based on classes present in this image
            classes_present = set()
            for obj in annotation['objects']:
                classes_present.add(obj['class_id'])
                
            # Convert to sorted tuple for consistent hashing
            signature = tuple(sorted(classes_present))
            signature_groups[signature].append(annotation)
            
        return list(signature_groups.values())
        
    def get_split_statistics(self, train_annotations: List[Dict], 
                           val_annotations: List[Dict], 
                           test_annotations: List[Dict]) -> Dict:
        """Get statistics about the dataset split."""
        def get_class_counts(annotations):
            counts = Counter()
            for annotation in annotations:
                for obj in annotation['objects']:
                    class_id = obj['class_id']
                    if class_id in CLASS_MAPPING:
                        counts[CLASS_MAPPING[class_id]] += 1
            return dict(counts)
            
        train_counts = get_class_counts(train_annotations)
        val_counts = get_class_counts(val_annotations)
        test_counts = get_class_counts(test_annotations)
        
        # Calculate total counts
        total_counts = Counter()
        for counts in [train_counts, val_counts, test_counts]:
            total_counts.update(counts)
            
        # Calculate ratios
        train_ratios = {k: v / total_counts[k] if total_counts[k] > 0 else 0 
                       for k, v in train_counts.items()}
        val_ratios = {k: v / total_counts[k] if total_counts[k] > 0 else 0 
                     for k, v in val_counts.items()}
        test_ratios = {k: v / total_counts[k] if total_counts[k] > 0 else 0 
                      for k, v in test_counts.items()}
        
        return {
            'train': {
                'count': len(train_annotations),
                'class_counts': train_counts,
                'class_ratios': train_ratios
            },
            'val': {
                'count': len(val_annotations),
                'class_counts': val_counts,
                'class_ratios': val_ratios
            },
            'test': {
                'count': len(test_annotations),
                'class_counts': test_counts,
                'class_ratios': test_ratios
            },
            'total': {
                'count': len(train_annotations) + len(val_annotations) + len(test_annotations),
                'class_counts': dict(total_counts)
            }
        }


class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    def __init__(self, target_size: int = 640, normalize: bool = True):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size for resizing
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (preprocessed_image, preprocessing_info)
        """
        original_shape = image.shape[:2]
        
        # Resize while maintaining aspect ratio
        resized_image, scale_factor, padding = self._resize_with_padding(image)
        
        # Normalize if requested
        if self.normalize:
            resized_image = resized_image.astype(np.float32) / 255.0
            
        preprocessing_info = {
            'original_shape': original_shape,
            'target_size': self.target_size,
            'scale_factor': scale_factor,
            'padding': padding,
            'normalized': self.normalize
        }
        
        return resized_image, preprocessing_info
        
    def _resize_with_padding(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image with padding to maintain aspect ratio."""
        h, w = image.shape[:2]
        
        # Calculate scale factor
        scale = min(self.target_size / w, self.target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.full((self.target_size, self.target_size, image.shape[2]), 
                           114, dtype=image.dtype)
        else:
            padded = np.full((self.target_size, self.target_size), 
                           114, dtype=image.dtype)
        
        # Calculate padding offsets
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        
        # Place resized image in center
        if len(image.shape) == 3:
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        else:
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            
        return padded, scale, (pad_x, pad_y)
        
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert normalized image back to [0, 255] range."""
        if self.normalize and image.dtype == np.float32:
            return (image * 255).astype(np.uint8)
        return image
        
    def adjust_bbox_for_preprocessing(self, bbox: List[float], 
                                    preprocessing_info: Dict) -> List[float]:
        """Adjust bounding box coordinates for preprocessing transformations."""
        scale_factor = preprocessing_info['scale_factor']
        pad_x, pad_y = preprocessing_info['padding']
        target_size = preprocessing_info['target_size']
        
        # Convert from normalized to absolute coordinates
        x_center, y_center, width, height = bbox
        
        # Apply scale and padding
        x_center = (x_center * preprocessing_info['original_shape'][1] * scale_factor + pad_x) / target_size
        y_center = (y_center * preprocessing_info['original_shape'][0] * scale_factor + pad_y) / target_size
        width = width * preprocessing_info['original_shape'][1] * scale_factor / target_size
        height = height * preprocessing_info['original_shape'][0] * scale_factor / target_size
        
        return [x_center, y_center, width, height]


def create_balanced_dataset_splits(dataset_path: str, output_dir: str, 
                                 train_ratio: float = 0.7, val_ratio: float = 0.15, 
                                 test_ratio: float = 0.15, random_seed: int = 42) -> Dict:
    """
    Create balanced dataset splits and save split information.
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save split information
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split statistics
    """
    from .dataset import PCBDataset
    
    # Load dataset
    dataset = PCBDataset(dataset_path, mode="train")
    annotations = dataset.annotations
    
    # Create splitter and split dataset
    splitter = DatasetSplitter(train_ratio, val_ratio, test_ratio, random_seed)
    train_annotations, val_annotations, test_annotations = splitter.split_dataset(annotations)
    
    # Get statistics
    statistics = splitter.get_split_statistics(train_annotations, val_annotations, test_annotations)
    
    # Save split information
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save annotations for each split
    splits = {
        'train': train_annotations,
        'val': val_annotations,
        'test': test_annotations
    }
    
    for split_name, split_annotations in splits.items():
        split_file = output_path / f"{split_name}_annotations.json"
        with open(split_file, 'w') as f:
            json.dump(split_annotations, f, indent=2, default=str)
            
    # Save statistics
    stats_file = output_path / "split_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
        
    print(f"Dataset split completed:")
    print(f"  Train: {len(train_annotations)} samples")
    print(f"  Val: {len(val_annotations)} samples")
    print(f"  Test: {len(test_annotations)} samples")
    print(f"  Split information saved to: {output_dir}")
    
    return statistics