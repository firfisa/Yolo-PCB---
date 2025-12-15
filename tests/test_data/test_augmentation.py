"""
Unit tests for data augmentation functionality.
"""

import unittest
import numpy as np

from pcb_detection.data.augmentation import DataAugmentation


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'rotation_range': (-15, 15),
            'scale_range': (0.8, 1.2),
            'brightness_range': (-0.2, 0.2),
            'contrast_range': (0.8, 1.2),
            'augmentation_prob': 1.0  # Always apply for testing
        }
        self.augmenter = DataAugmentation(self.config)
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Create test annotations
        self.test_annotations = [
            {'class_id': 0, 'bbox': [0.5, 0.5, 0.2, 0.2]},
            {'class_id': 1, 'bbox': [0.3, 0.7, 0.1, 0.1]}
        ]
        
    def test_color_jittering_range(self):
        """Test that color jittering keeps values in valid range."""
        brightness = 0.1
        contrast = 1.1
        
        jittered = self.augmenter.color_jittering(self.test_image, brightness, contrast)
        
        # Check that all values are in valid range [0, 255]
        self.assertTrue(np.all(jittered >= 0))
        self.assertTrue(np.all(jittered <= 255))
        self.assertEqual(jittered.dtype, np.uint8)
        
    def test_rotation_preserves_shape(self):
        """Test that rotation preserves image shape."""
        rotated, rotation_matrix = self.augmenter.random_rotation(self.test_image, (-10, 10))
        
        self.assertEqual(rotated.shape, self.test_image.shape)
        self.assertEqual(rotation_matrix.shape, (2, 3))
        
    def test_scaling_preserves_shape(self):
        """Test that scaling preserves image shape."""
        scaled, scale_factor = self.augmenter.random_scaling(self.test_image, (0.9, 1.1))
        
        self.assertEqual(scaled.shape, self.test_image.shape)
        self.assertIsInstance(scale_factor, float)
        
    def test_augmentation_preserves_annotation_format(self):
        """Test that augmentation preserves annotation format."""
        aug_image, aug_annotations = self.augmenter.apply_augmentation(
            self.test_image, self.test_annotations)
        
        # Check image shape is preserved
        self.assertEqual(aug_image.shape, self.test_image.shape)
        
        # Check annotation format is preserved
        for ann in aug_annotations:
            self.assertIn('class_id', ann)
            self.assertIn('bbox', ann)
            self.assertEqual(len(ann['bbox']), 4)
            
            # Check bbox coordinates are normalized
            bbox = ann['bbox']
            for coord in bbox:
                self.assertGreaterEqual(coord, 0.0)
                self.assertLessEqual(coord, 1.0)
                
    def test_flip_functionality(self):
        """Test flip functionality."""
        flipped, flip_h, flip_v = self.augmenter.random_flip(self.test_image)
        
        self.assertEqual(flipped.shape, self.test_image.shape)
        self.assertIsInstance(flip_h, bool)
        self.assertIsInstance(flip_v, bool)


if __name__ == '__main__':
    unittest.main()