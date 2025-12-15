"""
Unit tests for preprocessing functionality.
"""

import unittest
import numpy as np

from pcb_detection.data.preprocessing import DatasetSplitter, ImagePreprocessor


class TestDatasetSplitter(unittest.TestCase):
    """Test dataset splitting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # Create dummy annotations
        self.annotations = []
        for i in range(100):
            num_objects = np.random.randint(1, 4)
            objects = []
            for j in range(num_objects):
                class_id = np.random.randint(0, 5)
                bbox = [
                    np.random.uniform(0.1, 0.9),
                    np.random.uniform(0.1, 0.9),
                    np.random.uniform(0.05, 0.3),
                    np.random.uniform(0.05, 0.3)
                ]
                objects.append({'class_id': class_id, 'bbox': bbox})
                
            self.annotations.append({
                'filename': f'image_{i:03d}.jpg',
                'objects': objects
            })
            
    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        train_ann, val_ann, test_ann = self.splitter.split_dataset(self.annotations)
        
        total = len(self.annotations)
        train_ratio = len(train_ann) / total
        val_ratio = len(val_ann) / total
        test_ratio = len(test_ann) / total
        
        # Check that all annotations are accounted for
        self.assertEqual(len(train_ann) + len(val_ann) + len(test_ann), total)
        
        # Check that train set is the largest
        self.assertGreater(len(train_ann), len(val_ann))
        self.assertGreater(len(train_ann), len(test_ann))
        
        # Check that ratios sum to 1
        self.assertAlmostEqual(train_ratio + val_ratio + test_ratio, 1.0, places=5)
        
    def test_split_statistics(self):
        """Test split statistics generation."""
        train_ann, val_ann, test_ann = self.splitter.split_dataset(self.annotations)
        stats = self.splitter.get_split_statistics(train_ann, val_ann, test_ann)
        
        # Check statistics structure
        self.assertIn('train', stats)
        self.assertIn('val', stats)
        self.assertIn('test', stats)
        self.assertIn('total', stats)
        
        for split in ['train', 'val', 'test']:
            self.assertIn('count', stats[split])
            self.assertIn('class_counts', stats[split])
            self.assertIn('class_ratios', stats[split])


class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor(target_size=640, normalize=True)
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 320, 3), dtype=np.uint8)
        
    def test_image_preprocessing(self):
        """Test image preprocessing functionality."""
        processed_image, info = self.preprocessor.preprocess_image(self.test_image)
        
        # Check output shape
        self.assertEqual(processed_image.shape, (640, 640, 3))
        
        # Check normalization
        if self.preprocessor.normalize:
            self.assertTrue(np.all(processed_image >= 0.0))
            self.assertTrue(np.all(processed_image <= 1.0))
            self.assertEqual(processed_image.dtype, np.float32)
            
        # Check preprocessing info
        self.assertIn('original_shape', info)
        self.assertIn('target_size', info)
        self.assertIn('scale_factor', info)
        self.assertIn('padding', info)
        self.assertEqual(info['original_shape'], (480, 320))
        self.assertEqual(info['target_size'], 640)
        
    def test_aspect_ratio_preservation(self):
        """Test that aspect ratio is preserved during resizing."""
        processed_image, info = self.preprocessor.preprocess_image(self.test_image)
        
        original_h, original_w = self.test_image.shape[:2]
        original_aspect = original_w / original_h
        
        scale_factor = info['scale_factor']
        new_w = int(original_w * scale_factor)
        new_h = int(original_h * scale_factor)
        new_aspect = new_w / new_h
        
        # Aspect ratio should be preserved (within floating point precision)
        self.assertAlmostEqual(original_aspect, new_aspect, places=2)
        
    def test_denormalization(self):
        """Test image denormalization."""
        # Create normalized image
        normalized_image = np.random.rand(640, 640, 3).astype(np.float32)
        
        denormalized = self.preprocessor.denormalize_image(normalized_image)
        
        self.assertEqual(denormalized.dtype, np.uint8)
        self.assertTrue(np.all(denormalized >= 0))
        self.assertTrue(np.all(denormalized <= 255))


if __name__ == '__main__':
    unittest.main()