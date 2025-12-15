"""
Unit tests for advanced data augmentation functionality.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from pcb_detection.data.advanced_augmentation import (
    MosaicAugmentation,
    CopyPasteAugmentation,
    MixUpAugmentation,
    AlbumentationsAugmentation,
    PCBAdvancedAugmentation
)


class TestMosaicAugmentation(unittest.TestCase):
    """Test Mosaic augmentation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mosaic = MosaicAugmentation(image_size=640, prob=1.0)  # Always apply
        
        # Create test images
        self.test_images = [
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)
        ]
        
        # Create test annotations
        self.test_annotations = [
            [{'class_id': 0, 'bbox': [0.5, 0.5, 0.2, 0.2]}],
            [{'class_id': 1, 'bbox': [0.3, 0.7, 0.1, 0.1]}],
            [{'class_id': 2, 'bbox': [0.7, 0.3, 0.15, 0.15]}],
            [{'class_id': 0, 'bbox': [0.6, 0.6, 0.1, 0.1]}]
        ]
        
    def test_mosaic_output_shape(self):
        """Test that mosaic produces correct output shape."""
        result_image, result_annotations = self.mosaic(self.test_images, self.test_annotations)
        
        self.assertEqual(result_image.shape, (640, 640, 3))
        self.assertIsInstance(result_annotations, list)
        
    def test_mosaic_annotation_format(self):
        """Test that mosaic preserves annotation format."""
        result_image, result_annotations = self.mosaic(self.test_images, self.test_annotations)
        
        for ann in result_annotations:
            self.assertIn('class_id', ann)
            self.assertIn('bbox', ann)
            self.assertEqual(len(ann['bbox']), 4)
            
            # Check bbox coordinates are normalized
            bbox = ann['bbox']
            for coord in bbox:
                self.assertGreaterEqual(coord, 0.0)
                self.assertLessEqual(coord, 1.0)
                
    def test_mosaic_with_insufficient_images(self):
        """Test mosaic behavior with insufficient images."""
        single_image = [self.test_images[0]]
        single_annotation = [self.test_annotations[0]]
        
        result_image, result_annotations = self.mosaic(single_image, single_annotation)
        
        # Should return original image when insufficient images
        np.testing.assert_array_equal(result_image, single_image[0])
        self.assertEqual(result_annotations, single_annotation[0])


class TestCopyPasteAugmentation(unittest.TestCase):
    """Test Copy-Paste augmentation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.copy_paste = CopyPasteAugmentation(prob=1.0, max_paste=2)  # Always apply
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Create test annotations
        self.test_annotations = [
            {'class_id': 0, 'bbox': [0.5, 0.5, 0.2, 0.2]},
            {'class_id': 1, 'bbox': [0.3, 0.7, 0.1, 0.1]}
        ]
        
    def test_copy_paste_output_shape(self):
        """Test that copy-paste preserves image shape."""
        result_image, result_annotations = self.copy_paste(self.test_image, self.test_annotations)
        
        self.assertEqual(result_image.shape, self.test_image.shape)
        self.assertIsInstance(result_annotations, list)
        
    def test_copy_paste_increases_annotations(self):
        """Test that copy-paste can increase number of annotations."""
        result_image, result_annotations = self.copy_paste(self.test_image, self.test_annotations)
        
        # Should have at least original annotations
        self.assertGreaterEqual(len(result_annotations), len(self.test_annotations))
        
    def test_copy_paste_annotation_format(self):
        """Test that copy-paste preserves annotation format."""
        result_image, result_annotations = self.copy_paste(self.test_image, self.test_annotations)
        
        for ann in result_annotations:
            self.assertIn('class_id', ann)
            self.assertIn('bbox', ann)
            self.assertEqual(len(ann['bbox']), 4)
            
            # Check bbox coordinates are normalized
            bbox = ann['bbox']
            for coord in bbox:
                self.assertGreaterEqual(coord, 0.0)
                self.assertLessEqual(coord, 1.0)
                
    def test_copy_paste_with_empty_annotations(self):
        """Test copy-paste behavior with empty annotations."""
        result_image, result_annotations = self.copy_paste(self.test_image, [])
        
        # Should return original image and empty annotations
        np.testing.assert_array_equal(result_image, self.test_image)
        self.assertEqual(result_annotations, [])


class TestMixUpAugmentation(unittest.TestCase):
    """Test MixUp augmentation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mixup = MixUpAugmentation(alpha=0.2, prob=1.0)  # Always apply
        
        # Create test images
        self.test_image1 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.test_image2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Create test annotations
        self.test_annotations1 = [{'class_id': 0, 'bbox': [0.5, 0.5, 0.2, 0.2]}]
        self.test_annotations2 = [{'class_id': 1, 'bbox': [0.3, 0.7, 0.1, 0.1]}]
        
    def test_mixup_output_shape(self):
        """Test that mixup produces correct output shape."""
        result_image, result_annotations = self.mixup(
            self.test_image1, self.test_annotations1,
            self.test_image2, self.test_annotations2
        )
        
        self.assertEqual(result_image.shape, self.test_image1.shape)
        self.assertIsInstance(result_annotations, list)
        
    def test_mixup_combines_annotations(self):
        """Test that mixup combines annotations from both images."""
        result_image, result_annotations = self.mixup(
            self.test_image1, self.test_annotations1,
            self.test_image2, self.test_annotations2
        )
        
        # Should combine annotations from both images
        expected_count = len(self.test_annotations1) + len(self.test_annotations2)
        self.assertEqual(len(result_annotations), expected_count)
        
    def test_mixup_annotation_format(self):
        """Test that mixup preserves annotation format."""
        result_image, result_annotations = self.mixup(
            self.test_image1, self.test_annotations1,
            self.test_image2, self.test_annotations2
        )
        
        for ann in result_annotations:
            self.assertIn('class_id', ann)
            self.assertIn('bbox', ann)
            self.assertEqual(len(ann['bbox']), 4)


class TestAlbumentationsAugmentation(unittest.TestCase):
    """Test Albumentations augmentation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.albu_train = AlbumentationsAugmentation(image_size=640, train=True)
            self.albu_val = AlbumentationsAugmentation(image_size=640, train=False)
            self.albumentations_available = True
        except ImportError:
            self.albumentations_available = False
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Create test annotations
        self.test_annotations = [
            {'class_id': 0, 'bbox': [0.5, 0.5, 0.2, 0.2]},
            {'class_id': 1, 'bbox': [0.3, 0.7, 0.1, 0.1]}
        ]
        
    def test_albumentations_initialization(self):
        """Test Albumentations initialization."""
        if not self.albumentations_available:
            self.skipTest("Albumentations not available")
        
        # Test that initialization works
        self.assertIsNotNone(self.albu_train)
        self.assertIsNotNone(self.albu_val)
        
    def test_albumentations_output_shape(self):
        """Test that Albumentations produces correct output shape."""
        if not self.albumentations_available:
            self.skipTest("Albumentations not available")
            
        result_image, result_annotations = self.albu_train(self.test_image, self.test_annotations)
        
        self.assertEqual(result_image.shape, (640, 640, 3))
        self.assertIsInstance(result_annotations, list)
            
    def test_albumentations_with_empty_annotations(self):
        """Test Albumentations behavior with empty annotations."""
        if not self.albumentations_available:
            self.skipTest("Albumentations not available")
            
        result_image, result_annotations = self.albu_train(self.test_image, [])
        
        self.assertEqual(result_image.shape, (640, 640, 3))
        self.assertEqual(result_annotations, [])


class TestPCBAdvancedAugmentation(unittest.TestCase):
    """Test combined PCB advanced augmentation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = PCBAdvancedAugmentation(
            image_size=640,
            mosaic_prob=0.5,
            copy_paste_prob=0.3,
            mixup_prob=0.2,
            use_albumentations=False  # Disable to avoid import issues in tests
        )
        
        # Create test images
        self.test_images = [
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)
        ]
        
        # Create test annotations
        self.test_annotations = [
            [{'class_id': 0, 'bbox': [0.5, 0.5, 0.2, 0.2]}],
            [{'class_id': 1, 'bbox': [0.3, 0.7, 0.1, 0.1]}],
            [{'class_id': 2, 'bbox': [0.7, 0.3, 0.15, 0.15]}],
            [{'class_id': 0, 'bbox': [0.6, 0.6, 0.1, 0.1]}]
        ]
        
    def test_pipeline_output_shape(self):
        """Test that pipeline produces correct output shape."""
        result_image, result_annotations = self.pipeline(self.test_images, self.test_annotations)
        
        self.assertEqual(result_image.shape, (640, 640, 3))
        self.assertIsInstance(result_annotations, list)
        
    def test_pipeline_annotation_format(self):
        """Test that pipeline preserves annotation format."""
        result_image, result_annotations = self.pipeline(self.test_images, self.test_annotations)
        
        for ann in result_annotations:
            self.assertIn('class_id', ann)
            self.assertIn('bbox', ann)
            self.assertEqual(len(ann['bbox']), 4)
            
            # Check bbox coordinates are normalized
            bbox = ann['bbox']
            for coord in bbox:
                self.assertGreaterEqual(coord, 0.0)
                self.assertLessEqual(coord, 1.0)
                
    def test_pipeline_with_single_image(self):
        """Test pipeline behavior with single image."""
        single_image = [self.test_images[0]]
        single_annotation = [self.test_annotations[0]]
        
        result_image, result_annotations = self.pipeline(single_image, single_annotation)
        
        self.assertEqual(result_image.shape, (640, 640, 3))
        self.assertIsInstance(result_annotations, list)
        
    def test_pipeline_configuration_options(self):
        """Test different pipeline configurations."""
        # Test basic configuration (no advanced augmentation)
        basic_pipeline = PCBAdvancedAugmentation(
            image_size=640,
            mosaic_prob=0.0,
            copy_paste_prob=0.0,
            mixup_prob=0.0,
            use_albumentations=False
        )
        
        result_image, result_annotations = basic_pipeline(self.test_images, self.test_annotations)
        
        self.assertEqual(result_image.shape, (640, 640, 3))
        self.assertIsInstance(result_annotations, list)


if __name__ == '__main__':
    unittest.main()