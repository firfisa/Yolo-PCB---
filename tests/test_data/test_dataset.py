"""
Unit tests for PCB dataset functionality.
"""

import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import cv2

from pcb_detection.data.dataset import PCBDataset
from pcb_detection.core.types import CLASS_MAPPING


class TestPCBDataset(unittest.TestCase):
    """Test PCB dataset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def test_class_distribution_calculation(self):
        """Test class distribution calculation."""
        # Create mock annotations
        annotations = [
            {
                'filename': 'test1.jpg',
                'objects': [
                    {'class_id': 0, 'bbox': [0.5, 0.5, 0.2, 0.2]},
                    {'class_id': 1, 'bbox': [0.3, 0.3, 0.1, 0.1]}
                ]
            },
            {
                'filename': 'test2.jpg', 
                'objects': [
                    {'class_id': 0, 'bbox': [0.4, 0.4, 0.15, 0.15]}
                ]
            }
        ]
        
        # Create dataset instance without loading data
        dataset = PCBDataset.__new__(PCBDataset)  # Create without calling __init__
        dataset.annotations = annotations
        
        # Test distribution calculation
        distribution = dataset.get_class_distribution()
        
        expected = {name: 0 for name in CLASS_MAPPING.values()}
        expected['Mouse_bite'] = 2  # class_id 0 appears twice
        expected['Open_circuit'] = 1  # class_id 1 appears once
        
        self.assertEqual(distribution, expected)
        
    def test_bbox_validation(self):
        """Test bounding box coordinate validation."""
        # Test valid bbox
        valid_bbox = [0.5, 0.5, 0.2, 0.2]
        self.assertTrue(all(0 <= coord <= 1 for coord in valid_bbox))
        
        # Test invalid bbox (coordinates > 1)
        invalid_bbox = [1.5, 0.5, 0.2, 0.2]
        self.assertFalse(all(0 <= coord <= 1 for coord in invalid_bbox))
        
    def test_yolo_annotation_parsing(self):
        """Test YOLO format annotation parsing."""
        # Test YOLO line parsing logic
        yolo_line = "0 0.5 0.5 0.2 0.2"
        parts = yolo_line.split()
        
        self.assertEqual(len(parts), 5)
        self.assertEqual(int(parts[0]), 0)  # class_id
        self.assertEqual(float(parts[1]), 0.5)  # x_center
        self.assertEqual(float(parts[2]), 0.5)  # y_center
        self.assertEqual(float(parts[3]), 0.2)  # width
        self.assertEqual(float(parts[4]), 0.2)  # height


if __name__ == '__main__':
    unittest.main()