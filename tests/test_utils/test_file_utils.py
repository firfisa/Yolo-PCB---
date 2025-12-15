"""
Tests for file utilities.
"""

import pytest
import os
import json
import tempfile
from pcb_detection.utils.file_utils import FileUtils


class TestFileUtils:
    """Test FileUtils class."""
    
    def test_ensure_dir(self, temp_dir):
        """Test directory creation."""
        test_path = os.path.join(temp_dir, "test", "nested", "dir")
        FileUtils.ensure_dir(test_path)
        assert os.path.exists(test_path)
    
    def test_json_operations(self, temp_dir):
        """Test JSON save and load operations."""
        test_data = {"key": "value", "number": 42}
        json_path = os.path.join(temp_dir, "test.json")
        
        # Save JSON
        FileUtils.save_json(test_data, json_path)
        assert os.path.exists(json_path)
        
        # Load JSON
        loaded_data = FileUtils.load_json(json_path)
        assert loaded_data == test_data
    
    def test_yaml_operations(self, temp_dir):
        """Test YAML save and load operations."""
        test_data = {"key": "value", "list": [1, 2, 3]}
        yaml_path = os.path.join(temp_dir, "test.yaml")
        
        # Save YAML
        FileUtils.save_yaml(test_data, yaml_path)
        assert os.path.exists(yaml_path)
        
        # Load YAML
        loaded_data = FileUtils.load_yaml(yaml_path)
        assert loaded_data == test_data
    
    def test_csv_operations(self, temp_dir):
        """Test CSV save operations."""
        test_data = [
            {"name": "test1", "value": 1},
            {"name": "test2", "value": 2}
        ]
        csv_path = os.path.join(temp_dir, "test.csv")
        
        # Save CSV
        FileUtils.save_csv(test_data, csv_path)
        assert os.path.exists(csv_path)