"""
File utility functions.
"""

import os
import json
import yaml
from typing import Dict, Any, List
import pandas as pd


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def ensure_dir(path: str) -> None:
        """
        Ensure directory exists.
        
        Args:
            path: Directory path
        """
        os.makedirs(path, exist_ok=True)
        
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Loaded data
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    @staticmethod
    def save_json(data: Dict[str, Any], path: str) -> None:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            path: Path to save file
        """
        FileUtils.ensure_dir(os.path.dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Loaded data
        """
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    @staticmethod
    def save_yaml(data: Dict[str, Any], path: str) -> None:
        """
        Save data to YAML file.
        
        Args:
            data: Data to save
            path: Path to save file
        """
        FileUtils.ensure_dir(os.path.dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
    @staticmethod
    def save_csv(data: List[Dict[str, Any]], path: str) -> None:
        """
        Save data to CSV file.
        
        Args:
            data: Data to save
            path: Path to save file
        """
        FileUtils.ensure_dir(os.path.dirname(path))
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)