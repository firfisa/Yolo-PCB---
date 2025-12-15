"""
PCB Dataset implementation.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path

from ..core.interfaces import DatasetInterface
from ..core.types import CLASS_MAPPING, CLASS_NAME_TO_ID, NUM_CLASSES
from .augmentation import DataAugmentation
from .advanced_augmentation import PCBAdvancedAugmentation


class PCBDataset(Dataset, DatasetInterface):
    """PCB dataset for loading and processing PCB defect data."""
    
    def __init__(self, data_path: str, mode: str = "train", image_size: int = 640, 
                 augmentation_config: Optional[Dict] = None, 
                 use_advanced_augmentation: bool = False):
        """
        Initialize PCB dataset.
        
        Args:
            data_path: Path to dataset directory
            mode: Dataset mode ('train', 'val', 'test')
            image_size: Target image size for resizing
            augmentation_config: Configuration for data augmentation
            use_advanced_augmentation: Whether to use advanced augmentation techniques
        """
        self.data_path = Path(data_path)
        self.mode = mode
        self.image_size = image_size
        self.use_advanced_augmentation = use_advanced_augmentation
        
        # Initialize augmentation
        self.augmentation = None
        self.advanced_augmentation = None
        
        if augmentation_config and mode == "train":
            # Basic augmentation
            basic_config = augmentation_config.get('basic', {})
            if basic_config:
                self.augmentation = DataAugmentation(basic_config)
            
            # Advanced augmentation
            if use_advanced_augmentation:
                advanced_config = augmentation_config.get('advanced', {})
                self.advanced_augmentation = PCBAdvancedAugmentation(
                    image_size=image_size,
                    mosaic_prob=advanced_config.get('mosaic_prob', 0.5),
                    copy_paste_prob=advanced_config.get('copy_paste_prob', 0.3),
                    mixup_prob=advanced_config.get('mixup_prob', 0.2),
                    use_albumentations=advanced_config.get('use_albumentations', True)
                )
        
        # Load and validate data
        self.annotations = self.load_annotations()
        self.image_paths = self._get_image_paths()
        
        # Validate data consistency
        self._validate_data_consistency()
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.annotations)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        if self.use_advanced_augmentation and self.advanced_augmentation and self.mode == "train":
            return self._get_item_with_advanced_augmentation(idx)
        else:
            return self._get_item_basic(idx)
    
    def _get_item_basic(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item with basic augmentation."""
        annotation = self.annotations[idx]
        image_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply basic augmentation if available
        if self.augmentation:
            image, objects = self.augmentation.apply_augmentation(image, annotation['objects'])
        else:
            objects = annotation['objects']
        
        # Resize image while maintaining aspect ratio
        image, scale_factor = self._resize_image(image)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Process annotations
        targets = self._process_annotations(objects, scale_factor)
        
        return image_tensor, targets
    
    def _get_item_with_advanced_augmentation(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item with advanced augmentation techniques."""
        # For advanced augmentation, we may need multiple images
        images = []
        annotations_list = []
        
        # Get primary image
        primary_annotation = self.annotations[idx]
        primary_image_path = self.image_paths[idx]
        
        primary_image = cv2.imread(str(primary_image_path))
        if primary_image is None:
            raise ValueError(f"Could not load image: {primary_image_path}")
        
        primary_image = cv2.cvtColor(primary_image, cv2.COLOR_BGR2RGB)
        images.append(primary_image)
        annotations_list.append(primary_annotation['objects'])
        
        # For mosaic and mixup, get additional images
        if len(self.annotations) > 1:
            # Get 3 more random images for mosaic (total 4)
            import random
            additional_indices = random.sample(
                [i for i in range(len(self.annotations)) if i != idx], 
                min(3, len(self.annotations) - 1)
            )
            
            for add_idx in additional_indices:
                add_annotation = self.annotations[add_idx]
                add_image_path = self.image_paths[add_idx]
                
                add_image = cv2.imread(str(add_image_path))
                if add_image is not None:
                    add_image = cv2.cvtColor(add_image, cv2.COLOR_BGR2RGB)
                    images.append(add_image)
                    annotations_list.append(add_annotation['objects'])
        
        # Apply advanced augmentation
        augmented_image, augmented_objects = self.advanced_augmentation(images, annotations_list)
        
        # Apply basic augmentation if available (after advanced)
        if self.augmentation:
            augmented_image, augmented_objects = self.augmentation.apply_augmentation(
                augmented_image, augmented_objects)
        
        # Resize image while maintaining aspect ratio
        augmented_image, scale_factor = self._resize_image(augmented_image)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(augmented_image).permute(2, 0, 1).float() / 255.0
        
        # Process annotations
        targets = self._process_annotations(augmented_objects, scale_factor)
        
        return image_tensor, targets
        
    def load_annotations(self) -> List[Dict]:
        """Load annotation data."""
        annotations = []
        
        if self.mode == "test":
            # Handle test dataset with YOLO format
            annotations = self._load_yolo_annotations()
        else:
            # Handle training dataset with XML format
            annotations = self._load_xml_annotations()
            
        return annotations
        
    def _load_xml_annotations(self) -> List[Dict]:
        """Load XML format annotations from training dataset."""
        annotations = []
        
        # Iterate through all defect type folders
        annotations_dir = self.data_path / "Annotations"
        if not annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {annotations_dir}")
            
        for defect_folder in annotations_dir.iterdir():
            if not defect_folder.is_dir():
                continue
                
            defect_name = defect_folder.name
            if defect_name not in CLASS_NAME_TO_ID:
                print(f"Warning: Unknown defect type {defect_name}, skipping...")
                continue
                
            # Process all XML files in this defect folder
            for xml_file in defect_folder.glob("*.xml"):
                try:
                    annotation = self._parse_xml_annotation(xml_file, defect_name)
                    if annotation:
                        annotations.append(annotation)
                except Exception as e:
                    print(f"Error parsing {xml_file}: {e}")
                    
        return annotations
        
    def _load_yolo_annotations(self) -> List[Dict]:
        """Load YOLO format annotations from test dataset."""
        annotations = []
        
        # Find all defect type folders in test dataset
        for defect_folder in self.data_path.iterdir():
            if not defect_folder.is_dir() or not defect_folder.name.endswith("_txt"):
                continue
                
            # Extract defect name from folder name (remove _txt suffix)
            defect_name = defect_folder.name[:-4]
            
            # Process all txt files in this folder
            for txt_file in defect_folder.glob("*.txt"):
                if txt_file.name == "classes.txt":
                    continue
                    
                try:
                    annotation = self._parse_yolo_annotation(txt_file, defect_name)
                    if annotation:
                        annotations.append(annotation)
                except Exception as e:
                    print(f"Error parsing {txt_file}: {e}")
                    
        return annotations
        
    def _parse_xml_annotation(self, xml_file: Path, defect_name: str) -> Optional[Dict]:
        """Parse a single XML annotation file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image info
            filename = root.find('filename').text
            size_elem = root.find('size')
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            
            # Get objects
            objects = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # Map object name to class ID
                class_id = self._map_object_name_to_class_id(name, defect_name)
                if class_id is None:
                    continue
                    
                bbox_elem = obj.find('bndbox')
                xmin = int(bbox_elem.find('xmin').text)
                ymin = int(bbox_elem.find('ymin').text)
                xmax = int(bbox_elem.find('xmax').text)
                ymax = int(bbox_elem.find('ymax').text)
                
                # Convert to normalized YOLO format (x_center, y_center, width, height)
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                objects.append({
                    'class_id': class_id,
                    'bbox': [x_center, y_center, bbox_width, bbox_height]
                })
                
            # Construct correct image path
            image_path = self.data_path / "images" / defect_name / filename
            
            return {
                'filename': filename,
                'image_path': image_path,
                'width': width,
                'height': height,
                'objects': objects
            }
            
        except Exception as e:
            print(f"Error parsing XML file {xml_file}: {e}")
            return None
            
    def _parse_yolo_annotation(self, txt_file: Path, defect_name: str) -> Optional[Dict]:
        """Parse a single YOLO format annotation file."""
        try:
            # Find corresponding image file
            image_name = txt_file.stem + ".bmp"  # Test images are in BMP format
            image_folder = txt_file.parent.parent / f"{defect_name}_Img"
            image_path = image_folder / image_name
            
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}")
                return None
                
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image: {image_path}")
                return None
                
            height, width = image.shape[:2]
            
            # Parse annotations
            objects = []
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                        
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    bbox_width = float(parts[3])
                    bbox_height = float(parts[4])
                    
                    objects.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, bbox_width, bbox_height]
                    })
                    
            return {
                'filename': image_name,
                'image_path': image_path,
                'width': width,
                'height': height,
                'objects': objects
            }
            
        except Exception as e:
            print(f"Error parsing YOLO file {txt_file}: {e}")
            return None
            
    def _map_object_name_to_class_id(self, object_name: str, defect_name: str) -> Optional[int]:
        """Map object name from XML to class ID."""
        # Normalize names for mapping
        name_mapping = {
            'mouse_bite': 'Mouse_bite',
            'open_circuit': 'Open_circuit', 
            'short': 'Short',
            'spur': 'Spur',
            'spurious_copper': 'Spurious_copper'
        }
        
        # Try direct mapping first
        if defect_name in CLASS_NAME_TO_ID:
            return CLASS_NAME_TO_ID[defect_name]
            
        # Try normalized mapping
        normalized_name = name_mapping.get(object_name.lower())
        if normalized_name and normalized_name in CLASS_NAME_TO_ID:
            return CLASS_NAME_TO_ID[normalized_name]
            
        print(f"Warning: Unknown object name '{object_name}' in defect '{defect_name}'")
        return None
        
    def _get_image_paths(self) -> List[Path]:
        """Get list of image paths corresponding to annotations."""
        return [Path(ann['image_path']) for ann in self.annotations]
        
    def _validate_data_consistency(self) -> None:
        """Validate data consistency between images and annotations."""
        missing_images = []
        invalid_annotations = []
        valid_annotations = []
        valid_image_paths = []
        
        for i, (annotation, image_path) in enumerate(zip(self.annotations, self.image_paths)):
            # Check if image file exists
            if not image_path.exists():
                missing_images.append(str(image_path))
                continue
                
            # Check if annotation has valid objects
            if not annotation['objects']:
                invalid_annotations.append(f"No objects in annotation {i}: {annotation['filename']}")
                continue
                
            # Validate bbox coordinates
            valid_objects = []
            for obj in annotation['objects']:
                bbox = obj['bbox']
                if all(0 <= coord <= 1 for coord in bbox):
                    valid_objects.append(obj)
                else:
                    invalid_annotations.append(f"Invalid bbox coordinates in {annotation['filename']}: {bbox}")
                    
            # Only keep annotations with valid objects and existing images
            if valid_objects:
                annotation['objects'] = valid_objects
                valid_annotations.append(annotation)
                valid_image_paths.append(image_path)
                
        # Update with only valid data
        self.annotations = valid_annotations
        self.image_paths = valid_image_paths
                    
        if missing_images:
            print(f"Warning: {len(missing_images)} missing image files (filtered out)")
                
        if invalid_annotations:
            print(f"Warning: {len(invalid_annotations)} invalid annotations (filtered out)")
            
        print(f"Dataset loaded with {len(self.annotations)} valid samples")
                
    def _resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        
        # Calculate scale factor
        scale = min(self.image_size / w, self.image_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (self.image_size - new_w) // 2
        pad_y = (self.image_size - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded, scale
        
    def _process_annotations(self, objects: List[Dict], scale_factor: float) -> torch.Tensor:
        """Process annotations into tensor format."""
        if not objects:
            # Return empty tensor if no objects
            return torch.zeros((0, 6))  # [batch_idx, class_id, x, y, w, h]
            
        targets = []
        for obj in objects:
            class_id = obj['class_id']
            bbox = obj['bbox']  # [x_center, y_center, width, height] normalized
            
            # Add batch index (0 for single image)
            target = [0, class_id] + bbox
            targets.append(target)
            
        return torch.tensor(targets, dtype=torch.float32)
        
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution statistics."""
        distribution = {name: 0 for name in CLASS_MAPPING.values()}
        
        for annotation in self.annotations:
            for obj in annotation['objects']:
                class_id = obj['class_id']
                if class_id in CLASS_MAPPING:
                    class_name = CLASS_MAPPING[class_id]
                    distribution[class_name] += 1
                    
        return distribution


def create_pcb_dataset_with_advanced_augmentation(
    data_path: str, 
    mode: str = "train", 
    image_size: int = 640,
    config_type: str = "balanced"
) -> PCBDataset:
    """
    Create PCB dataset with advanced augmentation configuration.
    
    Args:
        data_path: Path to dataset directory
        mode: Dataset mode ('train', 'val', 'test')
        image_size: Target image size
        config_type: Configuration type ('basic', 'balanced', 'performance')
        
    Returns:
        PCBDataset with appropriate augmentation configuration
    """
    # Define augmentation configurations
    augmentation_configs = {
        'basic': {
            'basic': {
                'rotation_range': (-10, 10),
                'scale_range': (0.9, 1.1),
                'brightness_range': (-0.1, 0.1),
                'contrast_range': (0.9, 1.1),
                'flip_horizontal': True,
                'flip_vertical': False,
                'augmentation_prob': 0.5
            },
            'advanced': {
                'mosaic_prob': 0.0,
                'copy_paste_prob': 0.0,
                'mixup_prob': 0.0,
                'use_albumentations': False
            }
        },
        'balanced': {
            'basic': {
                'rotation_range': (-15, 15),
                'scale_range': (0.8, 1.2),
                'brightness_range': (-0.2, 0.2),
                'contrast_range': (0.8, 1.2),
                'flip_horizontal': True,
                'flip_vertical': True,
                'augmentation_prob': 0.6
            },
            'advanced': {
                'mosaic_prob': 0.3,
                'copy_paste_prob': 0.2,
                'mixup_prob': 0.1,
                'use_albumentations': True
            }
        },
        'performance': {
            'basic': {
                'rotation_range': (-20, 20),
                'scale_range': (0.7, 1.3),
                'brightness_range': (-0.3, 0.3),
                'contrast_range': (0.7, 1.3),
                'flip_horizontal': True,
                'flip_vertical': True,
                'augmentation_prob': 0.8
            },
            'advanced': {
                'mosaic_prob': 0.5,
                'copy_paste_prob': 0.3,
                'mixup_prob': 0.2,
                'use_albumentations': True
            }
        }
    }
    
    config = augmentation_configs.get(config_type, augmentation_configs['balanced'])
    use_advanced = any(config['advanced'].values()) if isinstance(config['advanced'], dict) else False
    
    return PCBDataset(
        data_path=data_path,
        mode=mode,
        image_size=image_size,
        augmentation_config=config,
        use_advanced_augmentation=use_advanced
    )