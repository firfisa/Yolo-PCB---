#!/usr/bin/env python3
"""
Verification script to test the PCB detection package setup.
"""

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from pcb_detection.core.types import Detection, EvaluationMetrics, TrainingConfig
        from pcb_detection.core.types import CLASS_MAPPING, CLASS_LABELS, CLASS_COLORS
        print("✓ Core types imported successfully")
        
        # Test interface imports
        from pcb_detection.core.interfaces import (
            DatasetInterface, ModelInterface, EvaluatorInterface, 
            VisualizerInterface, DataAugmentationInterface
        )
        print("✓ Core interfaces imported successfully")
        
        # Test utility imports
        from pcb_detection.utils.file_utils import FileUtils
        from pcb_detection.utils.config_utils import ConfigUtils
        from pcb_detection.utils.image_utils import ImageUtils
        print("✓ Utility classes imported successfully")
        
        # Test main package imports
        from pcb_detection import Detection, EvaluationMetrics, TrainingConfig
        print("✓ Main package imports working")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_types():
    """Test that data types work correctly."""
    print("\nTesting data types...")
    
    try:
        from pcb_detection.core.types import Detection, TrainingConfig, CLASS_MAPPING
        
        # Test Detection creation
        detection = Detection(
            bbox=(0.1, 0.2, 0.3, 0.4),
            confidence=0.85,
            class_id=0,
            class_name="Mouse_bite"
        )
        print(f"✓ Detection created: {detection.class_name} with confidence {detection.confidence}")
        
        # Test TrainingConfig creation
        config = TrainingConfig(
            model_name="yolov8n",
            epochs=100,
            batch_size=16
        )
        print(f"✓ TrainingConfig created: {config.model_name}, {config.epochs} epochs")
        
        # Test class mapping
        print(f"✓ Class mapping loaded: {len(CLASS_MAPPING)} classes")
        for class_id, class_name in CLASS_MAPPING.items():
            print(f"  {class_id}: {class_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data type error: {e}")
        return False


def test_config_utils():
    """Test configuration utilities."""
    print("\nTesting configuration utilities...")
    
    try:
        from pcb_detection.utils.config_utils import ConfigUtils
        
        # Test default config creation
        config = ConfigUtils.create_default_config()
        print(f"✓ Default config created with {len(config)} sections")
        
        # Test config validation
        is_valid = ConfigUtils.validate_config(config)
        print(f"✓ Config validation: {is_valid}")
        
        # Test config to training config conversion
        training_config = ConfigUtils.config_to_training_config(config)
        print(f"✓ Training config conversion: {training_config.model_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config utils error: {e}")
        return False


def main():
    """Run all verification tests."""
    print("PCB Defect Detection System - Setup Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_types,
        test_config_utils,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Verification Summary:")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! Setup is complete.")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1


if __name__ == "__main__":
    exit(main())