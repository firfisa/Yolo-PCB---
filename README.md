# PCB Defect Detection System

A comprehensive system for detecting PCB defects using YOLO-based object detection. The system can identify and localize five types of PCB defects: Mouse_bite, Open_circuit, Short, Spur, and Spurious_copper.

## Features

- **Multi-class Detection**: Supports 5 PCB defect types
- **YOLO Integration**: Built on state-of-the-art YOLO architectures
- **Comprehensive Evaluation**: mAP calculation and per-class metrics
- **Visualization**: Side-by-side comparison of predictions vs ground truth
- **Data Augmentation**: Advanced augmentation techniques for better performance
- **Property-Based Testing**: Rigorous testing with Hypothesis framework

## Project Structure

```
pcb_detection/
├── core/                   # Core data types and interfaces
│   ├── types.py           # Data classes (Detection, EvaluationMetrics, etc.)
│   └── interfaces.py      # Abstract interfaces for system components
├── data/                  # Data processing and augmentation
│   ├── dataset.py         # PCB dataset implementation
│   └── augmentation.py    # Data augmentation utilities
├── models/                # Model implementations
│   ├── yolo_detector.py   # Main YOLO detector class
│   └── detection_head.py  # Detection head component
├── training/              # Training management
│   ├── trainer.py         # Training loop manager
│   └── checkpoint_manager.py # Model checkpoint handling
├── evaluation/            # Evaluation and metrics
│   ├── evaluator.py       # Main evaluator class
│   └── metrics.py         # Metrics calculation utilities
├── visualization/         # Visualization and plotting
│   ├── visualizer.py      # Main visualizer class
│   └── plotting.py        # Plotting utilities
└── utils/                 # Utility functions
    ├── file_utils.py      # File I/O operations
    ├── image_utils.py     # Image processing utilities
    └── config_utils.py    # Configuration management
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pcb-defect-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from pcb_detection import YOLODetector, Evaluator, Visualizer
from pcb_detection.core.types import TrainingConfig

# Initialize components
detector = YOLODetector(model_config={}, num_classes=5)
evaluator = Evaluator(iou_threshold=0.5)
visualizer = Visualizer(class_names=["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"])

# Training configuration
config = TrainingConfig(
    model_name="yolov8n",
    epochs=300,
    batch_size=16,
    learning_rate=0.01
)
```

### Configuration

The system uses YAML configuration files. See `config/default_config.yaml` for the default configuration.

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pcb_detection

# Run specific test module
pytest tests/test_core/test_types.py
```

### Property-Based Testing

The system includes property-based tests using the Hypothesis framework to ensure correctness across a wide range of inputs.

## Data Format

The system expects data in YOLO format:
- Images in standard formats (JPG, PNG, BMP)
- Annotations in YOLO format: `class_id x_center y_center width height` (normalized coordinates)

### Class Mapping

```python
CLASS_MAPPING = {
    0: "Mouse_bite",      # 鼠标咬痕
    1: "Open_circuit",    # 开路
    2: "Short",           # 短路  
    3: "Spur",            # 毛刺
    4: "Spurious_copper"  # 杂散铜
}
```

## Development

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking

Run code quality checks:
```bash
black pcb_detection/
flake8 pcb_detection/
mypy pcb_detection/
```

### Adding New Features

1. Follow the established interfaces in `pcb_detection/core/interfaces.py`
2. Add comprehensive tests including property-based tests where applicable
3. Update documentation and type hints
4. Ensure all tests pass before submitting

## License

[Add license information here]

## Contributing

[Add contribution guidelines here]