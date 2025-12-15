"""
Tests for evaluation functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path

from pcb_detection.evaluation import Evaluator, MetricsCalculator, ReportGenerator, EvaluationPipeline
from pcb_detection.core.types import Detection, EvaluationMetrics


class TestMetricsCalculator:
    """Test metrics calculation utilities."""
    
    def test_iou_calculation_identical_boxes(self):
        """Test IoU calculation for identical boxes."""
        box1 = (0.5, 0.5, 0.2, 0.2)
        box2 = (0.5, 0.5, 0.2, 0.2)
        iou = MetricsCalculator.calculate_iou(box1, box2)
        assert abs(iou - 1.0) < 1e-6
    
    def test_iou_calculation_no_overlap(self):
        """Test IoU calculation for non-overlapping boxes."""
        box1 = (0.2, 0.2, 0.1, 0.1)
        box2 = (0.8, 0.8, 0.1, 0.1)
        iou = MetricsCalculator.calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_iou_calculation_partial_overlap(self):
        """Test IoU calculation for partially overlapping boxes."""
        box1 = (0.5, 0.5, 0.2, 0.2)
        box2 = (0.6, 0.6, 0.2, 0.2)
        iou = MetricsCalculator.calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
    
    def test_precision_recall_perfect_match(self):
        """Test precision and recall for perfect predictions."""
        predictions = [Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, class_id=0, class_name='Mouse_bite')]
        ground_truths = [Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=1.0, class_id=0, class_name='Mouse_bite')]
        
        precision, recall = MetricsCalculator.calculate_precision_recall(predictions, ground_truths)
        assert precision == 1.0
        assert recall == 1.0
    
    def test_precision_recall_no_predictions(self):
        """Test precision and recall with no predictions."""
        predictions = []
        ground_truths = [Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=1.0, class_id=0, class_name='Mouse_bite')]
        
        precision, recall = MetricsCalculator.calculate_precision_recall(predictions, ground_truths)
        assert precision == 0.0
        assert recall == 0.0
    
    def test_precision_recall_no_ground_truths(self):
        """Test precision and recall with no ground truths."""
        predictions = [Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, class_id=0, class_name='Mouse_bite')]
        ground_truths = []
        
        precision, recall = MetricsCalculator.calculate_precision_recall(predictions, ground_truths)
        assert precision == 0.0
        assert recall == 0.0


class TestEvaluator:
    """Test evaluator functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = Evaluator(iou_threshold=0.6)
        assert evaluator.iou_threshold == 0.6
    
    def test_map_calculation_empty_data(self):
        """Test mAP calculation with empty data."""
        evaluator = Evaluator()
        predictions = [[], []]
        ground_truths = [[], []]
        
        map_value = evaluator.calculate_map(predictions, ground_truths)
        assert map_value == 0.0
    
    def test_map_calculation_single_class(self):
        """Test mAP calculation with single class."""
        evaluator = Evaluator()
        predictions = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, class_id=0, class_name='Mouse_bite')]]
        ground_truths = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=1.0, class_id=0, class_name='Mouse_bite')]]
        
        map_value = evaluator.calculate_map(predictions, ground_truths)
        assert map_value > 0.0
    
    def test_ap_per_class_calculation(self):
        """Test AP per class calculation."""
        evaluator = Evaluator()
        predictions = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, class_id=0, class_name='Mouse_bite')]]
        ground_truths = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=1.0, class_id=0, class_name='Mouse_bite')]]
        
        ap_per_class = evaluator.calculate_ap_per_class(predictions, ground_truths)
        assert 'Mouse_bite' in ap_per_class
        assert ap_per_class['Mouse_bite'] > 0.0
        assert len(ap_per_class) == 5  # All 5 PCB defect classes
    
    def test_metrics_report_generation(self):
        """Test comprehensive metrics report generation."""
        evaluator = Evaluator()
        predictions = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, class_id=0, class_name='Mouse_bite')]]
        ground_truths = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=1.0, class_id=0, class_name='Mouse_bite')]]
        
        metrics = evaluator.generate_metrics_report(predictions, ground_truths)
        assert isinstance(metrics, EvaluationMetrics)
        assert 0.0 <= metrics.map_50 <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
    
    def test_save_results_json(self):
        """Test saving results in JSON format."""
        evaluator = Evaluator()
        metrics = EvaluationMetrics(
            map_50=0.5,
            ap_per_class={'Mouse_bite': 0.6, 'Open_circuit': 0.4, 'Short': 0.3, 'Spur': 0.7, 'Spurious_copper': 0.5},
            precision=0.8,
            recall=0.7,
            f1_score=0.75
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_results.json")
            evaluator.save_results(metrics, output_path, "json")
            assert os.path.exists(output_path)
    
    def test_save_results_csv(self):
        """Test saving results in CSV format."""
        evaluator = Evaluator()
        metrics = EvaluationMetrics(
            map_50=0.5,
            ap_per_class={'Mouse_bite': 0.6, 'Open_circuit': 0.4, 'Short': 0.3, 'Spur': 0.7, 'Spurious_copper': 0.5},
            precision=0.8,
            recall=0.7,
            f1_score=0.75
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_results.csv")
            evaluator.save_results(metrics, output_path, "csv")
            assert os.path.exists(output_path)


class TestReportGenerator:
    """Test report generation functionality."""
    
    def test_report_generator_initialization(self):
        """Test report generator initialization."""
        generator = ReportGenerator()
        assert hasattr(generator, 'timestamp')
    
    def test_detailed_report_generation(self):
        """Test detailed report generation."""
        generator = ReportGenerator()
        metrics = EvaluationMetrics(
            map_50=0.5,
            ap_per_class={'Mouse_bite': 0.6, 'Open_circuit': 0.4, 'Short': 0.3, 'Spur': 0.7, 'Spurious_copper': 0.5},
            precision=0.8,
            recall=0.7,
            f1_score=0.75
        )
        predictions = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, class_id=0, class_name='Mouse_bite')]]
        ground_truths = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=1.0, class_id=0, class_name='Mouse_bite')]]
        
        report = generator.generate_detailed_report(metrics, predictions, ground_truths)
        
        assert 'metadata' in report
        assert 'overall_metrics' in report
        assert 'per_class_metrics' in report
        assert 'class_distribution' in report
        assert report['metadata']['total_images'] == 1
    
    def test_predictions_export_and_load(self):
        """Test predictions export and load functionality."""
        generator = ReportGenerator()
        predictions = [
            [Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, class_id=0, class_name='Mouse_bite')],
            [Detection(bbox=(0.3, 0.3, 0.1, 0.1), confidence=0.8, class_id=1, class_name='Open_circuit')]
        ]
        image_names = ['img1.jpg', 'img2.jpg']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "predictions.json")
            generator.export_predictions(predictions, image_names, output_path)
            assert os.path.exists(output_path)
            
            # Test loading
            loaded_predictions = generator.load_predictions(output_path)
            assert len(loaded_predictions) == len(predictions)
            assert len(loaded_predictions[0]) == len(predictions[0])
            assert loaded_predictions[0][0].class_name == predictions[0][0].class_name


class TestEvaluationPipeline:
    """Test evaluation pipeline functionality."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = EvaluationPipeline(output_dir=temp_dir)
            assert pipeline.evaluator.iou_threshold == 0.5
            assert Path(temp_dir).exists()
    
    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        predictions = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, class_id=0, class_name='Mouse_bite')]]
        ground_truths = [[Detection(bbox=(0.5, 0.5, 0.2, 0.2), confidence=1.0, class_id=0, class_name='Mouse_bite')]]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = EvaluationPipeline(output_dir=temp_dir)
            results = pipeline.run_complete_evaluation(
                predictions, ground_truths,
                model_name='TestModel',
                dataset_name='TestDataset'
            )
            
            assert 'metrics' in results
            assert 'detailed_report' in results
            assert 'file_paths' in results
            
            # Check that files were created
            for file_path in results['file_paths'].values():
                if file_path:
                    assert os.path.exists(file_path)