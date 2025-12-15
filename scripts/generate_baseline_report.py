#!/usr/bin/env python3
"""
Baseline Evaluation Report Generator for PCB Defect Detection.

This script generates comprehensive baseline evaluation reports including
performance analysis, visualizations, and recommendations for optimization.
"""

import os
import sys
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcb_detection.utils.file_utils import FileUtils


class BaselineReportGenerator:
    """Generator for comprehensive baseline evaluation reports."""
    
    def __init__(self, results_file: str, output_dir: str = "outputs/baseline/reports"):
        """
        Initialize report generator.
        
        Args:
            results_file: Path to baseline results JSON file
            output_dir: Output directory for reports
        """
        self.results_file = results_file
        self.output_dir = output_dir
        self.results = self._load_results()
        
        # Create output directory
        FileUtils.ensure_dir(output_dir)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _load_results(self) -> Dict[str, Any]:
        """Load baseline results from JSON file."""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load results from {self.results_file}: {e}")
            
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive baseline evaluation report.
        
        Returns:
            Path to generated report file
        """
        print("Generating comprehensive baseline evaluation report...")
        
        # Generate individual components
        performance_chart = self._create_performance_visualization()
        class_comparison_chart = self._create_class_comparison_chart()
        metrics_table = self._create_metrics_table()
        analysis_text = self._generate_performance_analysis()
        recommendations = self._generate_optimization_recommendations()
        
        # Create HTML report
        html_report = self._create_html_report(
            performance_chart, class_comparison_chart, metrics_table,
            analysis_text, recommendations
        )
        
        # Save HTML report
        report_file = os.path.join(self.output_dir, 'baseline_evaluation_report.html')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
            
        # Generate additional formats
        self._create_markdown_report()
        self._create_pdf_summary()
        
        print(f"✓ Comprehensive report generated: {report_file}")
        return report_file
        
    def _create_performance_visualization(self) -> str:
        """Create performance visualization chart."""
        
        # Extract metrics
        evaluation = self.results.get('evaluation', {})
        map_score = evaluation.get('map_50', 0.0)
        precision = evaluation.get('precision', 0.0)
        recall = evaluation.get('recall', 0.0)
        f1_score = evaluation.get('f1_score', 0.0)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance metrics bar chart
        metrics = ['mAP@0.5', 'Precision', 'Recall', 'F1-Score']
        values = [map_score, precision, recall, f1_score]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_title('Baseline Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, max(0.1, max(values) * 1.2))
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
                    
        # mAP range comparison
        expected_min, expected_max = 0.005, 0.01
        ax2.barh(['Expected Range'], [expected_max - expected_min], 
                left=[expected_min], color='lightgray', alpha=0.7, label='Expected Range')
        ax2.barh(['Actual mAP'], [0.001], left=[map_score], 
                color='#FF6B6B', alpha=0.8, label='Actual mAP')
        
        ax2.set_title('mAP vs Expected Baseline Range', fontsize=14, fontweight='bold')
        ax2.set_xlabel('mAP@0.5', fontsize=12)
        ax2.set_xlim(0, 0.015)
        ax2.legend()
        
        # Add status indicator
        status = "✓ WITHIN RANGE" if expected_min <= map_score <= expected_max else "⚠ OUT OF RANGE"
        ax2.text(0.012, 0.5, status, fontsize=12, fontweight='bold',
                color='green' if expected_min <= map_score <= expected_max else 'red')
        
        plt.tight_layout()
        
        # Save chart
        chart_file = os.path.join(self.output_dir, 'performance_metrics.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_file
        
    def _create_class_comparison_chart(self) -> str:
        """Create per-class performance comparison chart."""
        
        evaluation = self.results.get('evaluation', {})
        ap_per_class = evaluation.get('ap_per_class', {})
        
        if not ap_per_class:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        classes = list(ap_per_class.keys())
        ap_values = list(ap_per_class.values())
        
        # Create horizontal bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = ax.barh(classes, ap_values, color=colors, alpha=0.8)
        
        ax.set_title('Per-Class Average Precision (AP)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Average Precision', fontsize=12)
        ax.set_xlim(0, max(0.03, max(ap_values) * 1.2))
        
        # Add value labels
        for bar, value in zip(bars, ap_values):
            width = bar.get_width()
            ax.text(width + 0.0005, bar.get_y() + bar.get_height()/2.,
                   f'{value:.4f}', ha='left', va='center', fontsize=10)
                   
        # Add average line
        avg_ap = np.mean(ap_values)
        ax.axvline(avg_ap, color='red', linestyle='--', alpha=0.7, 
                  label=f'Average: {avg_ap:.4f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Save chart
        chart_file = os.path.join(self.output_dir, 'class_performance.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_file
        
    def _create_metrics_table(self) -> str:
        """Create detailed metrics table."""
        
        evaluation = self.results.get('evaluation', {})
        baseline_verification = self.results.get('baseline_verification', {})
        
        # Create metrics data
        metrics_data = [
            ['Overall Performance', '', ''],
            ['mAP@0.5', f"{evaluation.get('map_50', 0.0):.6f}", 'Primary evaluation metric'],
            ['Precision', f"{evaluation.get('precision', 0.0):.6f}", 'Positive prediction accuracy'],
            ['Recall', f"{evaluation.get('recall', 0.0):.6f}", 'True positive detection rate'],
            ['F1-Score', f"{evaluation.get('f1_score', 0.0):.6f}", 'Harmonic mean of precision and recall'],
            ['', '', ''],
            ['Baseline Verification', '', ''],
            ['Expected Range', '0.005 - 0.01', 'Target range for untrained model'],
            ['Within Range', 'Yes' if baseline_verification.get('within_range', False) else 'No', 'Baseline validation status'],
            ['Status', baseline_verification.get('status', 'UNKNOWN'), 'Overall baseline status'],
        ]
        
        # Add per-class metrics
        ap_per_class = evaluation.get('ap_per_class', {})
        if ap_per_class:
            metrics_data.extend([['', '', ''], ['Per-Class Performance', '', '']])
            for class_name, ap_value in ap_per_class.items():
                metrics_data.append([f'{class_name} AP', f'{ap_value:.6f}', 'Class-specific average precision'])
                
        # Create HTML table
        table_html = '<table class="metrics-table">\n'
        table_html += '<thead><tr><th>Metric</th><th>Value</th><th>Description</th></tr></thead>\n'
        table_html += '<tbody>\n'
        
        for row in metrics_data:
            if row[0] == '' and row[1] == '' and row[2] == '':
                table_html += '<tr class="separator"><td colspan="3"></td></tr>\n'
            elif row[1] == '' and row[2] == '':
                table_html += f'<tr class="section-header"><td colspan="3"><strong>{row[0]}</strong></td></tr>\n'
            else:
                table_html += f'<tr><td>{row[0]}</td><td><strong>{row[1]}</strong></td><td>{row[2]}</td></tr>\n'
                
        table_html += '</tbody></table>\n'
        
        return table_html
        
    def _generate_performance_analysis(self) -> str:
        """Generate performance analysis text."""
        
        evaluation = self.results.get('evaluation', {})
        baseline_verification = self.results.get('baseline_verification', {})
        
        map_score = evaluation.get('map_50', 0.0)
        within_range = baseline_verification.get('within_range', False)
        
        analysis = f"""
        <h3>Performance Analysis</h3>
        
        <h4>Baseline Validation</h4>
        <p>The baseline model achieved an mAP@0.5 of <strong>{map_score:.6f}</strong>, which is 
        {'<span class="success">within</span>' if within_range else '<span class="warning">outside</span>'} 
        the expected baseline range of 0.005-0.01 for an untrained YOLO model.</p>
        
        <h4>Key Observations</h4>
        <ul>
        """
        
        # Add specific observations based on results
        if within_range:
            analysis += "<li>✓ The baseline performance meets expectations for an untrained model</li>"
        else:
            if map_score < 0.005:
                analysis += "<li>⚠ Performance is below expected baseline - may indicate implementation issues</li>"
            else:
                analysis += "<li>✓ Performance exceeds baseline expectations - good starting point</li>"
                
        # Analyze per-class performance
        ap_per_class = evaluation.get('ap_per_class', {})
        if ap_per_class:
            best_class = max(ap_per_class.items(), key=lambda x: x[1])
            worst_class = min(ap_per_class.items(), key=lambda x: x[1])
            
            analysis += f"<li>Best performing class: <strong>{best_class[0]}</strong> (AP: {best_class[1]:.4f})</li>"
            analysis += f"<li>Worst performing class: <strong>{worst_class[0]}</strong> (AP: {worst_class[1]:.4f})</li>"
            
            # Check for class imbalance indicators
            ap_values = list(ap_per_class.values())
            ap_std = np.std(ap_values)
            if ap_std > 0.005:
                analysis += "<li>⚠ High variance in per-class performance suggests potential class imbalance</li>"
                
        analysis += """
        </ul>
        
        <h4>Baseline Establishment</h4>
        <p>This baseline serves as the reference point for measuring improvements from advanced techniques including:</p>
        <ul>
            <li>CBAM attention mechanisms</li>
            <li>Advanced loss functions (Focal Loss, CIoU Loss)</li>
            <li>Data augmentation strategies (Mosaic, Copy-Paste, MixUp)</li>
            <li>Multi-scale training</li>
            <li>Model ensemble techniques</li>
        </ul>
        """
        
        return analysis
        
    def _generate_optimization_recommendations(self) -> str:
        """Generate optimization recommendations."""
        
        evaluation = self.results.get('evaluation', {})
        map_score = evaluation.get('map_50', 0.0)
        
        recommendations = """
        <h3>Optimization Recommendations</h3>
        
        <h4>Immediate Next Steps</h4>
        <ol>
            <li><strong>Implement Basic Training</strong>
                <ul>
                    <li>Train YOLOv8n model with standard configuration</li>
                    <li>Use basic data augmentation (rotation, scaling, color jittering)</li>
                    <li>Target: mAP > 0.15 (15x baseline improvement)</li>
                </ul>
            </li>
            
            <li><strong>Apply Advanced Techniques</strong>
                <ul>
                    <li>Integrate CBAM attention mechanism for small defect detection</li>
                    <li>Use Focal Loss to handle class imbalance</li>
                    <li>Implement CIoU Loss for better bounding box regression</li>
                    <li>Target: mAP > 0.30 (30x baseline improvement)</li>
                </ul>
            </li>
            
            <li><strong>Advanced Data Augmentation</strong>
                <ul>
                    <li>Implement Mosaic augmentation for multi-scale training</li>
                    <li>Use Copy-Paste augmentation to increase defect diversity</li>
                    <li>Apply MixUp for improved generalization</li>
                    <li>Target: mAP > 0.40 (40x baseline improvement)</li>
                </ul>
            </li>
        </ol>
        
        <h4>Performance Targets</h4>
        <div class="performance-targets">
            <div class="target-item">
                <strong>Basic Training:</strong> mAP 0.15-0.25 (15-25x improvement)
            </div>
            <div class="target-item">
                <strong>Advanced Techniques:</strong> mAP 0.30-0.40 (30-40x improvement)
            </div>
            <div class="target-item">
                <strong>Full Optimization:</strong> mAP 0.45-0.55 (45-55x improvement)
            </div>
        </div>
        """
        
        # Add specific recommendations based on current performance
        if map_score < 0.005:
            recommendations += """
            <h4>⚠ Critical Issues to Address</h4>
            <ul>
                <li>Verify model implementation and data loading</li>
                <li>Check annotation format and class mapping</li>
                <li>Ensure proper loss function configuration</li>
            </ul>
            """
            
        recommendations += """
        <h4>Monitoring and Evaluation</h4>
        <ul>
            <li>Track mAP improvements at each optimization stage</li>
            <li>Monitor per-class performance for balanced improvement</li>
            <li>Evaluate inference speed vs accuracy trade-offs</li>
            <li>Generate visualization comparisons for each milestone</li>
        </ul>
        """
        
        return recommendations
        
    def _create_html_report(self, performance_chart: str, class_chart: str, 
                          metrics_table: str, analysis: str, recommendations: str) -> str:
        """Create comprehensive HTML report."""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PCB Defect Detection - Baseline Evaluation Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #4ECDC4;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #2C3E50;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    color: #7F8C8D;
                    margin: 10px 0 0 0;
                    font-size: 1.1em;
                }}
                .section {{
                    margin: 30px 0;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 14px;
                }}
                .metrics-table th {{
                    background-color: #4ECDC4;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                .metrics-table td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #ddd;
                }}
                .metrics-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metrics-table .section-header {{
                    background-color: #ECF0F1;
                    font-weight: bold;
                }}
                .metrics-table .separator {{
                    height: 10px;
                }}
                .success {{
                    color: #27AE60;
                    font-weight: bold;
                }}
                .warning {{
                    color: #E74C3C;
                    font-weight: bold;
                }}
                .performance-targets {{
                    background-color: #F8F9FA;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #4ECDC4;
                }}
                .target-item {{
                    margin: 10px 0;
                    padding: 10px;
                    background-color: white;
                    border-radius: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #7F8C8D;
                }}
                h3 {{
                    color: #2C3E50;
                    border-bottom: 2px solid #4ECDC4;
                    padding-bottom: 5px;
                }}
                h4 {{
                    color: #34495E;
                    margin-top: 25px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>PCB Defect Detection</h1>
                    <p>Baseline Evaluation Report</p>
                    <p>Generated on {timestamp}</p>
                </div>
                
                <div class="section">
                    <h3>Executive Summary</h3>
                    <p>This report presents the baseline performance evaluation for the PCB defect detection system. 
                    The baseline establishes the reference point for measuring improvements from advanced deep learning 
                    techniques including attention mechanisms, advanced loss functions, and data augmentation strategies.</p>
                </div>
                
                <div class="section">
                    <h3>Performance Metrics</h3>
                    <div class="chart-container">
                        <img src="{os.path.basename(performance_chart)}" alt="Performance Metrics Chart">
                    </div>
                </div>
                
                <div class="section">
                    <h3>Per-Class Performance</h3>
                    <div class="chart-container">
                        <img src="{os.path.basename(class_chart) if class_chart else ''}" alt="Class Performance Chart">
                    </div>
                </div>
                
                <div class="section">
                    <h3>Detailed Metrics</h3>
                    {metrics_table}
                </div>
                
                <div class="section">
                    {analysis}
                </div>
                
                <div class="section">
                    {recommendations}
                </div>
                
                <div class="footer">
                    <p>PCB Defect Detection System - Baseline Evaluation Report</p>
                    <p>Generated automatically by the evaluation system</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    def _create_markdown_report(self) -> str:
        """Create markdown version of the report."""
        
        evaluation = self.results.get('evaluation', {})
        baseline_verification = self.results.get('baseline_verification', {})
        
        markdown = f"""# PCB Defect Detection - Baseline Evaluation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the baseline performance evaluation for the PCB defect detection system using an untrained YOLO model. The baseline establishes the reference point for measuring improvements from advanced techniques.

## Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| mAP@0.5 | {evaluation.get('map_50', 0.0):.6f} | {'✓ Within Range' if baseline_verification.get('within_range', False) else '⚠ Out of Range'} |
| Precision | {evaluation.get('precision', 0.0):.6f} | - |
| Recall | {evaluation.get('recall', 0.0):.6f} | - |
| F1-Score | {evaluation.get('f1_score', 0.0):.6f} | - |

## Per-Class Performance

"""
        
        ap_per_class = evaluation.get('ap_per_class', {})
        if ap_per_class:
            markdown += "| Class | Average Precision |\n|-------|------------------|\n"
            for class_name, ap_value in ap_per_class.items():
                markdown += f"| {class_name} | {ap_value:.6f} |\n"
                
        markdown += f"""

## Baseline Verification

- **Expected Range**: 0.005 - 0.01
- **Actual mAP**: {evaluation.get('map_50', 0.0):.6f}
- **Status**: {baseline_verification.get('status', 'UNKNOWN')}

## Next Steps

1. **Basic Training**: Implement standard YOLO training (Target: mAP > 0.15)
2. **Advanced Techniques**: Add CBAM attention and advanced losses (Target: mAP > 0.30)
3. **Data Augmentation**: Implement Mosaic, Copy-Paste, MixUp (Target: mAP > 0.40)

## Performance Targets

- **Basic Training**: 15-25x improvement over baseline
- **Advanced Techniques**: 30-40x improvement over baseline  
- **Full Optimization**: 45-55x improvement over baseline

---

*Report generated automatically by the PCB defect detection evaluation system.*
"""
        
        # Save markdown report
        markdown_file = os.path.join(self.output_dir, 'baseline_report.md')
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
            
        print(f"✓ Markdown report generated: {markdown_file}")
        return markdown_file
        
    def _create_pdf_summary(self) -> str:
        """Create PDF summary (placeholder - would need additional libraries)."""
        
        # Create a simple text summary for now
        summary_file = os.path.join(self.output_dir, 'baseline_summary.txt')
        
        evaluation = self.results.get('evaluation', {})
        baseline_verification = self.results.get('baseline_verification', {})
        
        summary = f"""PCB DEFECT DETECTION - BASELINE EVALUATION SUMMARY
{'=' * 60}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE METRICS:
- mAP@0.5: {evaluation.get('map_50', 0.0):.6f}
- Precision: {evaluation.get('precision', 0.0):.6f}
- Recall: {evaluation.get('recall', 0.0):.6f}
- F1-Score: {evaluation.get('f1_score', 0.0):.6f}

BASELINE VERIFICATION:
- Expected Range: 0.005 - 0.01
- Status: {baseline_verification.get('status', 'UNKNOWN')}
- Within Range: {'Yes' if baseline_verification.get('within_range', False) else 'No'}

PER-CLASS PERFORMANCE:
"""
        
        ap_per_class = evaluation.get('ap_per_class', {})
        for class_name, ap_value in ap_per_class.items():
            summary += f"- {class_name}: {ap_value:.6f}\n"
            
        summary += f"""
OPTIMIZATION TARGETS:
- Basic Training: mAP > 0.15 (15x improvement)
- Advanced Techniques: mAP > 0.30 (30x improvement)  
- Full Optimization: mAP > 0.45 (45x improvement)

NEXT STEPS:
1. Implement basic YOLO training
2. Add CBAM attention mechanism
3. Use advanced loss functions (Focal, CIoU)
4. Apply data augmentation (Mosaic, Copy-Paste, MixUp)
5. Evaluate performance improvements at each stage

{'=' * 60}
"""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        print(f"✓ Text summary generated: {summary_file}")
        return summary_file


def main():
    """Main function to generate baseline evaluation report."""
    
    # Check if baseline results exist
    results_file = "outputs/baseline/results/baseline_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Baseline results not found: {results_file}")
        print("Please run the baseline evaluation first:")
        print("  python simple_baseline.py")
        return 1
        
    try:
        # Generate comprehensive report
        generator = BaselineReportGenerator(results_file)
        report_file = generator.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("BASELINE EVALUATION REPORT GENERATED")
        print("=" * 60)
        print(f"📊 HTML Report: {report_file}")
        print(f"📝 Markdown Report: outputs/baseline/reports/baseline_report.md")
        print(f"📄 Text Summary: outputs/baseline/reports/baseline_summary.txt")
        print(f"📈 Charts: outputs/baseline/reports/*.png")
        
        print("\n🚀 Next Steps:")
        print("  1. Review the comprehensive report")
        print("  2. Implement basic YOLO training")
        print("  3. Apply advanced optimization techniques")
        print("  4. Compare results against this baseline")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())