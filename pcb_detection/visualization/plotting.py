"""
Plotting utilities for visualization and analysis.
"""

from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


class PlottingUtils:
    """Utility class for plotting and visualization."""
    
    @staticmethod
    def plot_training_curves(train_losses: List[float], 
                           val_losses: List[float],
                           save_path: Optional[str] = None) -> None:
        """
        Plot training and validation curves.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            save_path: Path to save plot
        """
        if not train_losses or not val_losses:
            raise ValueError("Loss lists cannot be empty")
            
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Training and Validation Loss Curves', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add minimum validation loss annotation
        min_val_idx = np.argmin(val_losses)
        min_val_loss = val_losses[min_val_idx]
        plt.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_idx + 1}',
                    xy=(min_val_idx + 1, min_val_loss),
                    xytext=(min_val_idx + 1 + len(epochs) * 0.1, min_val_loss + max(val_losses) * 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    @staticmethod
    def plot_class_distribution(class_counts: Dict[str, int],
                               save_path: Optional[str] = None) -> None:
        """
        Plot class distribution as a bar chart.
        
        Args:
            class_counts: Dictionary of class counts
            save_path: Path to save plot
        """
        if not class_counts:
            raise ValueError("Class counts dictionary cannot be empty")
            
        plt.figure(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Create bar plot with different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = plt.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                    str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.title('PCB Defect Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Defect Classes', fontsize=12)
        plt.ylabel('Number of Instances', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add total count
        total = sum(counts)
        plt.text(0.02, 0.98, f'Total Instances: {total}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray,
                             class_names: List[str],
                             save_path: Optional[str] = None,
                             normalize: bool = False) -> None:
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            confusion_matrix: Confusion matrix
            class_names: List of class names
            save_path: Path to save plot
            normalize: Whether to normalize the matrix
        """
        if confusion_matrix is None or len(confusion_matrix.shape) != 2:
            raise ValueError("Confusion matrix must be a 2D numpy array")
            
        if len(class_names) != confusion_matrix.shape[0]:
            raise ValueError("Number of class names must match confusion matrix dimensions")
            
        plt.figure(figsize=(10, 8))
        
        # Normalize if requested
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    @staticmethod
    def plot_detection_statistics(detections_per_image: List[int],
                                confidence_scores: List[float],
                                save_path: Optional[str] = None) -> None:
        """
        Plot detection statistics including detections per image and confidence distribution.
        
        Args:
            detections_per_image: Number of detections per image
            confidence_scores: All confidence scores
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot detections per image
        ax1.hist(detections_per_image, bins=max(1, max(detections_per_image) + 1), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Detections per Image Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Detections')
        ax1.set_ylabel('Number of Images')
        ax1.grid(alpha=0.3)
        
        # Add statistics
        mean_det = np.mean(detections_per_image)
        ax1.axvline(mean_det, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_det:.1f}')
        ax1.legend()
        
        # Plot confidence score distribution
        ax2.hist(confidence_scores, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.grid(alpha=0.3)
        
        # Add statistics
        mean_conf = np.mean(confidence_scores)
        ax2.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_conf:.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()