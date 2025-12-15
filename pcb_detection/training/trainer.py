"""
Training manager implementation.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from ..core.types import TrainingConfig
from .checkpoint_manager import CheckpointManager


class Trainer:
    """Training manager for PCB defect detection models."""
    
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, 
                 config: TrainingConfig, checkpoint_dir: str = "checkpoints"):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'learning_rate': []
        }
        
        # Multi-scale training setup
        self.image_sizes = self._setup_multiscale_sizes()
        self.scale_change_frequency = 10  # Change scale every 10 epochs
        
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        self.logger.info(f"Using device: {device}")
        return device
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with different parameter groups."""
        # Different learning rates for different parts of the model
        param_groups = []
        
        # Backbone parameters (lower learning rate)
        if hasattr(self.model, 'backbone'):
            param_groups.append({
                'params': self.model.backbone.parameters(),
                'lr': self.config.learning_rate * 0.1
            })
        
        # Head parameters (normal learning rate)
        if hasattr(self.model, 'head'):
            param_groups.append({
                'params': self.model.head.parameters(),
                'lr': self.config.learning_rate
            })
        
        # If no specific parts, use all parameters
        if not param_groups:
            param_groups = [{'params': self.model.parameters(), 'lr': self.config.learning_rate}]
        
        optimizer = optim.AdamW(param_groups, weight_decay=0.0005)
        return optimizer
        
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        # Cosine annealing with warm restarts
        T_0 = max(1, self.config.epochs // 4)  # Ensure T_0 is at least 1
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=T_0,  # First restart after 1/4 of epochs
            T_mult=2,  # Double the period after each restart
            eta_min=self.config.learning_rate * 0.01
        )
        return scheduler
        
    def _setup_logging(self) -> logging.Logger:
        """Setup training logger."""
        logger = logging.getLogger('PCBTrainer')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(self.checkpoint_dir, 'training.log')
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def _setup_multiscale_sizes(self) -> List[int]:
        """Setup multi-scale training image sizes."""
        base_size = self.config.image_size
        # Generate sizes around base size (±20%)
        sizes = [
            int(base_size * 0.8),   # 80% of base
            int(base_size * 0.9),   # 90% of base
            base_size,              # Base size
            int(base_size * 1.1),   # 110% of base
            int(base_size * 1.2)    # 120% of base
        ]
        # Ensure sizes are multiples of 32 (YOLO requirement)
        sizes = [size - (size % 32) for size in sizes]
        return sizes
        
    def train(self) -> Dict[str, Any]:
        """
        Execute complete training loop.
        
        Returns:
            Training results and metrics
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info(f"Multi-scale sizes: {self.image_sizes}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Multi-scale training: change image size periodically
            if epoch % self.scale_change_frequency == 0:
                current_size = self.image_sizes[epoch // self.scale_change_frequency % len(self.image_sizes)]
                self._update_image_size(current_size)
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_map'].append(val_metrics.get('map', 0.0))
            self.training_history['learning_rate'].append(current_lr)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val mAP: {val_metrics.get('map', 0.0):.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Check if this is the best model
            current_metric = val_metrics.get('map', 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                patience_counter = 0
                self.logger.info(f"New best model! mAP: {current_metric:.4f}")
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_period == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'training_time': training_time,
            'epochs_trained': self.current_epoch + 1
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Execute one training epoch.
        
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.model.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 50 == 0:
                self.logger.debug(
                    f"Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
        
    def validate(self) -> Dict[str, float]:
        """
        Execute validation.
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.model.compute_loss(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        # TODO: Add mAP calculation when evaluator is implemented
        # For now, return loss only
        return {'loss': avg_loss}
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model
        """
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best
        )
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information
        """
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path
        )
        
        # Restore training state
        self.current_epoch = checkpoint_info.get('epoch', 0)
        self.best_metric = checkpoint_info.get('best_metric', 0.0)
        
        self.logger.info(f"Resumed training from epoch {self.current_epoch}")
        return checkpoint_info
        
    def _update_image_size(self, new_size: int) -> None:
        """
        Update image size for multi-scale training.
        
        Args:
            new_size: New image size
        """
        self.logger.info(f"Updating image size to {new_size}x{new_size}")
        # Note: This would typically require updating the data loader
        # For now, we just log the change
        pass