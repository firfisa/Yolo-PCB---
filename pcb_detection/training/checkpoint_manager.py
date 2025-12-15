"""
Checkpoint manager implementation.
"""

import os
import json
import shutil
import logging
from typing import Dict, Optional, Any
import torch


class CheckpointManager:
    """Manager for model checkpoints and best model saving."""
    
    def __init__(self, checkpoint_dir: str, save_best: bool = True, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            max_checkpoints: Maximum number of regular checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_best = save_best
        self.max_checkpoints = max_checkpoints
        self.best_metric = 0.0
        self.best_checkpoint_path = None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('CheckpointManager')
        
        # Track saved checkpoints
        self.checkpoint_history = []
        
    def save_checkpoint(self, model, optimizer, epoch: int, 
                       metrics: Dict[str, float], 
                       is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        # Save regular checkpoint
        checkpoint_filename = f"checkpoint_epoch_{epoch:03d}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append(checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model if applicable
        if is_best and self.save_best:
            current_metric = metrics.get('map', metrics.get('loss', 0.0))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                
                # Copy checkpoint to best model
                shutil.copy2(checkpoint_path, best_path)
                self.best_checkpoint_path = best_path
                
                self.logger.info(f"Saved best model: {best_path} (metric: {current_metric:.4f})")
                
                # Save best model metadata
                self._save_best_model_info(epoch, metrics)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
        
    def load_checkpoint(self, model, optimizer, 
                       checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update best metric
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'best_metric': self.best_metric
        }
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")
        
        return checkpoint_info
        
    def get_best_checkpoint_path(self) -> Optional[str]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            return best_path
        return None
        
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoint_history:
            # Look for existing checkpoints in directory
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                              if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
            if checkpoint_files:
                # Sort by epoch number
                checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                return os.path.join(self.checkpoint_dir, checkpoint_files[-1])
            return None
        
        return self.checkpoint_history[-1] if self.checkpoint_history else None
        
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoint_files = []
        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('.pth'):
                    checkpoint_files.append(os.path.join(self.checkpoint_dir, filename))
        
        return sorted(checkpoint_files)
        
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                if checkpoint_path in self.checkpoint_history:
                    self.checkpoint_history.remove(checkpoint_path)
                self.logger.info(f"Deleted checkpoint: {checkpoint_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_path}: {e}")
        
        return False
        
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a checkpoint without loading it.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information or None if failed
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return {
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'best_metric': checkpoint.get('best_metric', 0.0),
                'file_size': os.path.getsize(checkpoint_path)
            }
        except Exception as e:
            self.logger.error(f"Failed to read checkpoint info: {e}")
            return None
            
    def _save_best_model_info(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save metadata about the best model.
        
        Args:
            epoch: Epoch of best model
            metrics: Metrics of best model
        """
        info = {
            'epoch': epoch,
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        info_path = os.path.join(self.checkpoint_dir, "best_model_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Remove oldest checkpoints
            checkpoints_to_remove = self.checkpoint_history[:-self.max_checkpoints]
            
            for checkpoint_path in checkpoints_to_remove:
                if os.path.exists(checkpoint_path):
                    # Don't delete if it's the best model
                    if checkpoint_path != self.best_checkpoint_path:
                        self.delete_checkpoint(checkpoint_path)
            
            # Update history
            self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]
            
    def resume_training(self, model, optimizer) -> Optional[Dict[str, Any]]:
        """
        Resume training from the latest checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            
        Returns:
            Checkpoint information or None if no checkpoint found
        """
        latest_checkpoint = self.get_latest_checkpoint_path()
        if latest_checkpoint:
            return self.load_checkpoint(model, optimizer, latest_checkpoint)
        
        self.logger.info("No checkpoint found for resuming training")
        return None