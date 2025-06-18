import numpy as np
import torch
from typing import Dict, Optional
from pathlib import Path

class Callback:
    """Базовий клас для callbacks"""
    
    def on_epoch_end(self, epoch: int, logs: Dict):
        pass
    
    def on_train_begin(self, logs: Dict):
        pass
    
    def on_train_end(self, logs: Dict):
        pass


class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def on_epoch_end(self, epoch: int, logs: Dict):
        current = logs[self.monitor][-1]
        
        if self.best_score is None:
            self.best_score = current
        elif self._is_improvement(current):
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {epoch + 1}")
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self.best_score - self.min_delta
        else:
            return current > self.best_score + self.min_delta


class ModelCheckpoint(Callback):
    """Збереження найкращої моделі"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 mode: str = 'min', save_best_only: bool = True):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
    def on_epoch_end(self, epoch: int, logs: Dict):
        current = logs[self.monitor][-1]
        
        if not self.save_best_only:
            self._save_checkpoint(epoch)
        elif self.best_score is None or self._is_improvement(current):
            self.best_score = current
            self._save_checkpoint(epoch)
            print(f"Saved best model at epoch {epoch + 1}")
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self.best_score
        else:
            return current > self.best_score
    
    def _save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'best_score': self.best_score,
            'monitor': self.monitor
        }
        torch.save(checkpoint, self.filepath)