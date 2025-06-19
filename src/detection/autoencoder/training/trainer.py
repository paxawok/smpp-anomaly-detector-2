import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class Trainer:
    """Основний клас для навчання моделей"""
    
    def __init__(self, model: nn.Module, config: Dict, device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Оптимізатор
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss
        self.criterion = nn.MSELoss() 
        
        # Історія
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Callbacks
        self.callbacks = []
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Створення оптимізатора"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Створення scheduler"""
        if self.config.get('lr_scheduler', True):
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
        return None
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Dict:
        """Основний цикл навчання"""
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self._validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self.history)
            
            # Logging
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Один epoch навчання"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, encoded = self.model(inputs)
            
            # Loss
            loss = self.criterion(outputs, inputs)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
        
        return total_loss / len(train_loader.dataset)
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Валідація"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                outputs, encoded = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                total_loss += loss.item() * inputs.size(0)
        
        return total_loss / len(val_loader.dataset)
    
    def add_callback(self, callback):
        """Додавання callback"""
        self.callbacks.append(callback)