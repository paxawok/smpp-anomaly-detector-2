import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Завантаження та підготовка даних"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
    def load_csv(self, filepath: str, encoding: str = 'utf-8-sig') -> pd.DataFrame:
        """Завантаження CSV файлу"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, encoding=encoding)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def prepare_dataloaders(self, X: np.ndarray, batch_size: int = 256, 
                          validation_split: float = 0.2,
                          augment: bool = True) -> Tuple[TorchDataLoader, TorchDataLoader]:
        """Підготовка DataLoader для навчання"""
        # Train/val split
        X_train, X_val = train_test_split(X, test_size=validation_split, 
                                          random_state=42, shuffle=True)
        
        # Data augmentation
        if augment:
            noise_factor = 0.01
            X_train_augmented = X_train + noise_factor * np.random.randn(*X_train.shape)
            X_train = np.vstack([X_train, X_train_augmented])
            logger.info(f"Applied augmentation. New size: {X_train.shape[0]}")
        
        # Create datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        
        # Create dataloaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )
        
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )
        
        return train_loader, val_loader