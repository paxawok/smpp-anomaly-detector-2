import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List

class BaseAutoencoder(nn.Module, ABC):
    """Базовий клас для всіх автокодувальників"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Кодування вхідних даних"""
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Декодування латентного представлення"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: повертає (reconstructed, encoded)"""
        pass
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Обчислення помилки реконструкції"""
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            return torch.mean((x - reconstructed) ** 2, dim=1)