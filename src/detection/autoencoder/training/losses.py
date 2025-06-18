import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class CombinedLoss(nn.Module):
    """Комбінована loss функція: MSE + sparsity"""
    
    def __init__(self, beta: float = 0.01, sparsity_target: float = 0.05):
        super().__init__()
        self.beta = beta
        self.sparsity_target = sparsity_target
        self.mse = nn.MSELoss()
    
    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, 
                encoded: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss
        mse_loss = self.mse(outputs, inputs)
        
        # Sparsity loss (KL divergence)
        rho = self.sparsity_target
        rho_hat = torch.mean(encoded, dim=0)
        
        # Стабільна версія KL divergence
        kl_div = torch.sum(
            rho * torch.log(rho / (rho_hat + 1e-8)) + 
            (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-8))
        )
        
        return mse_loss + self.beta * kl_div


class VAELoss(nn.Module):
    """Loss для Variational Autoencoder"""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, 
                mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class FocalMSELoss(nn.Module):
    """Focal MSE Loss для роботи з дисбалансом"""
    
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = (outputs - targets) ** 2
        focal_weight = (1 + mse) ** self.gamma
        return torch.mean(focal_weight * mse)