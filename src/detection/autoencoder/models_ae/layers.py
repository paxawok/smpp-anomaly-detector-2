import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ResidualBlock(nn.Module):
    """Residual block з batch normalization"""
    
    def __init__(self, in_features: int, out_features: int, 
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else None
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Отримання activation функції"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'gelu': nn.GELU(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            residual = self.skip(residual)
        
        out += residual
        out = self.activation(out)
        
        return out


class AttentionLayer(nn.Module):
    """Self-attention layer"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(attn_out + x)
        
        return out.squeeze(1)  # (batch, features)


class GatedLayer(nn.Module):
    """Gated linear unit"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out, gate = out.chunk(2, dim=-1)
        return out * torch.sigmoid(gate)