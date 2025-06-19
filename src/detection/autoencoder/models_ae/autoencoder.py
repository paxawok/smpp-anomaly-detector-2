import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from .base import BaseAutoencoder
from .layers import ResidualBlock

class ImprovedSMPPAutoencoder(BaseAutoencoder):
    """Покращений Autoencoder з attention механізмом"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.input_dim = config['architecture']['input_dim']
        self.encoding_dim = config['architecture']['encoding_dim']
        self.hidden_dims = config['architecture']['hidden_dims']
        self.dropout_rate = config['architecture']['dropout_rate']
        self.use_attention = config['architecture'].get('use_attention', True)
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dims[-1],
                num_heads=config['architecture'].get('attention_heads', 4),
                dropout=self.dropout_rate,
                batch_first=True
            )
        
        # Bottleneck
        self.encoding_layer = nn.Linear(self.hidden_dims[-1], self.encoding_dim)
        self.decoding_layer = nn.Linear(self.encoding_dim, self.hidden_dims[-1])
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Residual connection
        if self.hidden_dims[0] == self.input_dim:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _build_encoder(self) -> nn.Sequential:
        """Побудова encoder"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Побудова decoder"""
        layers = []
        hidden_dims_reversed = list(reversed(self.hidden_dims))
        
        for i in range(len(hidden_dims_reversed) - 1):
            layers.extend([
                nn.Linear(hidden_dims_reversed[i], hidden_dims_reversed[i+1]),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
        
        layers.append(nn.Linear(hidden_dims_reversed[-1], self.input_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Ініціалізація ваг"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Кодування"""
        encoded_features = self.encoder(x)
        
        return self.encoding_layer(encoded_features)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Декодування"""
        decoded = self.decoding_layer(z)
        reconstructed = self.decoder(decoded)
        
        if hasattr(self, 'residual_weight'):
            return reconstructed
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        
        if hasattr(self, 'residual_weight'):
            reconstructed = reconstructed + self.residual_weight * x
        
        return reconstructed, encoded