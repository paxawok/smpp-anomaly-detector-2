import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from .base import BaseAutoencoder

class SMPPVAE(BaseAutoencoder):
    """Variational Autoencoder для SMPP аномалій"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.input_dim = config['architecture']['input_dim']
        self.latent_dim = config['architecture']['encoding_dim']
        self.hidden_dims = config['architecture']['hidden_dims']
        self.dropout_rate = config['architecture']['dropout_rate']
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Latent space
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        
        # Decoder
        self.decoder = self._build_decoder()
        
        self._init_weights()
    
    def _build_encoder(self) -> nn.Sequential:
        """Побудова encoder для VAE"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Побудова decoder для VAE"""
        layers = []
        hidden_dims_reversed = [self.latent_dim] + list(reversed(self.hidden_dims))
        
        for i in range(len(hidden_dims_reversed) - 1):
            layers.extend([
                nn.Linear(hidden_dims_reversed[i], hidden_dims_reversed[i+1]),
                nn.BatchNorm1d(hidden_dims_reversed[i+1]),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
        
        layers.extend([
            nn.Linear(hidden_dims_reversed[-1], self.input_dim),
            nn.Sigmoid()  # Для нормалізованих даних [0, 1]
        ])
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Ініціалізація ваг"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    """def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #Кодування у латентний простір
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar"""
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Кодування у латентний простір для сумісності"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        return z  # Повертаємо тільки z для сумісності з base класом
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Декодування з латентного простору"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass VAE"""
        z, mu, logvar = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z, mu, logvar
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """VAE loss = Reconstruction loss + KL divergence"""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return {
            'loss': recon_loss + kl_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Обчислення помилки реконструкції для VAE"""
        with torch.no_grad():
            reconstructed, _, _, _ = self.forward(x)
            return torch.mean((x - reconstructed) ** 2, dim=1)