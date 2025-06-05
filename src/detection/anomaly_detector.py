"""
Anomaly Detector - Combines Isolation Forest and Autoencoder models
"""

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SMPPAutoencoder(nn.Module):
    """Autoencoder model definition"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, 
                 hidden_dims: list = None, dropout_rate: float = 0.2):
        super(SMPPAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetector:
    """Combined anomaly detector using Isolation Forest and Autoencoder"""
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize anomaly detector"""
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir
        
        # Model components
        self.isolation_forest = None
        self.isolation_forest_data = None
        self.autoencoder = None
        self.autoencoder_data = None
        
        # Feature names
        self.feature_names = None
        
        # Ensemble weights
        self.ensemble_weights = {
            'isolation_forest': 0.6,
            'autoencoder': 0.4
        }
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        # Load Isolation Forest
        self._load_isolation_forest()
        
        # Load Autoencoder
        self._load_autoencoder()
        
        if self.isolation_forest is None and self.autoencoder is None:
            self.logger.warning("No models loaded! Detector will use default thresholds.")
    
    def _load_isolation_forest(self):
        """Load Isolation Forest ensemble"""
        try:
            # Find latest Isolation Forest model
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.startswith('isolation_forest_ensemble_') and f.endswith('.pkl')]
            
            if model_files:
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join(self.models_dir, latest_model)
                
                self.logger.info(f"Loading Isolation Forest from {model_path}")
                self.isolation_forest_data = joblib.load(model_path)
                
                # Extract components
                self.isolation_forest = self.isolation_forest_data['models']
                self.feature_names = self.isolation_forest_data.get('feature_names', [])
                
                self.logger.info(f"Loaded Isolation Forest with {len(self.isolation_forest)} models")
            else:
                self.logger.warning("No Isolation Forest model found")
                
        except Exception as e:
            self.logger.error(f"Error loading Isolation Forest: {e}")
    
    def _load_autoencoder(self):
        """Load Autoencoder model"""
        try:
            model_path = os.path.join(self.models_dir, 'smpp_autoencoder.pth')
            
            if os.path.exists(model_path):
                self.logger.info(f"Loading Autoencoder from {model_path}")
                
                checkpoint = torch.load(model_path, map_location=device)
                
                # Initialize model
                config = checkpoint['model_config']
                self.autoencoder = SMPPAutoencoder(
                    input_dim=config['input_dim'],
                    encoding_dim=config['encoding_dim'],
                    hidden_dims=config['hidden_dims'],
                    dropout_rate=config['dropout_rate']
                ).to(device)
                
                # Load weights
                self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
                self.autoencoder.eval()
                
                # Store additional data
                self.autoencoder_data = {
                    'scaler': checkpoint['scaler'],
                    'threshold': checkpoint['threshold']
                }
                
                self.logger.info("Loaded Autoencoder successfully")
            else:
                self.logger.warning("No Autoencoder model found")
                
        except Exception as e:
            self.logger.error(f"Error loading Autoencoder: {e}")
    
    def extract_features(self, message_data: Dict) -> np.ndarray:
        """Extract features from message data"""
        # This is a simplified version - in production, use full FeatureExtractor
        features = []
        
        # Basic features
        message_text = message_data.get('message_text', '')
        source_addr = message_data.get('source_addr', '')
        
        # Message features
        features.append(len(message_text))  # message_length
        features.append(len(source_addr))   # source_addr_length
        features.append(1 if source_addr.isdigit() else 0)  # source_is_numeric
        
        # Add more features as needed...
        # In production, this should match the feature_names from training
        
        return np.array(features)
    
    def predict_isolation_forest(self, features: np.ndarray) -> Tuple[float, bool]:
        """Predict using Isolation Forest ensemble"""
        if self.isolation_forest is None:
            return 0.5, False
        
        try:
            # Get scaler
            scaler = self.isolation_forest_data['scaler']
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Get predictions from all models
            all_scores = []
            for model_name, model in self.isolation_forest.items():
                scores = model.score_samples(features_scaled)
                all_scores.append(scores[0])
            
            # Ensemble average
            ensemble_score = np.mean(all_scores)
            
            # Apply threshold
            threshold = self.isolation_forest_data['optimal_threshold']
            is_anomaly = ensemble_score < threshold
            
            # Convert to anomaly probability (0-1, higher = more anomalous)
            anomaly_score = 1 - (ensemble_score - threshold) / (2 * abs(threshold))
            anomaly_score = np.clip(anomaly_score, 0, 1)
            
            return float(anomaly_score), bool(is_anomaly)
            
        except Exception as e:
            self.logger.error(f"Error in Isolation Forest prediction: {e}")
            return 0.5, False
    
    def predict_autoencoder(self, features: np.ndarray) -> Tuple[float, bool]:
        """Predict using Autoencoder"""
        if self.autoencoder is None:
            return 0.5, False
        
        try:
            # Scale features
            scaler = self.autoencoder_data['scaler']
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            
            # Get reconstruction
            with torch.no_grad():
                reconstruction = self.autoencoder(features_tensor)
                
            # Calculate reconstruction error
            mse = torch.mean((features_tensor - reconstruction) ** 2).item()
            
            # Apply threshold
            threshold = self.autoencoder_data['threshold']
            is_anomaly = mse > threshold
            
            # Convert to anomaly probability
            anomaly_score = mse / (threshold * 2)
            anomaly_score = np.clip(anomaly_score, 0, 1)
            
            return float(anomaly_score), bool(is_anomaly)
            
        except Exception as e:
            self.logger.error(f"Error in Autoencoder prediction: {e}")
            return 0.5, False
    
    def predict(self, message_data: Dict) -> Dict:
        """Predict anomaly for a single message"""
        # Extract features
        if isinstance(message_data, dict):
            features = self.extract_features(message_data)
        else:
            # Assume it's already features array
            features = np.array(message_data)
        
        # Get predictions from both models
        if_score, if_anomaly = self.predict_isolation_forest(features)
        ae_score, ae_anomaly = self.predict_autoencoder(features)
        
        # Ensemble combination
        if self.isolation_forest is not None and self.autoencoder is not None:
            # Weighted average
            ensemble_score = (
                self.ensemble_weights['isolation_forest'] * if_score +
                self.ensemble_weights['autoencoder'] * ae_score
            )
            
            # Combined decision (OR logic - anomaly if either model says so)
            is_anomaly = if_anomaly or ae_anomaly
            
        elif self.isolation_forest is not None:
            ensemble_score = if_score
            is_anomaly = if_anomaly
            
        elif self.autoencoder is not None:
            ensemble_score = ae_score
            is_anomaly = ae_anomaly
            
        else:
            # No models available - use rule-based fallback
            ensemble_score = self._rule_based_detection(message_data)
            is_anomaly = ensemble_score > 0.7
        
        # Determine risk level
        if ensemble_score >= 0.8:
            risk_level = 'CRITICAL'
        elif ensemble_score >= 0.6:
            risk_level = 'HIGH'
        elif ensemble_score >= 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'anomaly_score': float(ensemble_score),
            'is_anomaly': bool(is_anomaly),
            'risk_level': risk_level,
            'model_scores': {
                'isolation_forest': float(if_score),
                'autoencoder': float(ae_score)
            },
            'model_version': self._get_model_version()
        }
    
    def predict_batch(self, features_array: np.ndarray) -> List[Dict]:
        """Predict anomalies for batch of messages"""
        results = []
        
        for i in range(features_array.shape[0]):
            features = features_array[i]
            result = self.predict(features)
            results.append(result)
        
        return results
    
    def _rule_based_detection(self, message_data: Dict) -> float:
        """Simple rule-based detection as fallback"""
        score = 0.0
        
        if isinstance(message_data, dict):
            message_text = message_data.get('message_text', '').lower()
            source_addr = message_data.get('source_addr', '').upper()
            
            # Suspicious patterns
            suspicious_patterns = [
                'заблоковано', 'виграли', 'безкоштов', 
                'термінов', 'увага', 'bit.ly', 'tinyurl'
            ]
            
            for pattern in suspicious_patterns:
                if pattern in message_text:
                    score += 0.2
            
            # Suspicious sources
            if any(char in source_addr for char in ['!', '@', '#', '$']):
                score += 0.3
            
            # Length anomalies
            if len(message_text) > 500 or len(message_text) == 0:
                score += 0.2
        
        return min(score, 1.0)
    
    def _get_model_version(self) -> str:
        """Get current model version"""
        versions = []
        
        if self.isolation_forest_data:
            versions.append(f"IF_{self.isolation_forest_data.get('timestamp', 'unknown')}")
        
        if self.autoencoder_data:
            versions.append("AE_loaded")
        
        return "_".join(versions) if versions else "no_model"
    
    def update_ensemble_weights(self, weights: Dict[str, float]):
        """Update ensemble weights"""
        total = sum(weights.values())
        
        # Normalize weights
        for model, weight in weights.items():
            if model in self.ensemble_weights:
                self.ensemble_weights[model] = weight / total
        
        self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'isolation_forest': {
                'loaded': self.isolation_forest is not None,
                'n_models': len(self.isolation_forest) if self.isolation_forest else 0,
                'features': len(self.feature_names) if self.feature_names else 0
            },
            'autoencoder': {
                'loaded': self.autoencoder is not None,
                'device': str(device)
            },
            'ensemble_weights': self.ensemble_weights
        }
        
        return info


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = AnomalyDetector(models_dir='models')
    
    # Get model info
    print("Model Info:", detector.get_model_info())
    
    # Test message
    test_message = {
        'message_text': 'УВАГА! Ваш рахунок заблоковано! Перейдіть bit.ly/unlock',
        'source_addr': 'BANK-FAKE',
        'dest_addr': '380501234567',
        'timestamp': datetime.now()
    }
    
    # Predict
    result = detector.predict(test_message)
    
    print(f"\nAnomaly Detection Result:")
    print(f"  Score: {result['anomaly_score']:.3f}")
    print(f"  Is Anomaly: {result['is_anomaly']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Model Scores: {result['model_scores']}")