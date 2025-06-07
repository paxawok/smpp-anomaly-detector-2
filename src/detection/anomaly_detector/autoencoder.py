import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from typing import Tuple, Dict, Optional, List
from pathlib import Path
from datetime import datetime

# Налаштування логування
def setup_logging(log_dir: Path = Path("logs/ae")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class SMPPAutoencoder(nn.Module):
    """Autoencoder for SMPP anomaly detection with GPU support."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, 
                 hidden_dims: List[int] = None, dropout_rate: float = 0.2):
        super().__init__()
        self.hidden_dims = hidden_dims or [128, 64]
        self.encoding_dim = encoding_dim
        
        # Encoder
        encoder_layers = self._build_layers(input_dim, self.hidden_dims, dropout_rate, True)
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoding_layer = nn.Linear(self.hidden_dims[-1], encoding_dim)
        
        # Decoder
        decoder_layers = self._build_layers(encoding_dim, list(reversed(self.hidden_dims[:-1])) + [self.hidden_dims[-1]], 
                                           dropout_rate, False)
        self.decoder = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(self.hidden_dims[-1], input_dim)
        
        self._init_weights()
    
    def _build_layers(self, input_dim: int, hidden_dims: List[int], 
                     dropout_rate: float, is_encoder: bool) -> List[nn.Module]:
        """Helper method to build encoder or decoder layers."""
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate * (i + 1 if is_encoder else len(hidden_dims) - i) / len(hidden_dims))
            ])
            prev_dim = hidden_dim
        return layers
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)  # Shape: (batch_size, hidden_dims[-1])
        encoded = self.encoding_layer(features)  # Shape: (batch_size, encoding_dim)
        decoded = self.decoder(encoded)  # Shape: (batch_size, hidden_dims[-1])
        return self.output_layer(decoded + features)  # Skip connection
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(x)
            return self.encoding_layer(features)

class DataPreprocessor:
    """Handles data preprocessing including scaling and encoding."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def preprocess(self, df: pd.DataFrame, categorical_columns: List[str]) -> np.ndarray:
        """Preprocess DataFrame, handling numerical and categorical features."""
        df = df.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Encode categorical columns
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Select numeric columns and scale
        numeric_df = df.select_dtypes(include=[np.number])
        return self.scaler.fit_transform(numeric_df)
    
    def transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> np.ndarray:
        """Transform new data using fitted preprocessors."""
        df = df.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        
        for col in categorical_columns:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        numeric_df = df.select_dtypes(include=[np.number])
        return self.scaler.transform(numeric_df)

class SMPPAutoencoderDetector:
    """Anomaly detector using an autoencoder."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.config = config or {
            'encoding_dim': 32,
            'hidden_dims': [128, 64],
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 100,
            'dropout_rate': 0.2,
            'early_stopping_patience': 10,
            'lr_scheduler': True,
            'weight_decay': 1e-5
        }
        
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.threshold = None
        self.history = {'train_loss': [], 'val_loss': [], 'best_val_loss': float('inf'), 'stopped_epoch': 0}
    
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training, filtering normal samples if labels are provided."""
        X_normal = X if y is None else X[y == 0]
        X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'] * 2, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, df: pd.DataFrame, y: Optional[np.ndarray] = None, categorical_columns: List[str] = None):
        """Train the autoencoder with logging and early stopping."""
        categorical_columns = categorical_columns or ['weekday', 'category', 'anomaly_type']
        logger.info("Starting training...")
        
        try:
            # Preprocess data
            X = self.preprocessor.preprocess(df, categorical_columns)
            train_loader, val_loader = self.prepare_data(X, y)
            
            # Initialize model
            self.model = SMPPAutoencoder(
                input_dim=X.shape[1],
                encoding_dim=self.config['encoding_dim'],
                hidden_dims=self.config['hidden_dims'],
                dropout_rate=self.config['dropout_rate']
            ).to(self.device)
            
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) if self.config['lr_scheduler'] else None
            criterion = nn.MSELoss()
            early_stopping_counter = 0
            
            for epoch in range(self.config['epochs']):
                self.model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    inputs = batch[0].to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                
                val_loss = self._validate(val_loader, criterion)
                avg_train_loss = train_loss / len(train_loader.dataset)
                self.history['train_loss'].append(avg_train_loss)
                self.history['val_loss'].append(val_loss)
                
                if scheduler:
                    scheduler.step(val_loss)
                
                if val_loss < self.history['best_val_loss']:
                    self.history['best_val_loss'] = val_loss
                    early_stopping_counter = 0
                    self._save_checkpoint(epoch, val_loss)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.config['early_stopping_patience']:
                        self.history['stopped_epoch'] = epoch
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            self._load_best_checkpoint()
            self._determine_threshold(val_loader)
            logger.info("Training completed!")
            return self.history
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def _validate(self, val_loader: DataLoader, criterion) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        return val_loss / len(val_loader.dataset)
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'preprocessor': self.preprocessor,
            'config': self.config,
            'threshold': self.threshold
        }
        torch.save(checkpoint, 'best_model_checkpoint.pt')
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        checkpoint = torch.load('best_model_checkpoint.pt', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.preprocessor = checkpoint['preprocessor']
        self.threshold = checkpoint['threshold']
        logger.info(f"Restored best model from epoch {checkpoint['epoch']}")
    
    def _determine_threshold(self, val_loader: DataLoader, percentile: float = 95):
        """Determine anomaly threshold based on reconstruction errors."""
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                mse = torch.mean((inputs - outputs) ** 2, dim=1)
                reconstruction_errors.extend(mse.cpu().numpy())
        
        self.threshold = np.percentile(reconstruction_errors, percentile)
        logger.info(f"Threshold ({percentile}%): {self.threshold:.6f}")
    
    def predict(self, df: pd.DataFrame, categorical_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in the data."""
        if self.model is None or self.threshold is None:
            raise ValueError("Model not trained or threshold not set")
        
        categorical_columns = categorical_columns or ['weekday', 'category', 'anomaly_type']
        X_scaled = self.preprocessor.transform(df, categorical_columns)
        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        loader = DataLoader(dataset, batch_size=self.config['batch_size'] * 4, shuffle=False)
        
        self.model.eval()
        all_errors = np.empty(len(X_scaled))
        
        with torch.no_grad():
            start_idx = 0
            for batch in loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                batch_errors = torch.mean((inputs - outputs) ** 2, dim=1).cpu().numpy()
                end_idx = start_idx + len(batch_errors)
                all_errors[start_idx:end_idx] = batch_errors
                start_idx = end_idx
        
        predictions = (all_errors > self.threshold).astype(int)
        return predictions, all_errors
    
    def save_model(self, path: str = "saved_models") -> str:
        """Save the model to a file."""
        model_dir = Path(path)
        model_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"smpp_autoencoder_{timestamp}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'preprocessor': self.preprocessor,
            'threshold': self.threshold,
            'config': self.config,
            'history': self.history,
            'input_dim': self.model.encoder[0].in_features
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)

# Example usage
if __name__ == "__main__":
    try:
        logger.info("Loading data...")
        df = pd.read_csv('data/datasets/smpp_weekly_dataset_features_optimized.csv', encoding='utf-8-sig')
        
        feature_columns = [col for col in df.columns 
                         if col not in ['message_id', 'source_addr', 'dest_addr', 'submit_time', 'message_text', 'category', 'is_anomaly']]
        categorical_columns = ['weekday', 'category', 'anomaly_type']
        
        detector = SMPPAutoencoderDetector()
        history = detector.train(df[feature_columns], df['is_anomaly'].values, categorical_columns)
        
        model_path = detector.save_model()
        predictions, errors = detector.predict(df[feature_columns], categorical_columns)
        logger.info(f"Detected {predictions.sum()} anomalies out of {len(predictions)} samples")
        
        # Optional: Plot training history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
        
    except Exception as e:
        logger.exception("Critical error occurred:")