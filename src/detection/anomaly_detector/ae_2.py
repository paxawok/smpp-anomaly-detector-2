import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import logging
import pandas as pd
from typing import Tuple, Dict, Optional, List, Union
from pathlib import Path
from datetime import datetime
import json
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Константи для features
FEATURE_COLUMNS = [
    "hour", "day_of_week", "message_length", "source_addr_length", 
    "source_is_numeric", "dest_is_valid", "message_parts", "encoding_issues", 
    "empty_message", "excessive_length", "sender_frequency", "recipient_frequency", 
    "sender_burst", "recipient_burst", "high_sender_frequency", 
    "high_recipient_frequency", "suspicious_word_count", "url_count", 
    "suspicious_url", "urgency_score", "message_entropy", "obfuscation_score", 
    "social_engineering", "night_time", "weekend", "business_hours", 
    "time_category_anomaly", "sender_legitimacy", "financial_patterns",
    "phone_numbers", "source_category_mismatch", "category_time_mismatch",
    "unusual_sender_pattern", "typosquatting"
]

# Категоріальні ознаки (потребують one-hot encoding)
CATEGORICAL_FEATURES = ["day_of_week", "time_category_anomaly"]

# Бінарні ознаки (0 або 1)
BINARY_FEATURES = [
    "source_is_numeric", "dest_is_valid", "encoding_issues", "empty_message",
    "excessive_length", "high_sender_frequency", "high_recipient_frequency",
    "suspicious_url", "social_engineering", "night_time", "weekend", 
    "business_hours", "sender_legitimacy", "source_category_mismatch",
    "category_time_mismatch", "unusual_sender_pattern", "typosquatting"
]

# Числові ознаки (потребують масштабування)
NUMERIC_FEATURES = [
    "hour", "message_length", "source_addr_length", "sender_frequency",
    "recipient_frequency", "sender_burst", "recipient_burst", 
    "suspicious_word_count", "url_count", "urgency_score", 
    "message_entropy", "obfuscation_score", "message_parts",
    "financial_patterns", "phone_numbers"
]

# Текстові ознаки для додаткової обробки
TEXT_FEATURES = ["source_addr", "dest_addr", "message_text", "category"]

# Налаштування логування
def setup_logging(log_dir: Path = Path("logs/ae")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Створюємо унікальне ім'я файлу логу з timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Налаштовуємо формат логування
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Logger setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class ImprovedSMPPAutoencoder(nn.Module):
    """Покращений Autoencoder з attention механізмом та різними типами layers."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, 
                 hidden_dims: List[int] = None, dropout_rate: float = 0.2,
                 use_attention: bool = True):
        super().__init__()
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.encoding_dim = encoding_dim
        self.use_attention = use_attention
        
        # Encoder з покращеною архітектурою
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate * (1 - i/len(self.hidden_dims)))  # Зменшуємо dropout глибше в мережі
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Attention mechanism (optional)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dims[-1],
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Bottleneck
        self.encoding_layer = nn.Linear(self.hidden_dims[-1], encoding_dim)
        self.decoding_layer = nn.Linear(encoding_dim, self.hidden_dims[-1])
        
        # Decoder (симетричний до encoder)
        decoder_layers = []
        hidden_dims_reversed = list(reversed(self.hidden_dims))
        
        for i in range(len(hidden_dims_reversed) - 1):
            decoder_layers.extend([
                nn.Linear(hidden_dims_reversed[i], hidden_dims_reversed[i+1]),
                nn.BatchNorm1d(hidden_dims_reversed[i+1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate * (i+1)/len(hidden_dims_reversed))
            ])
        
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Residual connections
        self.use_residual = True
        if self.use_residual and self.hidden_dims[0] == input_dim:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming initialization для кращої збіжності."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoding
        encoded_features = self.encoder(x)
        
        # Attention (if enabled)
        if self.use_attention:
            # Reshape for attention: (batch, 1, features)
            attn_input = encoded_features.unsqueeze(1)
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            encoded_features = attn_output.squeeze(1)
        
        # Bottleneck
        encoded = self.encoding_layer(encoded_features)
        
        # Decoding
        decoded = self.decoding_layer(encoded)
        reconstructed = self.decoder(decoded)
        
        # Residual connection (if applicable)
        if self.use_residual and hasattr(self, 'residual_weight'):
            reconstructed = reconstructed + self.residual_weight * x
        
        return reconstructed, encoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded_features = self.encoder(x)
            if self.use_attention:
                attn_input = encoded_features.unsqueeze(1)
                attn_output, _ = self.attention(attn_input, attn_input, attn_input)
                encoded_features = attn_output.squeeze(1)
            return self.encoding_layer(encoded_features)

class AdvancedDataPreprocessor:
    """Покращений препроцесор з обробкою текстових features."""
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        self.label_encoders = {}
        self.text_vectorizers = {}
        self.feature_stats = defaultdict(dict)
    
    def _extract_text_features(self, text_series: pd.Series, feature_name: str) -> pd.DataFrame:
        """Витягує статистичні ознаки з текстових полів."""
        features = pd.DataFrame()
        
        # Конвертуємо всі значення в строки та обробляємо NaN
        text_series = text_series.fillna('').astype(str)
        
        if feature_name in ['source_addr', 'dest_addr']:
            features[f'length'] = text_series.str.len()
            features[f'is_numeric'] = text_series.str.match(r'^\d+$').fillna(False).astype(int)
            features[f'has_special'] = text_series.str.contains(r'[^a-zA-Z0-9]', na=False).astype(int)
            features[f'unique_chars'] = text_series.apply(lambda x: len(set(str(x))))
            
        elif feature_name == 'message_text':
            features['length'] = text_series.str.len()
            features['word_count'] = text_series.str.split().str.len().fillna(0)
            features['unique_word_ratio'] = text_series.apply(
                lambda x: len(set(str(x).split())) / max(len(str(x).split()), 1) if x else 0
            )
            features['digit_ratio'] = text_series.apply(
                lambda x: sum(c.isdigit() for c in str(x)) / max(len(str(x)), 1) if x else 0
            )
            features['upper_ratio'] = text_series.apply(
                lambda x: sum(c.isupper() for c in str(x)) / max(len(str(x)), 1) if x else 0
            )
        
        elif feature_name == 'category':
            # Для категоріальної колонки просто повертаємо довжину
            features['length'] = text_series.str.len()
        
        # Заповнюємо всі NaN нулями
        features = features.fillna(0)
        
        # Важливо: повертаємо DataFrame з консистентними назвами колонок
        return features
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Комплексна обробка даних з логуванням."""
        logger.info(f"Preprocessing data with shape: {df.shape}")
        df = df.copy()
        
        # Перевірка наявності всіх необхідних ознак
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing required features: {missing_features}")
        
        # Обробка пропущених значень для числових колонок
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Заповнюємо пропущені значення медіаною для кожної колонки окремо
        for col in numeric_columns:
            if col in df.columns:
                median_value = df[col].median()
                if pd.isna(median_value):
                    median_value = 0
                df[col] = df[col].fillna(median_value)
        
        processed_features = []
        feature_names = []
        
        # 1. Обробка числових features (з NUMERIC_FEATURES)
        numeric_features_present = [col for col in NUMERIC_FEATURES if col in df.columns]
        if numeric_features_present:
            numeric_data = df[numeric_features_present].values
            if fit:
                numeric_scaled = self.scalers['standard'].fit_transform(numeric_data)
                self.numeric_features_order = numeric_features_present
            else:
                # При transform використовуємо той самий порядок
                if hasattr(self, 'numeric_features_order'):
                    # Перевіряємо, чи всі збережені features присутні
                    numeric_data = df[self.numeric_features_order].values
                    numeric_scaled = self.scalers['standard'].transform(numeric_data)
                else:
                    numeric_scaled = np.array([])
            
            if numeric_scaled.size > 0:
                processed_features.append(numeric_scaled)
                feature_names.extend([f'numeric_{col}' for col in (self.numeric_features_order if not fit else numeric_features_present)])
                logger.info(f"Processed {len(numeric_features_present)} numeric features")
        
        # 2. Обробка бінарних features
        binary_features_present = [col for col in BINARY_FEATURES if col in df.columns]
        if fit:
            self.binary_features_order = binary_features_present
        
        if hasattr(self, 'binary_features_order') and self.binary_features_order:
            # Перевіряємо наявність всіх бінарних ознак
            missing_binary = [col for col in self.binary_features_order if col not in df.columns]
            if missing_binary:
                logger.warning(f"Missing binary features during transform: {missing_binary}")
                # Додаємо відсутні колонки з нулями
                for col in missing_binary:
                    df[col] = 0
            
            binary_data = df[self.binary_features_order].fillna(0).values.astype(float)
            processed_features.append(binary_data)
            feature_names.extend([f'binary_{col}' for col in self.binary_features_order])
            logger.info(f"Processed {len(self.binary_features_order)} binary features")
        
        # 3. Обробка категоріальних features
        if fit:
            self.categorical_features_fitted = []
        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                if fit:
                    self.categorical_features_fitted.append(col)
                    
                    # Конвертуємо в строки та заповнюємо пропуски
                    col_data = df[col].fillna('unknown').astype(str)
                    
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    
                    # Fit з усіма можливими значеннями плюс 'unknown'
                    unique_values = list(col_data.unique())
                    if 'unknown' not in unique_values:
                        unique_values.append('unknown')
                    self.label_encoders[col].fit(unique_values)
                    encoded = self.label_encoders[col].transform(col_data)
                    
                    # One-hot encoding
                    n_categories = len(self.label_encoders[col].classes_)
                    one_hot = np.zeros((len(df), n_categories))
                    one_hot[np.arange(len(df)), encoded] = 1
                    processed_features.append(one_hot)
                    logger.info(f"Processed categorical feature {col} with {n_categories} categories")
                else:
                    # При transform
                    if hasattr(self, 'categorical_features_fitted') and col in self.categorical_features_fitted:
                        col_data = df[col].fillna('unknown').astype(str)
                        
                        # Transform з обробкою невідомих категорій
                        le = self.label_encoders[col]
                        known_labels = set(le.classes_)
                        col_data_safe = col_data.apply(lambda x: x if x in known_labels else 'unknown')
                        
                        # Якщо 'unknown' не в classes_, використовуємо першу категорію
                        if 'unknown' not in known_labels:
                            col_data_safe = col_data_safe.replace('unknown', le.classes_[0])
                        
                        encoded = le.transform(col_data_safe)
                        
                        # One-hot encoding
                        n_categories = len(le.classes_)
                        one_hot = np.zeros((len(df), n_categories))
                        one_hot[np.arange(len(df)), encoded] = 1
                        processed_features.append(one_hot)
        
        # 4. НЕ обробляємо текстові features для уникнення проблем з розмірністю
        # Якщо потрібно, їх можна додати з фіксованою кількістю ознак
        
        # Об'єднання всіх features
        if processed_features:
            X = np.hstack(processed_features)
        else:
            logger.warning("No features were processed!")
            X = np.zeros((len(df), 1))
        
        logger.info(f"Final preprocessed shape: {X.shape}")
        
        # Збереження статистики
        if fit:
            self.n_features_fitted = X.shape[1]
            self.feature_stats['n_features'] = X.shape[1]
            self.feature_stats['feature_types'] = {
                'numeric': len(numeric_features_present),
                'binary': len(binary_features_present),
                'categorical': sum(len(self.label_encoders[col].classes_) for col in self.categorical_features_fitted)
            }
            self.feature_stats['feature_order'] = {
                'numeric': self.numeric_features_order,
                'binary': self.binary_features_order,
                'categorical': self.categorical_features_fitted
            }
            logger.info(f"Fitted preprocessor expects {self.n_features_fitted} features")
            logger.info(f"Feature breakdown: {self.feature_stats['feature_types']}")
        else:
            # Перевірка розмірності при transform
            if hasattr(self, 'n_features_fitted') and X.shape[1] != self.n_features_fitted:
                logger.error(f"Feature dimension mismatch! Expected {self.n_features_fitted}, got {X.shape[1]}")
        
        return X

class SMPPAutoencoderDetector:
    """Покращений детектор аномалій з розширеними можливостями."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.config = config or {
            'encoding_dim': 32,
            'hidden_dims': [256, 128, 64],
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 150,
            'dropout_rate': 0.3,
            'early_stopping_patience': 15,
            'lr_scheduler': True,
            'weight_decay': 1e-5,
            'use_attention': True,
            'gradient_clip': 1.0,
            'warmup_epochs': 5
        }
        
        self.model = None
        self.preprocessor = AdvancedDataPreprocessor()
        self.threshold = None
        self.threshold_percentile = 95
        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'learning_rates': [],
            'best_val_loss': float('inf'), 
            'stopped_epoch': 0
        }
        self.training_time = None
    
    def prepare_data(self, X: np.ndarray, augment: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Підготовка даних з можливістю аугментації."""
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)
        
        # Data augmentation для тренувальних даних
        if augment:
            noise_factor = 0.01
            X_train_augmented = X_train + noise_factor * np.random.randn(*X_train.shape)
            X_train = np.vstack([X_train, X_train_augmented])
            logger.info(f"Applied data augmentation. New training size: {X_train.shape[0]}")
        
        # Створення DataLoader з pin_memory для GPU
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'] * 2, 
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def _combined_loss(self, inputs: torch.Tensor, outputs: torch.Tensor, 
                      encoded: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
        """Комбінована loss функція: MSE + sparsity regularization."""
        mse_loss = nn.functional.mse_loss(outputs, inputs)
        
        # Sparsity loss (KL divergence)
        rho = 0.05  # цільова sparsity
        rho_hat = torch.mean(encoded, dim=0)
        kl_div = torch.sum(rho * torch.log(rho / rho_hat) + 
                          (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))
        
        return mse_loss + beta * kl_div
    
    def train(self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None):
        """Покращене тренування з детальним логуванням."""
        start_time = datetime.now()
        logger.info("="*50)
        logger.info("Starting enhanced training process...")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        try:
            # Фільтрація тільки нормальних даних
            df_normal = df[df['is_anomaly'] == False].copy()
            logger.info(f"Training on {len(df_normal)} normal samples")
            
            # Препроцесинг
            X = self.preprocessor.preprocess(df_normal, fit=True)
            train_loader, val_loader = self.prepare_data(X, augment=True)
            
            # Ініціалізація моделі
            self.model = ImprovedSMPPAutoencoder(
                input_dim=X.shape[1],
                encoding_dim=self.config['encoding_dim'],
                hidden_dims=self.config['hidden_dims'],
                dropout_rate=self.config['dropout_rate'],
                use_attention=self.config['use_attention']
            ).to(self.device)
            
            logger.info(f"Model architecture:\n{self.model}")
            logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            # Оптимізатор та scheduler
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999)
            )
            
            # Learning rate scheduling
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
            
            # Training loop
            best_model_state = None
            early_stopping_counter = 0
            
            for epoch in range(self.config['epochs']):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    inputs = batch[0].to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs, encoded = self.model(inputs)
                    loss = self._combined_loss(inputs, outputs, encoded)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config['gradient_clip']:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['gradient_clip']
                        )
                    
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                    train_batches += 1
                    
                    # Логування прогресу
                    if batch_idx % 10 == 0:
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                                   f"Loss: {loss.item():.6f}")
                
                # Validation phase
                val_loss = self._validate(val_loader)
                avg_train_loss = train_loss / len(train_loader.dataset)
                
                # Збереження історії
                self.history['train_loss'].append(avg_train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                # Learning rate scheduling
                scheduler.step()
                
                # Early stopping logic
                if val_loss < self.history['best_val_loss']:
                    self.history['best_val_loss'] = val_loss
                    best_model_state = self.model.state_dict().copy()
                    early_stopping_counter = 0
                    logger.info(f"New best model at epoch {epoch+1}")
                else:
                    early_stopping_counter += 1
                
                # Логування
                if (epoch + 1) % 5 == 0 or early_stopping_counter == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.config['epochs']} | "
                        f"Train Loss: {avg_train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                        f"Best Val: {self.history['best_val_loss']:.6f}"
                    )
                
                # Early stopping
                if early_stopping_counter >= self.config['early_stopping_patience']:
                    self.history['stopped_epoch'] = epoch + 1
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Відновлення найкращої моделі
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                logger.info("Restored best model weights")
            
            # Визначення порогу
            self._determine_threshold(val_loader)
            
            # Час тренування
            self.training_time = (datetime.now() - start_time).total_seconds() / 60
            logger.info(f"Training completed in {self.training_time:.2f} minutes")
            
            # Збереження графіків
            self._save_training_plots()
            
            return self.history
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
            raise
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Валідація моделі."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                outputs, encoded = self.model(inputs)
                loss = self._combined_loss(inputs, outputs, encoded)
                val_loss += loss.item() * inputs.size(0)
        
        return val_loss / len(val_loader.dataset)
    
    def _determine_threshold(self, val_loader: DataLoader):
        """Визначення адаптивного порогу на основі розподілу помилок."""
        logger.info("Determining anomaly threshold...")
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                outputs, _ = self.model(inputs)
                mse = torch.mean((inputs - outputs) ** 2, dim=1)
                reconstruction_errors.extend(mse.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Статистичний аналіз помилок
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        
        logger.info(f"Reconstruction error statistics:")
        logger.info(f"  Mean: {mean_error:.6f}")
        logger.info(f"  Std: {std_error:.6f}")
        logger.info(f"  Min: {np.min(reconstruction_errors):.6f}")
        logger.info(f"  Max: {np.max(reconstruction_errors):.6f}")
        logger.info(f"  Threshold ({self.threshold_percentile}%): {self.threshold:.6f}")
        
        # Збереження розподілу помилок
        self._save_error_distribution(reconstruction_errors)
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Предикція з детальною статистикою."""
        if self.model is None or self.threshold is None:
            raise ValueError("Model not trained or threshold not set")
        
        logger.info(f"Making predictions on {len(df)} samples")
        
        # Препроцесинг
        X = self.preprocessor.preprocess(df, fit=False)
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'] * 4, 
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.model.eval()
        all_errors = []
        all_encodings = []
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                outputs, encoded = self.model(inputs)
                batch_errors = torch.mean((inputs - outputs) ** 2, dim=1).cpu().numpy()
                all_errors.extend(batch_errors)
                all_encodings.extend(encoded.cpu().numpy())
        
        all_errors = np.array(all_errors)
        predictions = (all_errors > self.threshold).astype(int)
        
        # Статистика предикцій
        stats = {
            'total_samples': len(predictions),
            'anomalies_detected': int(predictions.sum()),
            'anomaly_rate': float(predictions.mean()),
            'mean_error': float(all_errors.mean()),
            'max_error': float(all_errors.max()),
            'threshold': float(self.threshold)
        }
        
        logger.info(f"Prediction statistics: {json.dumps(stats, indent=2)}")
        
        return predictions, all_errors, stats
    
    def _save_training_plots(self):
        """Збереження графіків тренування."""
        plot_dir = Path("data/plots/ae")
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Loss curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rates'], label='Learning Rate', linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {plot_dir}")
    
    def _save_error_distribution(self, errors: np.ndarray):
        """Збереження розподілу помилок реконструкції."""
        plot_dir = Path("data/plots/ae")
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(10, 6))
        
        # Гістограма з KDE
        sns.histplot(errors, bins=50, kde=True, color='blue', alpha=0.7)
        plt.axvline(self.threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({self.threshold_percentile}%): {self.threshold:.6f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'error_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, path: str = "models") -> str:
        """Збереження моделі з метаданими."""
        model_dir = Path(path)
        model_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"smpp_autoencoder_{timestamp}"
        model_path = model_dir / f"{model_name}.pt"
        
        # Збереження моделі з додатковою інформацією про архітектуру
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.encoder[0].in_features,
                'encoding_dim': self.config['encoding_dim'],
                'hidden_dims': self.config['hidden_dims'],
                'use_attention': self.config['use_attention']
            },
            'preprocessor': self.preprocessor,
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'config': self.config,
            'history': self.history,
            'training_time_minutes': self.training_time,
            'feature_stats': self.preprocessor.feature_stats,
            'timestamp': timestamp,
            'input_shape': self.model.encoder[0].in_features  # Зберігаємо очікувану розмірність
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Expected input dimension: {self.model.encoder[0].in_features}")
        return str(model_path)
    
    def save_to_database(self, model_path: str, test_results: Dict):
        """Збереження метаданих моделі в базу даних."""
        db_path = Path("data/db/smpp.sqlite")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Підготовка даних для вставки
            model_data = {
                'model_name': f"SMPP_Autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_type': 'Autoencoder',
                'version': '2.0',
                'file_path': model_path,
                'config_json': json.dumps(self.config),
                'feature_names': json.dumps(FEATURE_COLUMNS),
                'accuracy': test_results.get('accuracy', 0),
                'precision_score': test_results.get('precision', 0),
                'recall': test_results.get('recall', 0),
                'f1_score': test_results.get('f1', 0),
                'roc_auc': test_results.get('roc_auc', 0),
                'training_dataset_size': len(self.history['train_loss']) * self.config['batch_size'],
                'training_duration_minutes': int(self.training_time),
                'validation_score': self.history['best_val_loss'],
                'is_active': 1,
                'deployment_status': 'ready',
                'trained_by': 'AutoML System',
                'training_notes': json.dumps({
                    'threshold': float(self.threshold),
                    'threshold_percentile': self.threshold_percentile,
                    'early_stopped_epoch': self.history.get('stopped_epoch', self.config['epochs']),
                    'feature_stats': self.preprocessor.feature_stats
                })
            }
            
            # SQL запит
            insert_query = """
            INSERT INTO models (
                model_name, model_type, version, file_path, config_json,
                feature_names, accuracy, precision_score, recall, f1_score,
                roc_auc, training_dataset_size, training_duration_minutes,
                validation_score, is_active, deployment_status, trained_by,
                training_notes, created_at
            ) VALUES (
                :model_name, :model_type, :version, :file_path, :config_json,
                :feature_names, :accuracy, :precision_score, :recall, :f1_score,
                :roc_auc, :training_dataset_size, :training_duration_minutes,
                :validation_score, :is_active, :deployment_status, :trained_by,
                :training_notes, datetime('now')
            )
            """
            
            cursor.execute(insert_query, model_data)
            conn.commit()
            
            logger.info(f"Model metadata saved to database with ID: {cursor.lastrowid}")
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}", exc_info=True)
            raise
        finally:
            conn.close()

def evaluate_model(detector: SMPPAutoencoderDetector, df: pd.DataFrame) -> Dict:
    """Комплексна оцінка моделі."""
    logger.info("Evaluating model performance...")
    
    # Предикція
    predictions, errors, stats = detector.predict(df)
    
    # Якщо є справжні мітки
    if 'is_anomaly' in df.columns:
        y_true = df['is_anomaly'].values
        
        # Метрики
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, errors)
        except:
            roc_auc = 0.5
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        logger.info(f"Model Performance Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Збереження confusion matrix
        save_confusion_matrix(cm)
        
        # Збереження ROC curve
        if roc_auc > 0.5:
            save_roc_curve(y_true, errors)
    else:
        results = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'roc_auc': 0
        }
    
    return results

def save_confusion_matrix(cm: np.ndarray):
    """Збереження confusion matrix."""
    plot_dir = Path("data/plots/ae")
    plot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(plot_dir / f'confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_roc_curve(y_true: np.ndarray, scores: np.ndarray):
    """Збереження ROC curve."""
    plot_dir = Path("data/plots/ae")
    plot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f'roc_curve_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_predictions(df: pd.DataFrame, predictions: np.ndarray, errors: np.ndarray):
    """Збереження результатів предикції."""
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Створення DataFrame з результатами
    results_df = df.copy()
    results_df['predicted_anomaly'] = predictions
    results_df['reconstruction_error'] = errors
    results_df['anomaly_score'] = (errors - errors.min()) / (errors.max() - errors.min())
    
    # Збереження
    output_path = pred_dir / f'predictions_{timestamp}.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Predictions saved to {output_path}")
    
    # Збереження топ аномалій
    top_anomalies = results_df.nlargest(100, 'reconstruction_error')
    top_path = pred_dir / f'top_anomalies_{timestamp}.csv'
    top_anomalies.to_csv(top_path, index=False, encoding='utf-8-sig')
    logger.info(f"Top anomalies saved to {top_path}")

def visualize_anomalies(df: pd.DataFrame, predictions: np.ndarray, errors: np.ndarray):
    """Візуалізація аномалій."""
    plot_dir = Path("data/plots/ae")
    plot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Розподіл помилок для нормальних та аномальних даних
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    normal_errors = errors[predictions == 0]
    anomaly_errors = errors[predictions == 1]
    
    plt.hist(normal_errors, bins=50, alpha=0.7, label=f'Normal (n={len(normal_errors)})', 
             color='blue', density=True)
    plt.hist(anomaly_errors, bins=30, alpha=0.7, label=f'Anomaly (n={len(anomaly_errors)})', 
             color='red', density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Часовий розподіл аномалій
    plt.subplot(1, 2, 2)
    if 'hour' in df.columns:
        hourly_anomalies = df[predictions == 1]['hour'].value_counts().sort_index()
        plt.bar(hourly_anomalies.index, hourly_anomalies.values, color='red', alpha=0.7)
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Anomalies')
        plt.title('Anomalies by Hour')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f'anomaly_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Головна функція для демонстрації
def main():
    """Основна функція для запуску всього pipeline."""
    try:
        logger.info("="*60)
        logger.info("SMPP Anomaly Detection with Enhanced Autoencoder")
        logger.info("="*60)
        
        # 1. Завантаження даних
        logger.info("Loading dataset...")
        data_path = Path('data/datasets/smpp_weekly_dataset_features_optimized.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # 2. Перевірка наявності необхідних колонок
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # 3. Ініціалізація та тренування моделі
        config = {
            'encoding_dim': 32,
            'hidden_dims': [256, 128, 64],
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 100,
            'dropout_rate': 0.3,
            'early_stopping_patience': 15,
            'lr_scheduler': True,
            'weight_decay': 1e-5,
            'use_attention': True,
            'gradient_clip': 1.0,
            'warmup_epochs': 5
        }
        
        detector = SMPPAutoencoderDetector(config)
        
        # 4. Тренування моделі
        logger.info("Starting model training...")
        history = detector.train(df)
        
        # 5. Збереження моделі
        model_path = detector.save_model()
        
        # 6. Оцінка моделі на всьому датасеті
        logger.info("Evaluating model on full dataset...")
        test_results = evaluate_model(detector, df)
        
        # 7. Збереження в базу даних
        detector.save_to_database(model_path, test_results)
        
        # 8. Предикція та збереження результатів
        predictions, errors, stats = detector.predict(df)
        save_predictions(df, predictions, errors)
        
        # 9. Візуалізація результатів
        visualize_anomalies(df, predictions, errors)
        
        # 10. Фінальна статистика
        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Total anomalies detected: {predictions.sum()} / {len(predictions)}")
        logger.info(f"Anomaly rate: {predictions.mean():.2%}")
        logger.info(f"Model performance - F1: {test_results['f1']:.4f}, ROC-AUC: {test_results['roc_auc']:.4f}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()