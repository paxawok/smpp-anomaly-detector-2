import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from typing import Tuple

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

# Перевірка доступності GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Використовується пристрій: {device}")

class SMPPAutoencoder(nn.Module):
    """Автоенкодер для виявлення аномалій в SMPP"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, 
                 hidden_dims: list = None, dropout_rate: float = 0.2):
        super(SMPPAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Енкодер
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
        
        # Декодер (дзеркальна архітектура)
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
    
    def encode(self, x):
        return self.encoder(x)

class SMPPAutoencoderDetector:
    """Детектор аномалій на основі автоенкодера"""
    
    def __init__(self, encoding_dim: int = 32, hidden_dims: list = None,
                 learning_rate: float = 0.001, batch_size: int = 64,
                 epochs: int = 100, dropout_rate: float = 0.2):
        
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = {'train_loss': [], 'val_loss': []}
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[DataLoader, DataLoader]:
        """Підготовка даних для навчання"""
        # Нормалізація
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Розділення на навчальну та валідаційну вибірки (тільки нормальні дані)
        if y is not None:
            X_normal = X_scaled[y == 0]
        else:
            X_normal = X_scaled
        
        # 80/20 split для навчання/валідації
        X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
        
        # Створення DataLoader'ів
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, X: np.ndarray, y: np.ndarray = None):
        """Навчання автоенкодера"""
        logging.info("Початок навчання автоенкодера...")
        
        # Підготовка даних
        train_loader, val_loader = self.prepare_data(X, y)
        
        # Ініціалізація моделі
        input_dim = X.shape[1]
        self.model = SMPPAutoencoder(
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(device)
        
        # Оптимізатор та функція втрат
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Навчання
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                inputs = batch[0].to(device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()
            
            # Збереження історії
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{self.epochs}], "
                             f"Train Loss: {avg_train_loss:.6f}, "
                             f"Val Loss: {avg_val_loss:.6f}")
        
        # Визначення порогу на валідаційних даних
        self._determine_threshold(val_loader)
        
        logging.info("Навчання завершено!")
    
    def _determine_threshold(self, val_loader: DataLoader):
        """Визначення оптимального порогу"""
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = self.model(inputs)
                mse = torch.mean((inputs - outputs) ** 2, dim=1)
                reconstruction_errors.extend(mse.cpu().numpy())
        
        # Використовуємо 95-й перцентиль як поріг
        self.threshold = np.percentile(reconstruction_errors, 95)
        logging.info(f"Визначено поріг: {self.threshold:.6f}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Передбачення аномалій"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        
        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                outputs = self.model(inputs)
                mse = torch.mean((inputs - outputs) ** 2, dim=1)
                reconstruction_errors.extend(mse.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        return predictions, reconstruction_errors

# Використання автоенкодера
if __name__ == "__main__":
    # Завантаження даних з ознаками
    logging.info("Завантаження даних...")
    df = pd.read_csv('data/datasets/smpp_weekly_dataset_features_optimized.csv', encoding='utf-8-sig')
    
    # Вибір ознак для навчання
    feature_columns = [col for col in df.columns 
                      if col not in ['message_id', 'source_addr', 'dest_addr', 
                                    'submit_time', 'message_text', 'category', 
                                    'is_anomaly', 'time_category', 'anomaly_type', 'weekday']]
    
    X = df[feature_columns].values
    y = df['is_anomaly'].values
    
    # Створення та навчання автоенкодера
    autoencoder = SMPPAutoencoderDetector(
        encoding_dim=24,
        hidden_dims=[96, 48],
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        dropout_rate=0.2
    )
    
    # Навчання
    autoencoder.train(X, y)
    
    # Передбачення аномалій
    predictions, reconstruction_errors = autoencoder.predict(X)
    
    # Виведення результатів
    print("Передбачення аномалій:", predictions)
