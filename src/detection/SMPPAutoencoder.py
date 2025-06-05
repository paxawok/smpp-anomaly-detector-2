import numpy as np
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import logging
from typing import Tuple, Dict
import joblib
import os

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
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Підготовка даних для навчання"""
        # Нормалізація
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X)
        
        
        # Розділення на навчальну та валідаційну вибірки (тільки нормальні дані)
        if y is not None:
            X_normal = X_scaled[y == 0]
            X_anomaly = X_scaled[y == 1]
        else:
            X_normal = X_scaled
            X_anomaly = None
        
        # 80/20 split для навчання/валідації
        X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
        
        # Створення DataLoader'ів
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Test loader (всі дані)
        test_dataset = TensorDataset(torch.FloatTensor(X_scaled))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train(self, X: np.ndarray, y: np.ndarray = None):
        """Навчання автоенкодера"""
        logging.info("Початок навчання автоенкодера...")
        
        # Підготовка даних
        train_loader, val_loader, _ = self.prepare_data(X, y)
        
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
            # Training phase
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
    
    def optimize_threshold(self, X: np.ndarray, y: np.ndarray):
        """Оптимізація порогу для максимізації F1-score"""
        _, reconstruction_errors = self.predict(X)
        # Усунення NaN перед передачею в precision_recall_curve
        reconstruction_errors = np.nan_to_num(reconstruction_errors, nan=0.0, posinf=1e6, neginf=0.0)

        precision, recall, thresholds = precision_recall_curve(y, reconstruction_errors)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        optimal_idx = np.argmax(f1_scores[:-1])
        self.threshold = thresholds[optimal_idx]
        
        logging.info(f"Оптимізований поріг: {self.threshold:.6f}")
        logging.info(f"F1-score: {f1_scores[optimal_idx]:.3f}")
        logging.info(f"Precision: {precision[optimal_idx]:.3f}")
        logging.info(f"Recall: {recall[optimal_idx]:.3f}")
    
    def visualize_training(self):
        """Візуалізація процесу навчання"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Автоенкодер: Історія навчання')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def visualize_results(self, X: np.ndarray, y: np.ndarray):
        """Візуалізація результатів"""
        predictions, reconstruction_errors = self.predict(X)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Розподіл помилок реконструкції
        ax = axes[0, 0]
        normal_errors = reconstruction_errors[y == 0]
        anomaly_errors = reconstruction_errors[y == 1]

        # Очистка NaN перед візуалізацією
        normal_errors = normal_errors[~np.isnan(normal_errors)]
        anomaly_errors = anomaly_errors[~np.isnan(anomaly_errors)]

        if len(normal_errors) > 0:
            ax.hist(normal_errors, bins=50, alpha=0.7, label='Нормальні', density=True)
        if len(anomaly_errors) > 0:
            ax.hist(anomaly_errors, bins=50, alpha=0.7, label='Аномалії', density=True)

        ax.axvline(x=self.threshold, color='red', linestyle='--', label=f'Поріг={self.threshold:.4f}')
        ax.set_xlabel('Помилка реконструкції')
        ax.set_ylabel('Щільність')
        ax.set_title('Розподіл помилок реконструкції')
        ax.legend()

        ax.axvline(x=self.threshold, color='red', linestyle='--', 
                  label=f'Поріг={self.threshold:.4f}')
        ax.set_xlabel('Помилка реконструкції')
        ax.set_ylabel('Щільність')
        ax.set_title('Розподіл помилок реконструкції')
        ax.legend()
        
        # 2. Confusion Matrix
        ax = axes[0, 1]
        cm = confusion_matrix(y, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Передбачено')
        ax.set_ylabel('Фактично')
        ax.set_title('Матриця плутанини')
        
        # 3. ROC простір
        ax = axes[1, 0]
        # Розрахунок для різних порогів
        thresholds = np.percentile(reconstruction_errors, np.arange(0, 100, 5))
        tpr_list, fpr_list = [], []
        
        for thresh in thresholds:
            preds = (reconstruction_errors > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        ax.plot(fpr_list, tpr_list, 'b-', linewidth=2)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC простір')
        ax.grid(True, alpha=0.3)
        
        # Позначаємо поточну робочу точку
        current_preds = (reconstruction_errors > self.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, current_preds).ravel()
        current_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        current_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        ax.scatter(current_fpr, current_tpr, color='red', s=100, 
                  label=f'Поточна точка (TPR={current_tpr:.2f}, FPR={current_fpr:.2f})')
        ax.legend()
        
        # 4. Помилки по категоріях (якщо є дані)
        ax = axes[1, 1]
        # Тут можна додати аналіз по категоріях, якщо передати додаткові дані
        ax.text(0.5, 0.5, 'Місце для додаткового аналізу', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Додатковий аналіз')
        
        plt.tight_layout()
        plt.show()
        
        # Виведення метрик
        print("\nРезультати автоенкодера:")
        print(classification_report(y, predictions, target_names=['Normal', 'Anomaly']))
        mask = ~np.isnan(reconstruction_errors)
        clean_errors = reconstruction_errors[mask]
        clean_y = y[mask]

        if len(clean_y) > 0 and len(clean_errors) > 0:
            print(f"ROC-AUC Score: {roc_auc_score(clean_y, clean_errors):.3f}")
        else:
            logging.warning("Недостатньо даних для обчислення ROC-AUC (усі значення були NaN або втрачено при фільтрації).")
 
    def save_model(self, path: str):
        """Збереження моделі"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.encoder[0].in_features,
                'encoding_dim': self.encoding_dim,
                'hidden_dims': self.hidden_dims,
                'dropout_rate': self.dropout_rate
            },
            'scaler': self.scaler,
            'threshold': self.threshold,
            'history': self.history
        }, path)
        
        logging.info(f"Модель збережено в {path}")
    
    def load_model(self, path: str):
        """Завантаження моделі"""
        checkpoint = torch.load(path, map_location=device)
        
        # Відновлення конфігурації
        config = checkpoint['model_config']
        self.encoding_dim = config['encoding_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout_rate']
        
        # Відновлення моделі
        self.model = SMPPAutoencoder(
            input_dim=config['input_dim'],
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']
        self.history = checkpoint['history']
        
        logging.info(f"Модель завантажено з {path}")


def compare_with_isolation_forest(df: pd.DataFrame, feature_columns: list):
    #Порівняння з Isolation Forest
    from sklearn.ensemble import IsolationForest
    
    X = df[feature_columns].values
    y = df['is_anomaly'].values
    
    # Isolation Forest
    logging.info("\nНавчання Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )
    
    X_normal = X[y == 0]
    iso_forest.fit(X_normal)
    
    iso_predictions = iso_forest.predict(X)
    iso_predictions = (iso_predictions == -1).astype(int)
    
    print("\nРезультати Isolation Forest:")
    print(classification_report(y, iso_predictions, target_names=['Normal', 'Anomaly']))
    
    return iso_predictions


if __name__ == "__main__":
    # Завантаження даних з ознаками
    logging.info("Завантаження даних...")
    df = pd.read_csv('datasets/smpp_weekly_dataset_features_optimized.csv', encoding='utf-8-sig')
    
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
    
    # Візуалізація навчання
    autoencoder.visualize_training()
    
    # Оптимізація порогу
    autoencoder.optimize_threshold(X, y)
    
    # Візуалізація результатів
    autoencoder.visualize_results(X, y)
    
    # Порівняння з Isolation Forest
    iso_predictions = compare_with_isolation_forest(df, feature_columns)
    
    # Збереження моделі
    autoencoder.save_model('models/smpp_autoencoder.pth')
    
    # Ансамбль: комбінація автоенкодера та Isolation Forest
    ae_predictions, ae_scores = autoencoder.predict(X)
    
    # Проста комбінація: якщо хоча б один метод каже що це аномалія
    ensemble_predictions = ((ae_predictions == 1) | (iso_predictions == 1)).astype(int)
    
    print("\nРезультати ансамблю (Автоенкодер + Isolation Forest):")
    print(classification_report(y, ensemble_predictions, target_names=['Normal', 'Anomaly']))

    df['ae_prediction'] = ae_predictions
    df['ae_score'] = ae_scores
    df['iso_prediction'] = iso_predictions
    df['ensemble_prediction'] = ensemble_predictions

    # Створення директорії, якщо не існує
    os.makedirs('results', exist_ok=True)

    # Назва файлу з таймштампом
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f'results/predictions_{timestamp}.csv'

    # Збереження у файл
    df.to_csv(result_filename, index=False, encoding='utf-8-sig')
    logging.info(f"Результати збережено у файл: {result_filename}")