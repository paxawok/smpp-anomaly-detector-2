"""
SMPP Anomaly Detector - головний модуль
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

from src.detection.autoencoder.config_ae import get_model_config, get_training_config, get_features_config
from src.detection.autoencoder.models_ae import ImprovedSMPPAutoencoder
from src.detection.autoencoder.preprocessing import AdvancedDataPreprocessor
from src.detection.autoencoder.training import Trainer
from src.detection.autoencoder.evaluation import ThresholdOptimizer, calculate_metrics, Visualizer
from src.detection.autoencoder.utils import DataLoader, ModelPersistence, setup_logger
import warnings
warnings.filterwarnings('ignore')


logger = setup_logger('detector')

class SMPPAnomalyDetector:
    """Головний клас для виявлення аномалій в SMPP"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Завантаження конфігурацій
        self.model_config = get_model_config()
        self.training_config = get_training_config()
        self.features_config = get_features_config()
        
        # Компоненти
        self.model = None
        self.preprocessor = AdvancedDataPreprocessor()
        self.threshold_optimizer = ThresholdOptimizer(
            percentile=self.training_config.get('threshold_percentile', 95)
        )
        self.data_loader = DataLoader()
        self.visualizer = Visualizer()
        
        # Стан
        self.is_trained = False
        self.threshold = None
        self.history = {}
        
        logger.info(f"Initialized detector with device: {self.device}")
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Повний pipeline навчання"""
        logger.info("Starting training pipeline...")
        
        # 1. Фільтрація нормальних даних
        df_normal = df[df['is_anomaly'] == False].copy()
        logger.info(f"Training on {len(df_normal)} normal samples")
        
        # 2. Препроцесинг
        X = self.preprocessor.preprocess(df_normal, fit=True)
        logger.info(f"Preprocessed shape: {X.shape}")
        
        # 3. Створення моделі
        self.model_config['architecture']['input_dim'] = X.shape[1]
        self.model = ImprovedSMPPAutoencoder(self.model_config)
        
        # 4. Підготовка даних
        train_loader, val_loader = self.data_loader.prepare_dataloaders(
            X, 
            batch_size=self.training_config['batch_size'],
            validation_split=0.2,
            augment=True
        )
        
        # 5. Навчання
        trainer = Trainer(self.model, self.training_config, self.device)
        self.history = trainer.train(
            train_loader, 
            val_loader, 
            epochs=self.training_config['epochs']
        )
        
        # 6. Визначення порогу
        self._determine_threshold(val_loader)
        
        # 7. Візуалізація
        self.visualizer.plot_training_history(self.history)
        
        self.is_trained = True
        logger.info("Training completed successfully")
        
        return self.history
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Предикція аномалій"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Препроцесинг
        X = self.preprocessor.preprocess(df, fit=False)
        
        # Отримання помилок реконструкції
        errors = self._get_reconstruction_errors(X)
        
        # Предикція
        predictions = self.threshold_optimizer.predict(errors)
        
        # Статистика
        stats = {
            'total_samples': len(predictions),
            'anomalies_detected': int(predictions.sum()),
            'anomaly_rate': float(predictions.mean()),
            'mean_error': float(errors.mean()),
            'threshold': float(self.threshold)
        }
        
        return predictions, errors, stats
    
    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Оцінка моделі"""
        predictions, errors, _ = self.predict(df)
        
        if 'is_anomaly' in df.columns:
            y_true = df['is_anomaly'].values
            metrics = calculate_metrics(y_true, predictions, errors)
            
            # Візуалізація
            self.visualizer.plot_confusion_matrix(y_true, predictions)
            self.visualizer.plot_roc_curve(y_true, errors)
            
            return metrics
        else:
            logger.warning("No ground truth labels for evaluation")
            return {}
    
    def save(self, save_path: str = "models") -> str:
        """Збереження моделі"""
        return ModelPersistence.save_model(
            self.model,
            self.preprocessor,
            self.threshold,
            {
                'model_config': self.model_config,
                'training_config': self.training_config
            },
            self.history,
            save_path
        )
    
    def load(self, model_path: str):
        """Завантаження моделі"""
        checkpoint = ModelPersistence.load_model(model_path)
        
        # Відновлення компонентів
        self.model_config = checkpoint['model_config']
        self.model = ImprovedSMPPAutoencoder(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        self.preprocessor = checkpoint['preprocessor']
        self.threshold = checkpoint['threshold']
        self.history = checkpoint.get('history', {})
        
        self.threshold_optimizer.threshold = self.threshold
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
    
    def _determine_threshold(self, val_loader):
        """Визначення порогу"""
        errors = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(self.device)
                errors.extend(self.model.get_reconstruction_error(inputs).cpu().numpy())
        
        errors = np.array(errors)
        self.threshold = self.threshold_optimizer.fit(errors)
        
        # Візуалізація
        self.visualizer.plot_error_distribution(errors, self.threshold)
    
    def _get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Отримання помилок реконструкції"""
        self.model.eval()
        errors = []
        
        # Батчована обробка
        batch_size = self.training_config['batch_size'] * 4
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))
                
                batch = torch.FloatTensor(X[start_idx:end_idx]).to(self.device)
                batch_errors = self.model.get_reconstruction_error(batch)
                errors.extend(batch_errors.cpu().numpy())
        
        return np.array(errors)


# Головна функція для швидкого запуску
def main():
    """Приклад використання детектора"""
    # Ініціалізація
    detector = SMPPAnomalyDetector()
    
    # Завантаження даних
    df = pd.read_csv('data/datasets/smpp_weekly_dataset_features_optimized.csv', 
                     encoding='utf-8-sig')
    
    # Навчання
    history = detector.train(df)
    
    # Оцінка
    metrics = detector.evaluate(df)
    print(f"Metrics: {metrics}")
    
    # Збереження
    model_path = detector.save()
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()