"""
Головний клас для виявлення аномалій за допомогою Isolation Forest
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from .config_if import ConfigLoader
from .models_if import OptimizedIsolationForest
from .preprocessing import FeatureSelector, DataPreprocessor
from .calibration import DynamicThresholdCalibrator
from .evaluation import MetricsCalculator, Visualizer
from .utils import ModelPersistence, setup_logger

logger = setup_logger('isolation_forest')

class IsolationForestDetector:
    """Детектор аномалій на основі Isolation Forest"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_loader = ConfigLoader(config_dir)
        
        # Завантаження конфігурацій
        self.model_config = self.config_loader.get_model_config()
        self.features_config = self.config_loader.get_features_config()
        self.calibration_config = self.config_loader.get_calibration_config()
        
        # Ініціалізація компонентів
        self.feature_selector = FeatureSelector(self.features_config)
        self.preprocessor = DataPreprocessor(self.features_config)
        self.model = OptimizedIsolationForest(self.model_config)
        self.calibrator = DynamicThresholdCalibrator(self.calibration_config)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # Стан
        self.is_trained = False
        self.selected_features = []
        self.threshold = None
        
        logger.info("IsolationForestDetector ініціалізовано")
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Навчання моделі"""
        logger.info("Початок навчання Isolation Forest...")
        
        # 1. Підготовка даних (тільки нормальні для IF)
        df_normal = df[df['is_anomaly'] == False].copy()
        logger.info(f"Навчання на {len(df_normal)} нормальних зразках")
        
        # 2. Вибір ознак
        self.selected_features, X_selected = self.feature_selector.select_features(df_normal)
        
        # 3. Препроцесинг
        X_processed = self.preprocessor.fit_transform(X_selected)
        
        # 4. Навчання моделі
        self.model.fit(X_processed)
        
        # 5. Калібрація порогу на всіх даних
        X_all = self.preprocessor.transform(
            self.feature_selector.transform(df)
        )
        scores = self.model.score_samples(X_all)
        y_all = df['is_anomaly'].values.astype(int)
        
        self.threshold = self.calibrator.initialize_threshold(scores, y_all)
        
        self.is_trained = True
        logger.info("Навчання завершено")
        
        return {"threshold": self.threshold, "n_features": len(self.selected_features)}
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Передбачення аномалій"""
        if not self.is_trained:
            raise ValueError("Модель не навчена")
        
        # Підготовка даних
        X_selected = self.feature_selector.transform(df)
        X_processed = self.preprocessor.transform(X_selected)
        
        # Отримання scores
        scores = self.model.score_samples(X_processed)
        
        # Передбачення
        predictions = (scores < self.threshold).astype(int)
        
        # Динамічне оновлення порогу
        new_threshold = self.calibrator.update_threshold(scores)
        if new_threshold is not None:
            self.threshold = new_threshold
            predictions = (scores < self.threshold).astype(int)
        
        # Статистика
        stats = {
            'total_samples': len(predictions),
            'anomalies_detected': int(predictions.sum()),
            'anomaly_rate': float(predictions.mean()),
            'threshold': float(self.threshold)
        }
        
        return predictions, scores, stats
    
    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Оцінка моделі"""
        predictions, scores, _ = self.predict(df)
        
        if 'is_anomaly' in df.columns:
            y_true = df['is_anomaly'].values.astype(int)
            metrics = self.metrics_calculator.calculate_metrics(y_true, predictions, scores)
            
            # Візуалізація
            self.visualizer.plot_confusion_matrix(y_true, predictions)
            self.visualizer.plot_score_distribution(y_true, scores, self.threshold)
            
            return metrics
        else:
            logger.warning("Немає міток для оцінки")
            return {}
    
    def save(self, save_path: str = "models") -> str:
        """Збереження моделі"""
        return ModelPersistence.save_model(
            model=self.model,
            preprocessor=self.preprocessor,
            feature_selector=self.feature_selector,
            threshold=self.threshold,
            configs={
                'model_config': self.model_config,
                'features_config': self.features_config,
                'calibration_config': self.calibration_config
            },
            save_path=save_path
        )
    
    def load(self, model_path: str):
        """Завантаження моделі"""
        checkpoint = ModelPersistence.load_model(model_path)
        
        self.model = checkpoint['model']
        self.preprocessor = checkpoint['preprocessor']
        self.feature_selector = checkpoint['feature_selector']
        self.threshold = checkpoint['threshold']
        
        self.is_trained = True
        logger.info(f"Модель завантажена з {model_path}")


def main():
    """Приклад використання"""
    detector = IsolationForestDetector()
    
    # Завантаження даних
    df = pd.read_csv('data/datasets/smpp_weekly_dataset_features_optimized.csv', 
                     encoding='utf-8-sig')
    
    # Навчання
    train_results = detector.train(df)
    logger.info(f"Навчання завершено: {train_results}")
    
    # Оцінка
    metrics = detector.evaluate(df)
    logger.info(f"Метрики: {metrics}")
    
    # Збереження
    model_path = detector.save()
    logger.info(f"Модель збережена: {model_path}")


if __name__ == "__main__":
    main()