import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """Оптимізація порогу для виявлення аномалій"""
    
    def __init__(self, percentile: float = 95, method: str = 'percentile'):
        self.percentile = percentile
        self.method = method
        self.threshold = None
        
    def fit(self, reconstruction_errors: np.ndarray, 
            y_true: Optional[np.ndarray] = None) -> float:
        """Визначення оптимального порогу"""
        
        if self.method == 'percentile':
            self.threshold = self._percentile_threshold(reconstruction_errors)
        elif self.method == 'optimal_f1' and y_true is not None:
            self.threshold = self._optimal_f1_threshold(reconstruction_errors, y_true)
        elif self.method == 'iqr':
            self.threshold = self._iqr_threshold(reconstruction_errors)
        else:
            self.threshold = self._percentile_threshold(reconstruction_errors)
        
        logger.info(f"Threshold set to: {self.threshold:.6f} using {self.method} method")
        return self.threshold
    
    def _percentile_threshold(self, errors: np.ndarray) -> float:
        """Поріг на основі percentile"""
        return np.percentile(errors, min(self.percentile, 98))
    
    def _optimal_f1_threshold(self, errors: np.ndarray, y_true: np.ndarray) -> float:
        """Оптимальний поріг для максимального F1"""
        precision, recall, thresholds = precision_recall_curve(y_true, errors)
        
        # F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Найкращий поріг
        best_idx = np.argmax(f1_scores[:-1])
        return thresholds[best_idx]
    
    def _iqr_threshold(self, errors: np.ndarray) -> float:
        """Поріг на основі IQR"""
        q1 = np.percentile(errors, 25)
        q3 = np.percentile(errors, 75)
        iqr = q3 - q1
        
        return q3 + 1.5 * iqr
    
    def predict(self, errors: np.ndarray) -> np.ndarray:
        """Предикція на основі порогу"""
        if self.threshold is None:
            raise ValueError("Threshold not fitted yet")
        
        return (errors > self.threshold).astype(int)
    
    def get_threshold_stats(self, errors: np.ndarray) -> Dict[str, float]:
        """Статистика порогу"""
        return {
            'threshold': self.threshold,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'percentile_95': np.percentile(errors, 95),
            'percentile_99': np.percentile(errors, 99)
        }