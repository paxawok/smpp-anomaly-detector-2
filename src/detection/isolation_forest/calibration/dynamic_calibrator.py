import numpy as np
from collections import deque
from typing import Dict, Optional
from scipy import stats
import logging
from sklearn.metrics import precision_recall_curve



class DynamicThresholdCalibrator:
    """Динамічна калібрація порогу"""
    
    def __init__(self, config: Dict):
        self.config = config['dynamic_calibration']
        self.drift_config = config['drift_detection']
        
        self.window_size = self.config['window_size']
        self.update_frequency = self.config['update_frequency']
        self.percentile = self.config['percentile_threshold']
        
        self.score_buffer = deque(maxlen=self.window_size)
        self.current_threshold = None
        self.samples_since_update = 0
        self.baseline_mean = None
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_threshold(self, scores: np.ndarray, 
                           y_true: Optional[np.ndarray] = None) -> float:
        """Початкова ініціалізація порогу"""
        if y_true is not None and len(np.unique(y_true)) > 1:
            threshold = self._optimize_with_labels(scores, y_true)
        else:
            threshold = -np.percentile(-scores, self.percentile)
        
        self.current_threshold = threshold
        self.baseline_mean = np.mean(scores)
        self.score_buffer.extend(scores)
        
        self.logger.info(f"Початковий поріг: {threshold:.6f}")
        return threshold
    
    def update_threshold(self, new_scores: np.ndarray) -> Optional[float]:
        """Оновлення порогу"""
        self.score_buffer.extend(new_scores)
        self.samples_since_update += len(new_scores)
        
        if self.samples_since_update < self.update_frequency:
            return None
        
        # Перевірка на дрейф
        if self.drift_config['enabled'] and self._detect_drift(new_scores):
            new_threshold = -np.percentile(-np.array(list(self.score_buffer)), self.percentile)
            
            if abs(new_threshold - self.current_threshold) > 0.01:
                self.logger.info(f"Поріг оновлено через дрейф: {self.current_threshold:.6f} → {new_threshold:.6f}")
                self.current_threshold = new_threshold
                self.samples_since_update = 0
                return new_threshold
        
        return None
    
    def _optimize_with_labels(self, scores: np.ndarray, y_true: np.ndarray) -> float:
        """Оптимізація з мітками"""

        from sklearn.metrics import precision_recall_curve
        
        anomaly_scores = -scores
        precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
        
        # Шукаємо поріг з precision >= 0.95
        #high_precision_idx = np.where(precision[:-1] >= 0.95)[0]

        target_precision = self.config.get("min_precision")
        high_precision_idx = np.where(precision[:-1] >= target_precision)[0]
        
        if len(high_precision_idx) > 0:
            best_idx = high_precision_idx[np.argmax(recall[high_precision_idx])]
        else:
            # Fallback до найкращого F1
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores[:-1])
        
        return -thresholds[best_idx]
    
    def _detect_drift(self, new_scores: np.ndarray) -> bool:
        """Виявлення дрейфу"""
        if len(new_scores) < 50 or self.baseline_mean is None:
            return False
        
        # Зміна середнього
        mean_shift = abs(np.mean(new_scores) - self.baseline_mean)
        threshold = self.drift_config['mean_shift_threshold']
        
        return mean_shift > threshold