import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict

class MetricsCalculator:
    """Розрахунок метрик"""
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         scores: np.ndarray) -> Dict[str, float]:
        """Розрахунок всіх метрик"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, -scores) if len(np.unique(y_true)) > 1 else 0
        }