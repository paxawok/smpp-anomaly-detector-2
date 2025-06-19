import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Розрахунок всіх метрик"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # ROC AUC якщо є scores
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, scores)
        except:
            metrics['roc_auc'] = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
    
    return metrics

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision для топ-K аномалій"""
    if k > len(scores):
        k = len(scores)
    
    # Індекси топ-K scores
    top_k_indices = np.argsort(scores)[-k:]
    
    # Precision
    return np.mean(y_true[top_k_indices])

def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Детальний звіт класифікації"""
    return classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Anomaly'],
        digits=4
    )