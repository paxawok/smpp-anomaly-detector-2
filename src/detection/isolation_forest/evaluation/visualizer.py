import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Optional
import matplotlib
matplotlib.use('Agg')

class Visualizer:
    """Візуалізація результатів"""
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None):
        """Матриця плутанини"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.ylabel('Справжні значення')
        plt.xlabel('Передбачені значення')
        plt.title('Матриця плутанини')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_score_distribution(self, y_true: np.ndarray, scores: np.ndarray, 
                              threshold: float, save_path: Optional[str] = None):
        """Розподіл аномальних балів"""
        plt.figure(figsize=(10, 6))
        
        # Розподіл для нормальних та аномальних
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
        plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
        
        # Лінія порогу
        plt.axvline(x=threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Threshold: {threshold:.4f}')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Розподіл аномальних балів')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()