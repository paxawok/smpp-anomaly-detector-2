import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')


class Visualizer:
    """Візуалізація результатів з автозбереженням"""

    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(f"data/plots/isolation_forest/{timestamp}")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Матриця плутанини"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        plt.ylabel('Справжні значення')
        plt.xlabel('Передбачені значення')
        plt.title('Матриця плутанини')

        save_path = self.save_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_score_distribution(self, y_true: np.ndarray, scores: np.ndarray, threshold: float):
        """Розподіл аномальних балів"""
        plt.figure(figsize=(10, 6))

        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]

        plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
        plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')

        plt.axvline(x=threshold, color='green', linestyle='--',
                    linewidth=2, label=f'Threshold: {threshold:.4f}')

        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Розподіл аномальних балів')
        plt.legend()

        save_path = self.save_dir / "score_distribution.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
