import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

class Visualizer:
    """Візуалізація результатів"""
    
    def __init__(self, save_dir: str = "data/plots/ae"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_training_history(self, history: Dict[str, List]) -> str:
        """Графіки історії навчання"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(history['learning_rates'], label='Learning Rate', linewidth=2, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.save_dir / f'training_history_{self.timestamp}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_error_distribution(self, errors: np.ndarray, threshold: float) -> str:
        """Розподіл помилок реконструкції"""
        plt.figure(figsize=(10, 6))
        
        sns.histplot(errors, bins=50, kde=True, color='blue', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold:.6f}')
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = self.save_dir / f'error_distribution_{self.timestamp}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        filepath = self.save_dir / f'confusion_matrix_{self.timestamp}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_roc_curve(self, y_true: np.ndarray, scores: np.ndarray) -> str:
        """ROC крива"""
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        filepath = self.save_dir / f'roc_curve_{self.timestamp}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)