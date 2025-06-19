import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, List
import logging

class OptimizedIsolationForest:
    """Оптимізований Isolation Forest з ensemble"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = []
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: np.ndarray):
        """Навчання ensemble моделей"""
        ensemble_size = self.config['ensemble_size']
        base_params = self.config['base_params']
        
        for i in range(ensemble_size):
            # Варіюємо random_state для різноманітності
            params = base_params.copy()
            params['random_state'] = base_params['random_state'] + i
            
            model = IsolationForest(**params)
            model.fit(X)
            self.models.append(model)
            
        self.logger.info(f"Навчано {ensemble_size} моделей")
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Ensemble передбачення"""
        all_scores = []
        
        for model in self.models:
            scores = model.score_samples(X)
            all_scores.append(scores)
        
        # Усереднення scores
        ensemble_scores = np.mean(all_scores, axis=0)
        return ensemble_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Бінарні передбачення"""
        all_predictions = []
        
        for model in self.models:
            predictions = model.predict(X)
            all_predictions.append(predictions)
        
        # Мажоритарне голосування
        ensemble_predictions = np.round(np.mean(all_predictions, axis=0))
        return ensemble_predictions.astype(int)