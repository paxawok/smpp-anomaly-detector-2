import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Dict

class AdaptiveScaler(BaseEstimator, TransformerMixin):
    """Адаптивний scaler що вибирає метод масштабування для кожної ознаки"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_scalers = {}
        
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None):
        """Fit різні scalers для різних типів features"""
        n_features = X.shape[1]
        
        for i in range(n_features):
            feature_data = X[:, i].reshape(-1, 1)
            
            # Визначаємо тип розподілу
            if self._is_binary(feature_data):
                # Бінарні - без масштабування
                self.feature_scalers[i] = None
            elif self._has_outliers(feature_data):
                # З викидами - RobustScaler
                scaler = RobustScaler()
                scaler.fit(feature_data)
                self.feature_scalers[i] = scaler
            else:
                # Звичайні - StandardScaler
                scaler = StandardScaler()
                scaler.fit(feature_data)
                self.feature_scalers[i] = scaler
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform з різними scalers"""
        X_scaled = X.copy()
        
        for i in range(X.shape[1]):
            if i in self.feature_scalers and self.feature_scalers[i] is not None:
                X_scaled[:, i] = self.feature_scalers[i].transform(X[:, i].reshape(-1, 1)).ravel()
        
        return X_scaled
    
    def _is_binary(self, feature: np.ndarray) -> bool:
        """Перевірка чи feature бінарна"""
        unique_values = np.unique(feature[~np.isnan(feature)])
        return len(unique_values) <= 2
    
    def _has_outliers(self, feature: np.ndarray) -> bool:
        """Перевірка на наявність викидів через IQR"""
        q1 = np.percentile(feature, 25)
        q3 = np.percentile(feature, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.sum((feature < lower_bound) | (feature > upper_bound))
        return outliers > len(feature) * 0.05  # >5% викидів