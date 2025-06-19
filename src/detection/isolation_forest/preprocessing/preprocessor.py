import numpy as np
import pandas as pd
from typing import Dict
import logging

class DataPreprocessor:
    """Препроцесинг даних для IF"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fitted = False
        self.logger = logging.getLogger(__name__)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Навчання та трансформація"""
        X = self._clean_data(df)
        self.fitted = True
        return X
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Трансформація нових даних"""
        if not self.fitted:
            raise ValueError("Preprocessor не навчений")
        return self._clean_data(df)
    
    def _clean_data(self, df: pd.DataFrame) -> np.ndarray:
        """Очищення даних"""
        # Заповнення пропущених значень
        df_clean = df.fillna(0)
        
        # Перетворення в numpy array
        X = df_clean.values.astype(np.float32)
        
        # Заміна inf значень
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.logger.info(f"Дані очищено. Форма: {X.shape}")
        return X