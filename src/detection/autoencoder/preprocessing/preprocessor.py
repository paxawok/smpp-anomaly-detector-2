import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class AdvancedDataPreprocessor:
    """Препроцесор для SMPP даних"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scalers = {'standard': StandardScaler()}
        self.label_encoders = {}
        self.feature_stats = defaultdict(dict)
        self.fitted = False
        
        # Порядок features для консистентності
        self.numeric_features_order = []
        self.binary_features_order = []
        self.categorical_features_fitted = []
        self.n_features_fitted = None
        
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Основний метод препроцесингу"""
        logger.info(f"Preprocessing data with shape: {df.shape}")
        df = df.copy()
        
        # Завантаження конфігурації features
        from ..config_ae import get_features_config
        features_config = get_features_config()
        feature_columns = features_config['feature_columns']
        
        # Категорії features
        numeric_features = [col for col in feature_columns if col in df.columns and col not in ['day_of_week', 'time_category_anomaly'] and df[col].dtype in ['int64', 'float64']]
        binary_features = [col for col in feature_columns if col in df.columns and df[col].nunique() == 2]
        categorical_features = ['day_of_week', 'time_category_anomaly'] if 'day_of_week' in df.columns else []
        
        # Очистка даних
        df = self._clean_data(df)
        
        processed_features = []
        
        # 1. Числові features
        if numeric_features:
            numeric_data = df[numeric_features].values
            if fit:
                numeric_scaled = self.scalers['standard'].fit_transform(numeric_data)
                self.numeric_features_order = numeric_features
            else:
                if hasattr(self, 'numeric_features_order'):
                    numeric_data = df[self.numeric_features_order].values
                    numeric_scaled = self.scalers['standard'].transform(numeric_data)
                else:
                    numeric_scaled = np.array([])
            
            if numeric_scaled.size > 0:
                processed_features.append(numeric_scaled)
        
        # 2. Бінарні features
        if fit:
            self.binary_features_order = binary_features
        
        if hasattr(self, 'binary_features_order') and self.binary_features_order:
            binary_data = df[self.binary_features_order].fillna(0).values.astype(float)
            processed_features.append(binary_data)
        
        # 3. Категоріальні features
        if fit:
            self.categorical_features_fitted = []
        
        for col in categorical_features:
            if col in df.columns:
                if fit:
                    self.categorical_features_fitted.append(col)
                    col_data = df[col].fillna('unknown').astype(str)
                    
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    
                    unique_values = list(col_data.unique())
                    if 'unknown' not in unique_values:
                        unique_values.append('unknown')
                    self.label_encoders[col].fit(unique_values)
                    encoded = self.label_encoders[col].transform(col_data)
                    
                    # One-hot encoding
                    n_categories = len(self.label_encoders[col].classes_)
                    one_hot = np.zeros((len(df), n_categories))
                    one_hot[np.arange(len(df)), encoded] = 1
                    processed_features.append(one_hot)
                else:
                    if hasattr(self, 'categorical_features_fitted') and col in self.categorical_features_fitted:
                        col_data = df[col].fillna('unknown').astype(str)
                        le = self.label_encoders[col]
                        known_labels = set(le.classes_)
                        col_data_safe = col_data.apply(lambda x: x if x in known_labels else 'unknown')
                        
                        if 'unknown' not in known_labels:
                            col_data_safe = col_data_safe.replace('unknown', le.classes_[0])
                        
                        encoded = le.transform(col_data_safe)
                        n_categories = len(le.classes_)
                        one_hot = np.zeros((len(df), n_categories))
                        one_hot[np.arange(len(df)), encoded] = 1
                        processed_features.append(one_hot)
        
        # Об'єднання
        if processed_features:
            X = np.hstack(processed_features)
        else:
            X = np.zeros((len(df), 1))
        
        if fit:
            self.n_features_fitted = X.shape[1]
            self.fitted = True
            logger.info(f"Fitted preprocessor expects {self.n_features_fitted} features")
        
        return X
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка даних"""
        # Заповнити NaN медіаною одразу для всіх числових
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df