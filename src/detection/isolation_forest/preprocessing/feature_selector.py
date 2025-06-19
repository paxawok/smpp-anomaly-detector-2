import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

class FeatureSelector:
    """Вибір оптимальних ознак для IF"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.selected_features = []
        self.logger = logging.getLogger(__name__)
    
    def select_features(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """Вибір та підготовка ознак"""
        # 1. Базові ознаки з конфігу
        base_features = self.config['selected_features']
        available_features = [f for f in base_features if f in df.columns]
        
        # 2. Створення композитних ознак
        df_enhanced = self._create_composite_features(df)
        
        # 3. Додавання композитних ознак до списку
        composite_config = self.config.get('composite_features', {})
        for feature_name, feature_config in composite_config.items():
            if feature_config['enabled'] and feature_name in df_enhanced.columns:
                available_features.append(feature_name)
        
        # 4. Фільтрація корельованих ознак
        if self.config['preprocessing']['remove_correlated']:
            available_features = self._remove_correlated(df_enhanced, available_features)
        
        self.selected_features = available_features
        self.logger.info(f"Вибрано {len(available_features)} ознак: {available_features}")
        
        return available_features, df_enhanced[available_features]
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Трансформація нових даних"""
        df_enhanced = self._create_composite_features(df)
        return df_enhanced[self.selected_features]
    
    def _create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Створення композитних ознак"""
        df_result = df.copy()
        
        composite_config = self.config.get('composite_features', {})
        
        for feature_name, feature_config in composite_config.items():
            if not feature_config['enabled']:
                continue
                
            components = feature_config['components']
            weights = feature_config['weights']
            
            # Перевіряємо наявність всіх компонентів
            if all(comp in df.columns for comp in components):
                if feature_name == "total_suspicion_score":
                    df_result[feature_name] = (
                        df[components[0]] * weights[0] +
                        df[components[1]] * weights[1] + 
                        df[components[2]] * weights[2]
                    )
                    self.logger.info(f"Створено композитну ознаку: {feature_name}")
        
        return df_result
    
    def _remove_correlated(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Видалення корельованих ознак"""
        threshold = self.config['preprocessing']['correlation_threshold']
        
        corr_matrix = df[features].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        filtered_features = [f for f in features if f not in to_drop]
        
        if to_drop:
            self.logger.info(f"Видалено корельовані ознаки: {to_drop}")
            
        return filtered_features