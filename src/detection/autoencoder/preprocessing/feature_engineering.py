import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class FeatureEngineer:
    """Створення додаткових features"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Створення всіх додаткових features"""
        df = df.copy()
        
        # Циклічні features
        df = self._create_cyclical_features(df)
        
        # Композитні features
        df = self._create_composite_features(df)
        
        # Статистичні features
        df = self._create_statistical_features(df)
        
        return df
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Циклічне кодування часових features"""
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Композитні features"""
        # Складність повідомлення
        if all(col in df.columns for col in ['message_entropy', 'obfuscation_score', 'url_count']):
            df['complexity_score'] = (
                df['message_entropy'] * 0.3 +
                df['obfuscation_score'] * 0.3 +
                (df['url_count'] > 0).astype(float) * 0.2 +
                df['suspicious_word_count'] / 10 * 0.2
            )
        
        # Ризик відправника
        if all(col in df.columns for col in ['sender_legitimacy', 'unusual_sender_pattern']):
            df['sender_risk'] = (
                (1 - df['sender_legitimacy']) * 0.4 +
                df['unusual_sender_pattern'] * 0.3 +
                df['source_category_mismatch'] * 0.3
            )
        
        # Часова аномалія
        if all(col in df.columns for col in ['night_time', 'time_category_anomaly']):
            df['time_risk'] = (
                df['night_time'] * 0.3 +
                df['time_category_anomaly'] * 0.4 +
                df['category_time_mismatch'] * 0.3
            )
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Статистичні features"""
        # Z-score для довжини повідомлення
        if 'message_length' in df.columns:
            mean_len = df['message_length'].mean()
            std_len = df['message_length'].std()
            if std_len > 0:
                df['message_length_zscore'] = (df['message_length'] - mean_len) / std_len
        
        # Співвідношення
        if 'message_entropy' in df.columns and 'message_length' in df.columns:
            df['entropy_length_ratio'] = df['message_entropy'] / np.log(df['message_length'] + 1)
        
        return df