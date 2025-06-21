"""
Configuration module for SMPP Autoencoder

This module provides centralized configuration management for the anomaly detection system.
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional

class ConfigManager:
    """Менеджер для роботи з конфігураційними файлами"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Завантаження всіх конфігураційних файлів"""
        config_files = {
            'model': 'model_config.json',
            'training': 'training_config.json',
            'features': 'features_config.json',
            'preprocessing': 'preprocessing_config.json'
        }
        
        for name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._configs[name] = json.load(f)
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Отримання конфігурації за назвою"""
        return self._configs.get(name, {})
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Отримання всіх конфігурацій"""
        return self._configs.copy()
    
    def update_config(self, name: str, updates: Dict[str, Any]):
        """Оновлення конфігурації"""
        if name in self._configs:
            self._configs[name].update(updates)
    
    def save_config(self, name: str):
        """Збереження конфігурації у файл"""
        if name in self._configs:
            config_path = self.config_dir / f"{name}_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._configs[name], f, indent=2, ensure_ascii=False)

# Глобальний екземпляр для зручності
config_manager = ConfigManager()

# Експорт основних функцій
def get_model_config() -> Dict[str, Any]:
    """Отримання конфігурації моделі"""
    return config_manager.get_config('model')

def get_training_config() -> Dict[str, Any]:
    """Отримання конфігурації навчання"""
    return config_manager.get_config('training')

def get_features_config() -> Dict[str, Any]:
    """Отримання конфігурації ознак"""
    return config_manager.get_config('features')

def get_preprocessing_config() -> Dict[str, Any]:
    """Отримання конфігурації препроцесингу"""
    return config_manager.get_config('preprocessing')

__all__ = [
    'ConfigManager',
    'config_manager',
    'get_model_config',
    'get_training_config',
    'get_features_config',
    'get_preprocessing_config'
]