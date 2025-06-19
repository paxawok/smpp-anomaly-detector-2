import json
from pathlib import Path
from typing import Dict

__all__ = ['ConfigLoader']

class ConfigLoader:
    """Завантаження конфігурацій"""

    def __init__(self):
        self.config_dir = Path(__file__).resolve().parent
        print("CONFIG PATH:", self.config_dir)

    
    def _load_config(self, filename: str) -> Dict:
        """Завантаження JSON конфігу"""
        config_path = self.config_dir / filename
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_model_config(self) -> Dict:
        return self._load_config('model_config.json')
    
    def get_features_config(self) -> Dict:
        return self._load_config('features_config.json')
    
    def get_calibration_config(self) -> Dict:
        return self._load_config('calibration_config.json')