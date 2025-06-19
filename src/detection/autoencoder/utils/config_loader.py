import json
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """Завантаження JSON конфігурацій"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache = {}
    
    def load(self, config_name: str) -> Dict[str, Any]:
        """Завантаження конфігу"""
        if config_name in self._cache:
            return self._cache[config_name]
        
        config_path = self.config_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self._cache[config_name] = config
        return config
    
    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """Отримання значення з конфігу"""
        config = self.load(config_name)
        return config.get(key, default)