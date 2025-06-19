import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class ModelPersistence:
    """Збереження та завантаження моделі"""
    
    @staticmethod
    def save_model(model, preprocessor, feature_selector, threshold, 
                   configs: Dict, save_path: str) -> str:
        """Збереження моделі"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"isolation_forest_{timestamp}.pkl"
        filepath = save_dir / filename
        
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'feature_selector': feature_selector,
            'threshold': threshold,
            'configs': configs,
            'timestamp': timestamp
        }
        
        joblib.dump(model_data, filepath)
        return str(filepath)
    
    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """Завантаження моделі"""
        return joblib.load(filepath)