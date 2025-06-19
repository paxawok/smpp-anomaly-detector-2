import torch
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelPersistence:
    """Збереження та завантаження моделей"""
    
    @staticmethod
    def save_model(model, preprocessor, threshold, config, history, 
                   save_path: str = "models") -> str:
        """Збереження повної моделі"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smpp_autoencoder_{timestamp}.pt"
        filepath = save_dir / filename
        
        # Підготовка даних для збереження
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'preprocessor': preprocessor,
            'threshold': threshold,
            'history': history,
            'timestamp': timestamp,
            'version': '2.0'
        }
        
        # Збереження
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Збереження метаданих окремо
        metadata_path = save_dir / f"metadata_{timestamp}.json"
        metadata = {
            'model_path': str(filepath),
            'timestamp': timestamp,
            'config': config,
            'threshold': float(threshold) if threshold else None,
            'metrics': history.get('best_metrics', {})
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(filepath)
    
    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """Завантаження моделі"""
        logger.info(f"Loading model from {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        return checkpoint