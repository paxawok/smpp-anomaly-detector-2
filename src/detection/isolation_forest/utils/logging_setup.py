import logging
import sys
from pathlib import Path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Налаштування логера"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Якщо вже має обробники, не додаємо нові
    if logger.handlers:
        return logger
    
    # Консольний обробник
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Формат
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger