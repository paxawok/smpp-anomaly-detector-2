import os
import json
import logging
from datetime import datetime
import pandas as pd
import sqlite3
from src.detection.anomaly_detector.autoencoder import SMPPAutoencoderDetector
from src.detection.anomaly_detector.isolation_forest import IsolationForestTrainer

def main():
    # Завантаження конфігурації
    with open('detector_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 1. Навчання Isolation Forest
    if_trainer = IsolationForestTrainer(
        input_file='datasets/smpp_dataset_features.csv',
        config_file='detector_config.json'
    )
    if_trainer.run_training_pipeline()  # або train_models(), залежно від вашої реалізації

    # 2. Навчання Autoencoder
    ae_trainer = SMPPAutoencoderDetector(
        input_file='datasets/smpp_dataset_features.csv',
        config=config['autoencoder'],
        feature_columns=config['feature_columns'],
        models_dir=config['paths']['models_dir']
    )
    ae_trainer.run_training_pipeline()  # або train(), залежно від вашої реалізації

    # 3. Інференс: отримати оцінки для всіх повідомлень
    df = pd.read_csv('datasets/smpp_dataset_features.csv')
    # ...отримати features...

    # Приклад: отримати оцінки
    if_scores = if_trainer.predict_batch(df[config['feature_columns']].values)
    ae_scores = ae_trainer.predict_batch(df[config['feature_columns']].values)

    # 4. Формування фінальної таблиці
    result_df = df.copy()
    result_df['if_score'] = [s['anomaly_score'] for s in if_scores]
    result_df['ae_score'] = [s['anomaly_score'] for s in ae_scores]
    result_df['ensemble_score'] = 0.6 * result_df['if_score'] + 0.4 * result_df['ae_score']

    # 5. Збереження у SQLite
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db_dir = config['paths']['db_dir']
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f'smpp_dataset_{timestamp}.db')

    with sqlite3.connect(db_path) as conn:
        result_df.to_sql('smpp_anomaly_scores', conn, index=False, if_exists='replace')

    print(f"Фінальна таблиця збережена у {db_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()