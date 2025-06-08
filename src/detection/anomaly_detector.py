import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time
import math
import warnings
warnings.filterwarnings('ignore')

# Константи
FEATURE_COLUMNS = [
   "hour", "day_of_week", "message_length", "source_addr_length", 
   "source_is_numeric", "dest_is_valid", "message_parts", "encoding_issues", 
   "empty_message", "excessive_length", "sender_frequency", "recipient_frequency", 
   "sender_burst", "recipient_burst", "high_sender_frequency", 
   "high_recipient_frequency", "suspicious_word_count", "url_count", 
   "suspicious_url", "urgency_score", "message_entropy", "obfuscation_score", 
   "social_engineering", "night_time", "weekend", "business_hours", 
   "time_category_anomaly", "sender_legitimacy", "financial_patterns",
   "phone_numbers", "source_category_mismatch", "category_time_mismatch",
   "unusual_sender_pattern", "typosquatting"
]


class AdvancedDataPreprocessor:
    pass

import __main__
__main__.AdvancedDataPreprocessor = AdvancedDataPreprocessor

class SMPPAnomalyDetectionPipeline:
    def __init__(self, db_path: str, autoencoder_path: str, isolation_forest_path: str):
        self.db_path = db_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- 1. Завантажуємо чекпоінт ---
        checkpoint = torch.load(
            autoencoder_path,
            map_location=self.device,
            weights_only=False
        )

        # Беремо з чекпоінта ВСІ потрібні параметри:
        model_cfg = {
            **checkpoint['model_config'],               # має містити input_dim, hidden_dims, encoding_dim=32
            'dropout_rate': checkpoint['model_config'].get('dropout_rate', 0.3),
            'use_attention': checkpoint['model_config'].get('use_attention', True),
        }

        # Тепер будуємо і завантажуємо
        self.ae_model = self._build_autoencoder(model_cfg)
        self.ae_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.ae_model.to(self.device).eval()
        self.ae_threshold = checkpoint['threshold']
        self.ae_input_dim = model_cfg['input_dim']


        
        # Завантажуємо дані з pickle
        if_data = joblib.load(isolation_forest_path)

        raw_models = if_data['models']
        if isinstance(raw_models, dict):
            # словник → беремо лише значення (моделі)
            self.if_models = list(raw_models.values())
        elif isinstance(raw_models, list):
            self.if_models = raw_models
        else:
            # навзаєм, якщо там один об’єкт
            self.if_models = [raw_models]

        self.if_scaler    = if_data['scaler']
        self.if_threshold = if_data['optimal_threshold']
        self.if_features  = if_data.get('feature_names', FEATURE_COLUMNS)


       
    def _build_autoencoder(self, cfg: Dict) -> nn.Module:
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dims, encoding_dim,
                        dropout_rate, use_attention):
                super().__init__()
                # --- ENCODER ---
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim,    hidden_dims[0]),  # encoder.0
                    nn.BatchNorm1d(hidden_dims[0]),           # encoder.1
                    nn.LeakyReLU(0.2),                        # encoder.2
                    nn.Dropout(dropout_rate),                 # encoder.3

                    nn.Linear(hidden_dims[0], hidden_dims[1]),# encoder.4
                    nn.BatchNorm1d(hidden_dims[1]),           # encoder.5
                    nn.LeakyReLU(0.2),                        # encoder.6
                    nn.Dropout(dropout_rate),                 # encoder.7

                    nn.Linear(hidden_dims[1], hidden_dims[2]),# encoder.8
                    nn.BatchNorm1d(hidden_dims[2]),           # encoder.9
                    nn.LeakyReLU(0.2),                        # encoder.10
                    nn.Dropout(dropout_rate),                 # encoder.11
                )

                # --- ATTENTION (якщо увімкнено) ---
                self.use_attention = use_attention
                if use_attention:
                    self.attention = nn.MultiheadAttention(
                        embed_dim=hidden_dims[2],
                        num_heads=1,
                        batch_first=True
                    )

                # --- BOTTLENECK ---
                self.encoding_layer = nn.Linear(hidden_dims[2], encoding_dim)

                # --- ДЕКОДЕР: 32→64; 64→128→256→43 ---
                self.decoding_layer = nn.Linear(encoding_dim, hidden_dims[2])  # (32→64)

                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dims[2], hidden_dims[1]), # decoder.0
                    nn.BatchNorm1d(hidden_dims[1]),            # decoder.1
                    nn.LeakyReLU(0.2),                         # decoder.2
                    nn.Dropout(dropout_rate),                  # decoder.3

                    nn.Linear(hidden_dims[1], hidden_dims[0]), # decoder.4
                    nn.BatchNorm1d(hidden_dims[0]),            # decoder.5
                    nn.LeakyReLU(0.2),                         # decoder.6
                    nn.Dropout(dropout_rate),                  # decoder.7

                    nn.Linear(hidden_dims[0], input_dim)       # decoder.8
                )

            def forward(self, x):
                x_enc = self.encoder(x)
                if self.use_attention:
                    att_in  = x_enc.unsqueeze(1)             # [B,1,64]
                    att_out, _ = self.attention(att_in, att_in, att_in)
                    x_enc = att_out.squeeze(1)
                z   = self.encoding_layer(x_enc)            # [B,32]
                y   = self.decoding_layer(z)                # [B,64]
                out = self.decoder(y)                       # [B,43]
                return out, z

        return Autoencoder(
            input_dim     = cfg['input_dim'],       # 43
            hidden_dims   = cfg['hidden_dims'],     # [256,128,64]
            encoding_dim  = cfg['encoding_dim'],    # 32
            dropout_rate  = cfg.get('dropout_rate', 0.0),
            use_attention = cfg.get('use_attention', False)
        )



    def analyze_message(self, message_id: int) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        start = time.time()
       # Отримання даних
        cursor.execute("SELECT * FROM smpp_messages WHERE id = ?", (message_id,))
        columns = [d[0] for d in cursor.description]
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Message {message_id} not found")
        
        message_data = dict(zip(columns, row))
       
       # Створення вектора ознак
        features = pd.DataFrame([{col: message_data.get(col, 0) for col in FEATURE_COLUMNS}])
        features = features.fillna(0).astype(np.float32)
       
        # Autoencoder предикція
        X_ae = features.values
        if X_ae.shape[1] < self.ae_input_dim:
            X_ae = np.pad(X_ae, ((0,0), (0, self.ae_input_dim - X_ae.shape[1])))
       
        X_tensor = torch.from_numpy(X_ae).to(device=self.device,dtype=torch.float32)
       
        with torch.no_grad():
            reconstructed, _ = self.ae_model(X_tensor)
            ae_error = torch.mean((X_tensor - reconstructed) ** 2).item()
        alpha = 0.1
        mid   = self.ae_threshold
        ae_score = 1 / (1 + math.exp(-alpha*(ae_error - mid)))
        ae_anomaly = ae_score > 0.92
       
       # Isolation Forest предикція
        X_if = features[self.if_features].fillna(0).values
        X_if_scaled = self.if_scaler.transform(X_if)
        
        if_scores = [model.decision_function(X_if_scaled)[0] for model in self.if_models]
        if_avg_score = np.mean(if_scores)
        if_score = np.max(if_scores)
        if_anomaly = bool(float(if_score) > 0.1)
       
        # Комбінування
        final_score = 0.5 * ae_score + 0.5 * if_score
        is_anomaly = (ae_anomaly or if_anomaly) and (final_score >= 0.5)
        risk_level = 'CRITICAL' if final_score > 0.8 else 'HIGH' if final_score > 0.6 else 'MEDIUM' if final_score > 0.3 else 'LOW'
        confidence_level = (2 * (ae_score * if_score)) / (ae_score * if_score) if (ae_score * if_score) > 0 else 0
        print(">>> IF scores per model:", if_score)
        print(f">>> avg: {if_avg_score:.4f}, threshold: {self.if_threshold:.4f}")
        print(f">>> clipped if_score: {if_score:.4f}")
        print(f">>> clipped ae_score: {ae_error:.4f}")
        print(f">>> AE score: {ae_score:.4f}, confidence_level: {confidence_level:.4f}")
        elapsed_ms  = int((time.time() - start) * 1000)
        # Збереження результату
        result = {
           'message_id': message_id,
           'timestamp': message_data['timestamp'],
           'final_anomaly_score': final_score,
           'is_anomaly': int(is_anomaly),
           'risk_level': risk_level,
           'confidence_level': confidence_level,
           'isolation_forest_score': if_score,
           'isolation_forest_anomaly': int(if_anomaly),
           'isolation_forest_version': '1.0',
           'autoencoder_score': ae_score,
           'autoencoder_anomaly': int(ae_anomaly),
           'autoencoder_reconstruction_error': ae_error,
           'autoencoder_version': '2.0',
           'ensemble_method': 'weighted_average',
           'ensemble_weights': 'AE:0.6,IF:0.4',
           'model_version': 'AE:2.0,IF:1.0',
           'processing_time_ms': elapsed_ms,
           'feature_vector_used': json.dumps(features.values[0].tolist())
        }
       
        cursor.execute("""
           INSERT INTO anomaly_analysis (
               message_id, timestamp, final_anomaly_score, is_anomaly, risk_level,
               confidence_level, isolation_forest_score, isolation_forest_anomaly,
               isolation_forest_version, autoencoder_score, autoencoder_anomaly,
               autoencoder_reconstruction_error, autoencoder_version, ensemble_method,
               ensemble_weights, model_version, processing_time_ms, feature_vector_used
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       """, tuple(result.values()))
       
        conn.commit()
        conn.close()
       
        return result
   
    def analyze_batch(self, limit: int = 1000) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
       
        cursor.execute("""
           SELECT id FROM smpp_messages 
           WHERE id NOT IN (SELECT message_id FROM anomaly_analysis)
           LIMIT ?
       """, (limit,))
       
        message_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
       
        stats = {'total': len(message_ids), 'processed': 0, 'anomalies': 0}
       
        for msg_id in message_ids:
            try:
                result = self.analyze_message(msg_id)
                stats['processed'] += 1
                if result['is_anomaly']:
                    stats['anomalies'] += 1
            except:
                pass
        
        return stats

if __name__ == "__main__":
    pipeline = SMPPAnomalyDetectionPipeline(
       db_path="data/db/smpp.sqlite",
       autoencoder_path="models/smpp_autoencoder_20250608_073914.pt",
       isolation_forest_path="models/isolation_forest_ensemble_20250607_172058.pkl"
    )
   
    #Аналіз одного повідомлення
    #result = pipeline.analyze_message(2)
    #print(f"Message 1 - Anomaly: {result['is_anomaly']}, Score: {result['final_anomaly_score']:.3f}")
   
   # Аналіз батчу
    stats = pipeline.analyze_batch(10000)
    print(f"Processed: {stats['processed']}, Anomalies: {stats['anomalies']}")