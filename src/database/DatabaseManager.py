"""
Database Manager - SQLite Storage
Модуль для роботи з SQLite базою даних
"""

import sqlite3
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import threading

# Створення папки для БД
os.makedirs('data/db', exist_ok=True)

class DatabaseManager:
    """Менеджер бази даних для зберігання SMPP даних"""
    
    def __init__(self, db_path: str = 'data/db/smpp_anomaly.db'):
        """
        Ініціалізація менеджера БД
        
        Args:
            db_path: шлях до файлу бази даних
        """
        self.db_path = db_path
        self.logger = logging.getLogger('DatabaseManager')
        
        # Thread-local storage для з'єднань
        self._local = threading.local()
        
        # Ініціалізація БД
        self._init_database()
        
        self.logger.info(f"Database initialized at {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager для отримання з'єднання з БД"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
            # Включення foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        else:
            self._local.connection.commit()
    
    def _init_database(self):
        """Ініціалізація структури бази даних"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблиця захоплених PDU
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS captured_pdus (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    src_ip TEXT,
                    dst_ip TEXT,
                    src_port INTEGER,
                    dst_port INTEGER,
                    command_id INTEGER NOT NULL,
                    command_name TEXT NOT NULL,
                    command_status INTEGER,
                    sequence_number INTEGER,
                    command_length INTEGER,
                    raw_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблиця SMPP повідомлень
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS smpp_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdu_id INTEGER,
                    timestamp DATETIME NOT NULL,
                    source_addr TEXT NOT NULL,
                    source_addr_ton INTEGER,
                    source_addr_npi INTEGER,
                    dest_addr TEXT NOT NULL,
                    dest_addr_ton INTEGER,
                    dest_addr_npi INTEGER,
                    message_text TEXT,
                    message_length INTEGER,
                    data_coding INTEGER,
                    esm_class INTEGER,
                    priority_flag INTEGER,
                    category TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pdu_id) REFERENCES captured_pdus(id)
                )
            """)
            
            # Таблиця результатів аналізу
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    anomaly_score REAL NOT NULL,
                    is_anomaly BOOLEAN NOT NULL,
                    risk_level TEXT,
                    model_version TEXT,
                    feature_vector TEXT,  -- JSON
                    details TEXT,  -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (message_id) REFERENCES smpp_messages(id)
                )
            """)
            
            # Таблиця алертів
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    source_addr TEXT NOT NULL,
                    dest_addr TEXT NOT NULL,
                    message_preview TEXT,
                    anomaly_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    category TEXT,
                    status TEXT DEFAULT 'new',  -- new, acknowledged, resolved, false_positive
                    handled_by TEXT,
                    handled_at DATETIME,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES anomaly_analysis(id)
                )
            """)
            
            # Таблиця системної статистики
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    component TEXT NOT NULL,  -- capture, detection, system
                    stats_data TEXT NOT NULL,  -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблиця моделей
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,  -- isolation_forest, autoencoder
                    version TEXT NOT NULL,
                    file_path TEXT,
                    performance_metrics TEXT,  -- JSON
                    is_active BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Індекси для оптимізації
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pdus_timestamp ON captured_pdus(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON smpp_messages(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_source ON smpp_messages(source_addr)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_dest ON smpp_messages(dest_addr)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_anomaly ON anomaly_analysis(is_anomaly)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts(risk_level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
            
            conn.commit()
            self.logger.info("Database schema created/verified")
    
    def save_captured_pdu(self, pdu: Dict) -> int:
        """
        Збереження захопленого PDU
        
        Args:
            pdu: словник з даними PDU
            
        Returns:
            ID збереженого запису
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO captured_pdus (
                    timestamp, src_ip, dst_ip, src_port, dst_port,
                    command_id, command_name, command_status, 
                    sequence_number, command_length, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pdu.get('timestamp', datetime.now()),
                pdu.get('src_ip'),
                pdu.get('dst_ip'),
                pdu.get('src_port'),
                pdu.get('dst_port'),
                pdu['header']['command_id'],
                pdu['header']['command_id_name'],
                pdu['header'].get('command_status', 0),
                pdu['header'].get('sequence_number', 0),
                pdu['header'].get('command_length', 0),
                pdu.get('raw_data', '')
            ))
            
            return cursor.lastrowid
    
    def save_smpp_message(self, pdu_id: int, pdu_data: Dict) -> int:
        """
        Збереження SMPP повідомлення
        
        Args:
            pdu_id: ID захопленого PDU
            pdu_data: дані PDU з розпарсеним body
            
        Returns:
            ID збереженого повідомлення
        """
        if 'body' not in pdu_data:
            return None
        
        body = pdu_data['body']
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO smpp_messages (
                    pdu_id, timestamp, source_addr, source_addr_ton, source_addr_npi,
                    dest_addr, dest_addr_ton, dest_addr_npi, message_text,
                    message_length, data_coding, esm_class, priority_flag, category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pdu_id,
                pdu_data.get('timestamp', datetime.now()),
                body.get('source_addr', ''),
                body.get('source_addr_ton', 0),
                body.get('source_addr_npi', 0),
                body.get('dest_addr', ''),
                body.get('dest_addr_ton', 0),
                body.get('dest_addr_npi', 0),
                body.get('message_text', ''),
                body.get('message_length', 0),
                body.get('data_coding', 0),
                body.get('esm_class', 0),
                body.get('priority_flag', 0),
                pdu_data.get('category', 'unknown')
            ))
            
            return cursor.lastrowid
    
    def save_anomaly_analysis(self, message_id: int, analysis_result: Dict) -> int:
        """
        Збереження результатів аналізу аномалій
        
        Args:
            message_id: ID повідомлення
            analysis_result: результати аналізу
            
        Returns:
            ID збереженого аналізу
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO anomaly_analysis (
                    message_id, timestamp, anomaly_score, is_anomaly,
                    risk_level, model_version, feature_vector, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                datetime.now(),
                analysis_result['anomaly_score'],
                analysis_result['is_anomaly'],
                analysis_result.get('risk_level', 'LOW'),
                analysis_result.get('model_version', 'unknown'),
                json.dumps(analysis_result.get('feature_vector', [])),
                json.dumps(analysis_result.get('details', {}))
            ))
            
            return cursor.lastrowid
    
    def save_alert(self, alert_data: Dict) -> int:
        """
        Збереження алерту
        
        Args:
            alert_data: дані алерту
            
        Returns:
            ID збереженого алерту
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (
                    analysis_id, timestamp, source_addr, dest_addr,
                    message_preview, anomaly_score, risk_level, category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_data['analysis_id'],
                alert_data['timestamp'],
                alert_data['source_addr'],
                alert_data['dest_addr'],
                alert_data['message_preview'],
                alert_data['anomaly_score'],
                alert_data['risk_level'],
                alert_data.get('category', 'unknown')
            ))
            
            return cursor.lastrowid
    
    def save_system_stats(self, component: str, stats: Dict):
        """
        Збереження системної статистики
        
        Args:
            component: назва компонента
            stats: статистичні дані
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_stats (timestamp, component, stats_data)
                VALUES (?, ?, ?)
            """, (
                datetime.now(),
                component,
                json.dumps(stats, default=str)
            ))
    
    def get_recent_messages(self, limit: int = 100) -> List[Dict]:
        """Отримання останніх повідомлень"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT m.*, a.anomaly_score, a.is_anomaly, a.risk_level
                FROM smpp_messages m
                LEFT JOIN anomaly_analysis a ON m.id = a.message_id
                ORDER BY m.timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_alerts(self, status: str = None, limit: int = 100) -> List[Dict]:
        """Отримання алертів"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM alerts
                    WHERE status = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM alerts
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self, start_time: datetime = None, end_time: datetime = None) -> Dict:
        """Отримання статистики"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Базова статистика
            cursor.execute("SELECT COUNT(*) as total FROM captured_pdus")
            total_pdus = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as total FROM smpp_messages")
            total_messages = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as total FROM anomaly_analysis WHERE is_anomaly = 1")
            total_anomalies = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as total FROM alerts")
            total_alerts = cursor.fetchone()['total']
            
            # Статистика по ризикам
            cursor.execute("""
                SELECT risk_level, COUNT(*) as count
                FROM alerts
                GROUP BY risk_level
            """)
            risk_distribution = {row['risk_level']: row['count'] for row in cursor.fetchall()}
            
            # Топ джерела аномалій
            cursor.execute("""
                SELECT source_addr, COUNT(*) as count
                FROM alerts
                GROUP BY source_addr
                ORDER BY count DESC
                LIMIT 10
            """)
            top_sources = [dict(row) for row in cursor.fetchall()]
            
            return {
                'total_pdus': total_pdus,
                'total_messages': total_messages,
                'total_anomalies': total_anomalies,
                'total_alerts': total_alerts,
                'risk_distribution': risk_distribution,
                'top_anomaly_sources': top_sources
            }
    
    def update_alert_status(self, alert_id: int, status: str, handled_by: str = None, notes: str = None):
        """Оновлення статусу алерту"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE alerts
                SET status = ?, handled_by = ?, handled_at = ?, notes = ?
                WHERE id = ?
            """, (status, handled_by, datetime.now() if handled_by else None, notes, alert_id))
    
    def save_model_info(self, model_info: Dict) -> int:
        """Збереження інформації про модель"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Деактивуємо попередні моделі цього типу
            if model_info.get('is_active', False):
                cursor.execute("""
                    UPDATE models SET is_active = 0
                    WHERE model_type = ?
                """, (model_info['model_type'],))
            
            cursor.execute("""
                INSERT INTO models (
                    model_name, model_type, version, file_path,
                    performance_metrics, is_active
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_info['model_name'],
                model_info['model_type'],
                model_info['version'],
                model_info.get('file_path'),
                json.dumps(model_info.get('performance_metrics', {})),
                model_info.get('is_active', False)
            ))
            
            return cursor.lastrowid
    
    def close(self):
        """Закриття з'єднання з БД"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            self.logger.info("Database connection closed")


# Singleton instance
_db_instance = None

def get_db() -> DatabaseManager:
    """Отримання екземпляру менеджера БД"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance


# Приклад використання
if __name__ == "__main__":
    # Налаштування логування
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    
    # Створення менеджера БД
    db = get_db()
    
    # Тестові дані
    test_pdu = {
        'timestamp': datetime.now(),
        'src_ip': '192.168.1.100',
        'src_port': 54321,
        'dst_ip': '192.168.1.200',
        'dst_port': 2775,
        'header': {
            'command_id': 0x00000004,
            'command_id_name': 'submit_sm',
            'command_status': 0,
            'sequence_number': 12345,
            'command_length': 150
        },
        'body': {
            'source_addr': 'TEST',
            'dest_addr': '380501234567',
            'message_text': 'Test message'
        },
        'raw_data': 'AABBCCDD...'
    }
    
    # Збереження PDU
    pdu_id = db.save_captured_pdu(test_pdu)
    print(f"Saved PDU with ID: {pdu_id}")
    
    # Збереження повідомлення
    msg_id = db.save_smpp_message(pdu_id, test_pdu)
    print(f"Saved message with ID: {msg_id}")
    
    # Збереження аналізу
    analysis = {
        'anomaly_score': 0.85,
        'is_anomaly': True,
        'risk_level': 'HIGH',
        'model_version': 'v1.0'
    }
    analysis_id = db.save_anomaly_analysis(msg_id, analysis)
    print(f"Saved analysis with ID: {analysis_id}")
    
    # Отримання статистики
    stats = db.get_statistics()
    print(f"Statistics: {stats}")