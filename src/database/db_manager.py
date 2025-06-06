"""
Database Manager - Pure SQL operations for SMPP system
Чистий SQL менеджер без бізнес-логіки
"""

import sqlite3
import logging
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

class DatabaseManager:
    """Чистий SQL менеджер бази даних"""
    
    def __init__(self, db_path: str = 'data/db/smpp_anomaly.db'):
        self.db_path = db_path
        self.logger = logging.getLogger('DatabaseManager')
        
        # Створення папки
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Ініціалізація
        self._init_database()
        self.logger.info(f"Database initialized: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Thread-safe connection manager"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        else:
            conn.commit()
        finally:
            conn.close()
    
    def _init_database(self):
        """Ініціалізація БД з SQL-файлу"""
        schema_path = Path("src/database/schema/init_schema.sql")
        with open(schema_path, encoding="utf-8") as f:
            sql_script = f.read()

        with self.get_connection() as conn:
            conn.executescript(sql_script)

    # ===== RAW SQL OPERATIONS =====
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Виконання SELECT запиту"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: Tuple = ()) -> int:
        """Виконання INSERT запиту"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.lastrowid
    
    def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Виконання UPDATE запиту"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount
    
    def execute_delete(self, query: str, params: Tuple = ()) -> int:
        """Виконання DELETE запиту"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Виконання множинних операцій"""
        with self.get_connection() as conn:
            cursor = conn.executemany(query, params_list)
            return cursor.rowcount
    
    # ===== BASIC CRUD OPERATIONS =====
    
    def insert_message(self, **kwargs) -> int:
        """Вставка повідомлення"""
        fields = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?' for _ in kwargs])
        query = f"INSERT INTO smpp_messages ({fields}) VALUES ({placeholders})"
        return self.execute_insert(query, tuple(kwargs.values()))
    
    def update_message(self, message_id: int, **kwargs) -> int:
        """Оновлення повідомлення"""
        set_clause = ', '.join([f"{k} = ?" for k in kwargs.keys()])
        query = f"UPDATE smpp_messages SET {set_clause} WHERE id = ?"
        params = list(kwargs.values()) + [message_id]
        return self.execute_update(query, tuple(params))
    
    def insert_analysis(self, **kwargs) -> int:
        """Вставка результату аналізу"""
        fields = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?' for _ in kwargs])
        query = f"INSERT INTO anomaly_analysis ({fields}) VALUES ({placeholders})"
        return self.execute_insert(query, tuple(kwargs.values()))
    
    def update_analysis(self, analysis_id: int, **kwargs) -> int:
        """Оновлення результату аналізу"""
        set_clause = ', '.join([f"{k} = ?" for k in kwargs.keys()])
        query = f"UPDATE anomaly_analysis SET {set_clause} WHERE id = ?"
        params = list(kwargs.values()) + [analysis_id]
        return self.execute_update(query, tuple(params))
    
    def insert_alert(self, **kwargs) -> int:
        """Вставка алерту"""
        fields = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?' for _ in kwargs])
        query = f"INSERT INTO alerts ({fields}) VALUES ({placeholders})"
        return self.execute_insert(query, tuple(kwargs.values()))
    
    # ===== COMMON QUERIES =====
    
    def get_messages_without_analysis(self, limit: int = 1000) -> List[sqlite3.Row]:
        """Повідомлення без аналізу"""
        query = """
            SELECT m.id, m.timestamp, m.source_addr, m.dest_addr, 
                   m.message_text, m.category
            FROM smpp_messages m
            LEFT JOIN anomaly_analysis a ON m.id = a.message_id
            WHERE a.message_id IS NULL
            ORDER BY m.timestamp
            LIMIT ?
        """
        return self.execute_query(query, (limit,))
    
    def get_messages_with_features_no_analysis(self, limit: int = 1000) -> List[sqlite3.Row]:
        """Повідомлення з ознаками без аналізу"""
        query = """
            SELECT m.id, m.timestamp, m.source_addr, m.dest_addr, 
                   m.message_text, m.category, a.feature_vector
            FROM smpp_messages m
            JOIN anomaly_analysis a ON m.id = a.message_id
            WHERE a.feature_vector IS NOT NULL 
              AND a.anomaly_score IS NULL
            ORDER BY m.timestamp
            LIMIT ?
        """
        return self.execute_query(query, (limit,))
    
    def get_anomalies(self, limit: int = 100, min_score: float = 0.5) -> List[sqlite3.Row]:
        """Отримання аномалій"""
        query = """
            SELECT m.source_addr, m.dest_addr, m.message_text, m.timestamp,
                   a.anomaly_score, a.risk_level, a.model_version
            FROM smpp_messages m
            JOIN anomaly_analysis a ON m.id = a.message_id
            WHERE a.is_anomaly = 1 AND a.anomaly_score >= ?
            ORDER BY a.anomaly_score DESC, m.timestamp DESC
            LIMIT ?
        """
        return self.execute_query(query, (min_score, limit))
    
    def get_alerts_by_status(self, status: str = None, limit: int = 100) -> List[sqlite3.Row]:
        """Отримання алертів по статусу"""
        if status:
            query = """
                SELECT * FROM alerts 
                WHERE status = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            return self.execute_query(query, (status, limit))
        else:
            query = """
                SELECT * FROM alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            return self.execute_query(query, (limit,))
    
    def get_recent_messages(self, hours: int = 24, limit: int = 1000) -> List[sqlite3.Row]:
        """Останні повідомлення"""
        query = """
            SELECT m.id, m.source_addr, m.dest_addr, m.message_text, m.timestamp,
                   a.anomaly_score, a.is_anomaly
            FROM smpp_messages m
            LEFT JOIN anomaly_analysis a ON m.id = a.message_id
            WHERE m.timestamp >= datetime('now', '-{} hours')
            ORDER BY m.timestamp DESC
            LIMIT ?
        """.format(hours)
        return self.execute_query(query, (limit,))
    
    def get_statistics(self) -> Dict[str, int]:
        """Базова статистика"""
        queries = {
            'total_messages': "SELECT COUNT(*) FROM smpp_messages",
            'with_analysis': "SELECT COUNT(*) FROM anomaly_analysis WHERE feature_vector IS NOT NULL",
            'analyzed': "SELECT COUNT(*) FROM anomaly_analysis WHERE anomaly_score IS NOT NULL",
            'anomalies': "SELECT COUNT(*) FROM anomaly_analysis WHERE is_anomaly = 1",
            'alerts_new': "SELECT COUNT(*) FROM alerts WHERE status = 'new'",
            'alerts_total': "SELECT COUNT(*) FROM alerts"
        }
        
        stats = {}
        for key, query in queries.items():
            result = self.execute_query(query)
            stats[key] = result[0][0] if result else 0
        
        # Додаємо відсоток аномалій
        analyzed = stats.get('analyzed', 0)
        anomalies = stats.get('anomalies', 0)
        stats['anomaly_rate'] = (anomalies / max(analyzed, 1)) * 100
        
        return stats
    
    # ===== MAINTENANCE OPERATIONS =====
    
    def vacuum_database(self):
        """Оптимізація БД"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
        self.logger.info("Database optimized")
    
    def get_database_size(self) -> Dict[str, Any]:
        """Розмір БД"""
        file_size = os.path.getsize(self.db_path)
        
        # Розміри таблиць
        table_sizes = {}
        tables = ['smpp_messages', 'anomaly_analysis', 'alerts', 'captured_pdus']
        
        for table in tables:
            result = self.execute_query(f"SELECT COUNT(*) FROM {table}")
            table_sizes[table] = result[0][0] if result else 0
        
        return {
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'table_counts': table_sizes
        }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Очищення старих даних"""
        query = """
            DELETE FROM smpp_messages 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days_to_keep)
        
        deleted_count = self.execute_delete(query)
        self.logger.info(f"Deleted {deleted_count} old messages")
        
        if deleted_count > 0:
            self.vacuum_database()
        
        return deleted_count