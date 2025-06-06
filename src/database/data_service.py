"""
Data Service - Business logic layer for database interactions
Сервіс бізнес-логіки для взаємодії з БД
"""

import json
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from db_manager import DatabaseManager

class SMPPDataService:
    """Сервіс для взаємодії з SMPP даними"""
    
    def __init__(self, db_path: str = None):
        self.db = DatabaseManager(db_path) if db_path else DatabaseManager()
        self.logger = logging.getLogger('SMPPDataService')
    
    # ===== MESSAGE OPERATIONS =====
    
    def save_smpp_message(self, message_data: Dict) -> int:
        """Збереження SMPP повідомлення"""
        # Підготовка даних
        data = {
            'pdu_id': message_data.get('pdu_id'),
            'timestamp': message_data.get('timestamp', datetime.now()),
            'source_addr': message_data['source_addr'],
            'source_addr_ton': message_data.get('source_addr_ton', 0),
            'source_addr_npi': message_data.get('source_addr_npi', 0),
            'dest_addr': message_data['dest_addr'],
            'dest_addr_ton': message_data.get('dest_addr_ton', 0),
            'dest_addr_npi': message_data.get('dest_addr_npi', 0),
            'message_text': message_data.get('message_text', ''),
            'message_length': len(message_data.get('message_text', '')),
            'data_coding': message_data.get('data_coding', 0),
            'esm_class': message_data.get('esm_class', 0),
            'priority_flag': message_data.get('priority_flag', 0),
            'category': message_data.get('category')
        }
        
        message_id = self.db.insert_message(**data)
        self.logger.debug(f"Saved message {message_id}")
        return message_id
    
    def update_message_category(self, message_id: int, category: str) -> bool:
        """Оновлення категорії повідомлення"""
        rows_updated = self.db.update_message(message_id, category=category)
        return rows_updated > 0
    
    def get_messages_for_feature_extraction(self, limit: int = 1000) -> pd.DataFrame:
        """Отримання повідомлень для вилучення ознак"""
        rows = self.db.get_messages_without_analysis(limit)
        
        if not rows:
            return pd.DataFrame()
        
        # Конвертуємо в DataFrame
        data = []
        for row in rows:
            data.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'source_addr': row['source_addr'],
                'dest_addr': row['dest_addr'],
                'message_text': row['message_text'],
                'category': row['category']
            })
        
        return pd.DataFrame(data)
    
    # ===== FEATURE OPERATIONS =====
    
    def save_message_features(self, message_id: int, features: List[float], 
                             feature_names: List[str] = None) -> int:
        """Збереження ознак повідомлення"""
        # Створюємо запис в anomaly_analysis з ознаками
        feature_data = {
            'message_id': message_id,
            'timestamp': datetime.now(),
            'feature_vector': json.dumps(features)
        }
        
        # Додаємо назви ознак якщо є
        if feature_names:
            details = {'feature_names': feature_names}
            feature_data['details'] = json.dumps(details)
        
        analysis_id = self.db.insert_analysis(**feature_data)
        self.logger.debug(f"Saved features for message {message_id}")
        return analysis_id
    
    def get_messages_for_anomaly_detection(self, limit: int = 1000) -> pd.DataFrame:
        """Отримання повідомлень з ознаками для детекції"""
        rows = self.db.get_messages_with_features_no_analysis(limit)
        
        if not rows:
            return pd.DataFrame()
        
        data = []
        for row in rows:
            # Парсимо feature_vector
            features = []
            if row['feature_vector']:
                try:
                    features = json.loads(row['feature_vector'])
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid feature vector for message {row['id']}")
                    continue
            
            data.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'source_addr': row['source_addr'],
                'dest_addr': row['dest_addr'],
                'message_text': row['message_text'],
                'category': row['category'],
                'features': features
            })
        
        return pd.DataFrame(data)
    
    # ===== ANOMALY OPERATIONS =====
    
    def save_anomaly_result(self, message_id: int, result: Dict) -> Optional[int]:
        """Збереження результату детекції аномалій"""
        # Оновлюємо існуючий запис в anomaly_analysis
        update_data = {
            'anomaly_score': result['anomaly_score'],
            'is_anomaly': result['is_anomaly'],
            'risk_level': result.get('risk_level'),
            'model_version': result.get('model_version'),
            'timestamp': datetime.now()
        }
        
        # Додаємо деталі якщо є
        if 'details' in result or 'model_scores' in result:
            details = result.get('details', {})
            if 'model_scores' in result:
                details['model_scores'] = result['model_scores']
            update_data['details'] = json.dumps(details)
        
        # Знаходимо analysis_id для цього повідомлення
        analysis_rows = self.db.execute_query(
            "SELECT id FROM anomaly_analysis WHERE message_id = ?", 
            (message_id,)
        )
        
        if not analysis_rows:
            self.logger.error(f"No analysis record found for message {message_id}")
            return None
        
        analysis_id = analysis_rows[0]['id']
        rows_updated = self.db.update_analysis(analysis_id, **update_data)
        
        if rows_updated > 0:
            self.logger.debug(f"Updated anomaly result for message {message_id}")
            
            # Створюємо алерт якщо це аномалія
            if result['is_anomaly'] and result['anomaly_score'] > 0.5:
                self._create_alert(analysis_id, message_id, result)
            
            return analysis_id
        
        return None
    
    def _create_alert(self, analysis_id: int, message_id: int, result: Dict):
        """Створення алерту для аномалії"""
        # Отримуємо дані повідомлення
        message_rows = self.db.execute_query(
            "SELECT * FROM smpp_messages WHERE id = ?", 
            (message_id,)
        )
        
        if not message_rows:
            return
        
        message = message_rows[0]
        
        # Створюємо превью повідомлення
        preview = message['message_text'][:100] if message['message_text'] else ''
        if len(message['message_text'] or '') > 100:
            preview += '...'
        
        alert_data = {
            'analysis_id': analysis_id,
            'timestamp': message['timestamp'],
            'source_addr': message['source_addr'],
            'dest_addr': message['dest_addr'],
            'message_preview': preview,
            'anomaly_score': result['anomaly_score'],
            'risk_level': result.get('risk_level', 'medium'),
            'category': message['category']
        }
        
        alert_id = self.db.insert_alert(**alert_data)
        self.logger.info(f"Created alert {alert_id} for message {message_id}")
    
    # ===== DATA IMPORT/EXPORT =====
    
    def import_from_csv(self, csv_file: str) -> int:
        """Імпорт даних з CSV файлу"""
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            count = 0
            
            for _, row in df.iterrows():
                try:
                    message_data = {
                        'source_addr': str(row.get('source_addr', '')),
                        'dest_addr': str(row.get('dest_addr', '')),
                        'message_text': str(row.get('message_text', '')),
                        'timestamp': pd.to_datetime(row.get('submit_time')),
                        'category': str(row.get('category', '')) if pd.notna(row.get('category')) else None
                    }
                    
                    self.save_smpp_message(message_data)
                    count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error importing row {row.name}: {e}")
            
            self.logger.info(f"Imported {count} messages from {csv_file}")
            return count
            
        except Exception as e:
            self.logger.error(f"Error reading CSV file {csv_file}: {e}")
            return 0
    
    def export_to_csv(self, output_file: str, include_analysis: bool = True) -> bool:
        """Експорт даних в CSV"""
        try:
            if include_analysis:
                query = """
                    SELECT m.*, a.anomaly_score, a.is_anomaly, a.risk_level
                    FROM smpp_messages m
                    LEFT JOIN anomaly_analysis a ON m.id = a.message_id
                    ORDER BY m.timestamp
                """
            else:
                query = "SELECT * FROM smpp_messages ORDER BY timestamp"
            
            rows = self.db.execute_query(query)
            
            if not rows:
                self.logger.warning("No data to export")
                return False
            
            # Конвертуємо в DataFrame і зберігаємо
            data = [dict(row) for row in rows]
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"Exported {len(df)} records to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False
    
    # ===== ANALYTICS AND REPORTING =====
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Отримання даних для дашборду"""
        return {
            'statistics': self.db.get_statistics(),
            'recent_alerts': self._format_alerts(self.db.get_alerts_by_status(limit=20)),
            'recent_anomalies': self._format_anomalies(self.db.get_anomalies(limit=10)),
            'recent_activity': self._format_recent_activity(self.db.get_recent_messages(hours=24))
        }
    
    def _format_alerts(self, alert_rows) -> List[Dict]:
        """Форматування алертів"""
        alerts = []
        for row in alert_rows:
            alerts.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'source_addr': row['source_addr'],
                'dest_addr': row['dest_addr'],
                'message_preview': row['message_preview'],
                'anomaly_score': row['anomaly_score'],
                'risk_level': row['risk_level'],
                'status': row['status']
            })
        return alerts
    
    def _format_anomalies(self, anomaly_rows) -> List[Dict]:
        """Форматування аномалій"""
        anomalies = []
        for row in anomaly_rows:
            anomalies.append({
                'source_addr': row['source_addr'],
                'dest_addr': row['dest_addr'],
                'message_text': row['message_text'],
                'timestamp': row['timestamp'],
                'anomaly_score': row['anomaly_score'],
                'risk_level': row['risk_level']
            })
        return anomalies
    
    def _format_recent_activity(self, message_rows) -> pd.DataFrame:
        """Форматування останньої активності"""
        if not message_rows:
            return pd.DataFrame()
        
        data = []
        for row in message_rows:
            data.append({
                'id': row['id'],
                'source_addr': row['source_addr'],
                'dest_addr': row['dest_addr'],
                'message_text': row['message_text'],
                'timestamp': row['timestamp'],
                'anomaly_score': row['anomaly_score'],
                'is_anomaly': bool(row['is_anomaly']) if row['is_anomaly'] is not None else False
            })
        
        return pd.DataFrame(data)
    
    def get_hourly_statistics(self, hours: int = 24) -> pd.DataFrame:
        """Почасова статистика"""
        query = """
            SELECT 
                strftime('%Y-%m-%d %H:00:00', m.timestamp) as hour,
                COUNT(*) as total_messages,
                SUM(CASE WHEN a.is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies
            FROM smpp_messages m
            LEFT JOIN anomaly_analysis a ON m.id = a.message_id
            WHERE m.timestamp >= datetime('now', '-{} hours')
            GROUP BY strftime('%Y-%m-%d %H:00:00', m.timestamp)
            ORDER BY hour
        """.format(hours)
        
        rows = self.db.execute_query(query)
        
        if not rows:
            return pd.DataFrame()
        
        data = []
        for row in rows:
            data.append({
                'hour': row['hour'],
                'total_messages': row['total_messages'],
                'anomalies': row['anomalies'] or 0
            })
        
        return pd.DataFrame(data)
    
    # ===== ALERT MANAGEMENT =====
    
    def update_alert_status(self, alert_id: int, status: str, 
                           handled_by: str = None, notes: str = None) -> bool:
        """Оновлення статусу алерту"""
        update_data = {
            'status': status,
            'handled_at': datetime.now()
        }
        
        if handled_by:
            update_data['handled_by'] = handled_by
        if notes:
            update_data['notes'] = notes
        
        query = """
            UPDATE alerts 
            SET status = ?, handled_at = ?, handled_by = ?, notes = ?
            WHERE id = ?
        """
        
        rows_updated = self.db.execute_update(
            query, 
            (status, update_data['handled_at'], handled_by, notes, alert_id)
        )
        
        return rows_updated > 0
    
    def get_alerts_by_status(self, status: str = None, limit: int = 100) -> List[Dict]:
        """Отримання алертів по статусу"""
        alert_rows = self.db.get_alerts_by_status(status, limit)
        return self._format_alerts(alert_rows)
    
    # ===== TRAINING DATA =====
    
    def get_training_data(self, include_features: bool = True) -> pd.DataFrame:
        """Отримання даних для навчання моделей"""
        if include_features:
            query = """
                SELECT m.*, a.feature_vector, a.anomaly_score, a.is_anomaly
                FROM smpp_messages m
                JOIN anomaly_analysis a ON m.id = a.message_id
                WHERE a.feature_vector IS NOT NULL
                ORDER BY m.timestamp
            """
        else:
            query = """
                SELECT m.*, a.anomaly_score, a.is_anomaly
                FROM smpp_messages m
                LEFT JOIN anomaly_analysis a ON m.id = a.message_id
                ORDER BY m.timestamp
            """
        
        rows = self.db.execute_query(query)
        
        if not rows:
            return pd.DataFrame()
        
        data = []
        for row in rows:
            row_data = dict(row)
            
            # Парсимо feature_vector якщо є
            if include_features and row_data.get('feature_vector'):
                try:
                    row_data['features'] = json.loads(row_data['feature_vector'])
                except json.JSONDecodeError:
                    row_data['features'] = []
            
            data.append(row_data)
        
        return pd.DataFrame(data)
    
    # ===== MAINTENANCE =====
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Очищення старих даних"""
        deleted_messages = self.db.cleanup_old_data(days_to_keep)
        
        # Очищення старої статистики
        stats_query = """
            DELETE FROM system_stats 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days_to_keep)
        
        deleted_stats = self.db.execute_delete(stats_query)
        
        return {
            'deleted_messages': deleted_messages,
            'deleted_stats': deleted_stats
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Інформація про систему"""
        return {
            'database_size': self.db.get_database_size(),
            'statistics': self.db.get_statistics(),
            'db_path': self.db.db_path
        }
    
    def optimize_database(self):
        """Оптимізація бази даних"""
        self.db.vacuum_database()
    
    # ===== UTILITY METHODS =====
    
    def test_connection(self) -> bool:
        """Тестування з'єднання з БД"""
        try:
            result = self.db.execute_query("SELECT 1")
            return len(result) > 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_table_counts(self) -> Dict[str, int]:
        """Кількість записів в таблицях"""
        tables = ['smpp_messages', 'anomaly_analysis', 'alerts', 'captured_pdus', 'system_stats', 'models']
        counts = {}
        
        for table in tables:
            try:
                result = self.db.execute_query(f"SELECT COUNT(*) FROM {table}")
                counts[table] = result[0][0] if result else 0
            except Exception as e:
                self.logger.error(f"Error counting {table}: {e}")
                counts[table] = 0
        
        return counts