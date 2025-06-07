import pandas as pd
import numpy as np
import re
import os
import logging
import json
import sqlite3
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
from classification.classification import TextSelector, TextFeatureExtractor, SourceFeatureExtractor, TimeFeatureExtractor

# Додаємо шляхи до модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obfuscation.obfuscation_detector import ObfuscationDetector

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

class SMPPFeatureExtractor:
    """Екстрактор ознак для SMPP повідомлень з підтримкою БД та CSV"""
    
    def __init__(self, config_path: str = "config/feature_config.json", db_path: str = None, csv_path: Optional[str] = None):
        """
        Ініціалізація екстрактора
        
        Args:
            config_path: Шлях до конфігураційного файлу
            db_path: Шлях до бази даних
            csv_path: Шлях до CSV файлу (опційно)
        """
        self.logger = logging.getLogger(__name__)
        # Завантаження конфігурації
        self.config = self._load_config(config_path)
        
        # Параметри з конфігурації - використовуємо абсолютні шляхи
        if db_path:
            self.db_path = os.path.abspath(db_path)
        else:
            config_db_path = self.config.get('db_path', 'data/db/smpp.sqlite')
            self.db_path = os.path.abspath(config_db_path)
        
        model_path = self.config.get('model_path', 'models/sms_classifier_20250607_0605.pkl')
        self.model_path = os.path.abspath(model_path)
        self.batch_size = self.config.get('batch_size', 1000)
        
        # Шлях до CSV файлу, якщо задано
        self.csv_path = os.path.abspath(csv_path) if csv_path else None
        
        # Діагностика шляхів
        self.logger.info(f"Робочий каталог: {os.getcwd()}")
        self.logger.info(f"Шлях до БД: {self.db_path}")
        self.logger.info(f"БД існує: {os.path.exists(self.db_path)}")
        self.logger.info(f"Шлях до моделі: {self.model_path}")
        self.logger.info(f"Модель існує: {os.path.exists(self.model_path)}")
        self.logger.info(f"Шлях до CSV: {self.csv_path}")
        
        # Завантаження компонентів
        self.classifier_model = None
        self.label_encoder = None
        self.classifier_config = {}
        self._load_classifier()
        
        self.obfuscation_detector = ObfuscationDetector()
        
        # Лічильники для частотного аналізу
        self.sender_counter = defaultdict(list)
        self.recipient_counter = defaultdict(list)
        
        # Версія екстрактора
        self.version = "1.0.0"
    
    def _load_config(self, config_path: str) -> dict:
        """Завантаження конфігурації"""
        if not os.path.exists(config_path):
            # Створюємо базову конфігурацію
            config = {
                "db_path": "data/db/smpp.sqlite",
                "model_path": "models/sms_classifier_20250607_0605.pkl",
                "batch_size": 1000,
                "shorteners": ["bit.ly", "tinyurl.com", "clck.ru", "goo.gl"],
                "source_addresses": {
                    "banking": ["PRIVAT24", "MONOBANK", "OSCHADBANK", "PUMB"],
                    "delivery": ["NOVAPOSHTA", "UKRPOSHTA", "JUSTIN", "MEEST"],
                    "government": ["DIIA", "GOV.UA", "DPS", "MVS"],
                    "medical": ["HELSI", "DOC.UA", "MEDICS"],
                    "retail": ["ROZETKA", "EPICENTR", "ATB", "SILPO"]
                },
                "suspicious_words": {
                    "uk": ["терміново", "заблоковано", "виграли", "безкоштовно", "увага"],
                    "ru": ["срочно", "заблокирован", "выиграли", "бесплатно", "внимание"],
                    "en": ["urgent", "blocked", "won", "free", "attention"]
                }
            }
            
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Створено конфігураційний файл: {config_path}")
            return config
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_data(self, limit: Optional[int] = None, where_clause: Optional[str] = None) -> pd.DataFrame:
        """Завантаження даних з БД або CSV залежно від параметра"""
        if self.csv_path:
            return self.load_data_from_csv()
        return self.load_data_from_db(limit, where_clause)
    
    def load_data_from_csv(self) -> pd.DataFrame:
        """Завантаження даних з CSV файлу"""
        self.logger.info(f"Завантаження даних з CSV файлу: {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path)
            self.logger.info(f"Завантажено {len(df)} записів з CSV файлу")
            return df
        except Exception as e:
            self.logger.error(f"Помилка завантаження CSV файлу: {e}")
            return pd.DataFrame()
    
    def _load_classifier(self):
        """Завантаження класифікатора категорій"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Файл моделі не знайдено: {self.model_path}")
                return
            
            # Додаємо всі необхідні класи в globals для pickle
            globals()['TextSelector'] = TextSelector
            globals()['TextFeatureExtractor'] = TextFeatureExtractor
            globals()['SourceFeatureExtractor'] = SourceFeatureExtractor
            globals()['TimeFeatureExtractor'] = TimeFeatureExtractor
                
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Структура моделі згідно з вашим форматом
            if isinstance(model_data, dict):
                self.classifier_model = model_data.get('best_model')
                self.label_encoder = model_data.get('label_encoder')
                self.classifier_config = model_data.get('source_addresses', {})
                
                if self.classifier_model is None:
                    raise ValueError("Модель 'best_model' не знайдена в файлі")
                    
                self.logger.info(f"Класифікатор завантажено з {self.model_path}")
                if self.label_encoder:
                    self.logger.info(f"Класи: {list(self.label_encoder.classes_)}")
                else:
                    self.logger.warning("Label encoder не знайдено")
            else:
                # Якщо модель збережена у старому форматі
                self.classifier_model = model_data
                self.label_encoder = None
                self.classifier_config = {}
                self.logger.warning("Завантажено модель у старому форматі без label_encoder")
        
        except Exception as e:
            self.logger.error(f"Помилка завантаження класифікатора: {e}")
            self.logger.info("Буде використано rule-based класифікацію")
            self.classifier_model = None
            self.label_encoder = None
            self.classifier_config = {}
    
    def load_data_from_db(self, limit: Optional[int] = None, 
                         where_clause: Optional[str] = None) -> pd.DataFrame:
        """
        Завантаження даних з БД
        
        Args:
            limit: Обмеження кількості записів
            where_clause: Додаткова умова WHERE або повний LIMIT/OFFSET
        
        Returns:
            DataFrame з даними
        """
        self.logger.info(f"Спроба підключення до БД: {self.db_path}")
        
        if not os.path.exists(self.db_path):
            self.logger.error(f"База даних не знайдена: {self.db_path}")
            # Спробуємо знайти БД в поточному каталозі
            alternative_paths = [
                "smpp.sqlite",
                "data/smpp.sqlite", 
                "db/smpp.sqlite",
                "../data/db/smpp.sqlite",
                "./data/db/smpp.sqlite"
            ]
            
            for alt_path in alternative_paths:
                abs_alt_path = os.path.abspath(alt_path)
                self.logger.info(f"Перевіряємо альтернативний шлях: {abs_alt_path}")
                if os.path.exists(abs_alt_path):
                    self.logger.info(f"Знайдено БД за альтернативним шляхом: {abs_alt_path}")
                    self.db_path = abs_alt_path
                    break
            else:
                self.logger.error("БД не знайдено за жодним з альтернативних шляхів")
                return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Перевіряємо існування таблиці
            table_check = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='smpp_messages'", 
                conn
            )
            
            if table_check.empty:
                self.logger.error("Таблиця 'smpp_messages' не існує в БД")
                # Показуємо які таблиці є
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table'", 
                    conn
                )
                self.logger.info(f"Доступні таблиці: {tables['name'].tolist()}")
                conn.close()
                return pd.DataFrame()
            
            query = """
                SELECT id, pdu_id, timestamp, source_addr, source_addr_ton, 
                       source_addr_npi, dest_addr, dest_addr_ton, dest_addr_npi,
                       message_text, message_length, data_coding, esm_class, 
                       priority_flag, category, message_parts, hour, day_of_week,
                       features_extracted, feature_extraction_version
                FROM smpp_messages
            """
            
            # Обробляємо where_clause
            if where_clause:
                # Якщо where_clause містить LIMIT/OFFSET, додаємо як є
                if "LIMIT" in where_clause.upper():
                    query += f" WHERE {where_clause}"
                else:
                    # Інакше це звичайна WHERE умова
                    query += f" WHERE {where_clause}"
                    if limit:
                        query += f" LIMIT {limit}"
            elif limit:
                query += f" LIMIT {limit}"
            
            self.logger.info(f"Виконуємо запит: {query[:100]}...")
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Конвертація типів
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.logger.info(f"Завантажено {len(df)} записів з БД")
            return df
            
        except Exception as e:
            self.logger.error(f"Помилка завантаження з БД: {e}")
            if 'conn' in locals():
                conn.close()
            return pd.DataFrame()
    
    def classify_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Класифікація повідомлень за категоріями
        
        Args:
            df: DataFrame з повідомленнями
            
        Returns:
            DataFrame з доданою колонкою predicted_category
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        if self.classifier_model is None:
            self.logger.warning("Класифікатор не завантажено, використовуємо базові правила")
            df['predicted_category'] = df.apply(
                lambda row: self._rule_based_category(row['message_text'], row['source_addr']), 
                axis=1
            )
        else:
            try:
                # Підготовка даних для класифікатора
                features_for_classifier = []
                
                for _, row in df.iterrows():
                    # Створюємо ознаки які очікує модель
                    feature_dict = {
                        'message_text': str(row['message_text']),
                        'source_addr': str(row['source_addr']),
                        'submit_time': row['timestamp'],
                        'processed_text': self._preprocess_text(str(row['message_text']))
                    }
                    features_for_classifier.append(feature_dict)
                
                # Створюємо DataFrame для класифікатора
                classifier_input = pd.DataFrame(features_for_classifier)
                
                # Передбачення
                predictions = self.classifier_model.predict(classifier_input)
                
                # Якщо є label_encoder, декодуємо
                if self.label_encoder:
                    df['predicted_category'] = self.label_encoder.inverse_transform(predictions)
                else:
                    df['predicted_category'] = predictions
                    
                self.logger.info(f"Класифіковано {len(df)} повідомлень")
                    
            except Exception as e:
                self.logger.error(f"Помилка класифікації: {e}")
                # Fallback до правил
                df['predicted_category'] = df.apply(
                    lambda row: self._rule_based_category(row['message_text'], row['source_addr']), 
                    axis=1
                )
        
        return df
    
    def _preprocess_text(self, text: str) -> str:
        """Попередня обробка тексту для класифікатора"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Нормалізація
        text = re.sub(r'\b\d{4,}\b', 'LONGNUM', text)
        text = re.sub(r'\d+\s*(?:грн|₴|uah)', 'MONEY', text)
        text = re.sub(r'\+?3?8?0?\d{9,}', 'PHONE', text)
        text = re.sub(r'https?://[^\s]+', 'URL', text)
        
        return text
    
    def _rule_based_category(self, text: str, source_addr: str) -> str:
        """Базова класифікація на основі правил"""
        text_lower = str(text).lower()
        source_upper = str(source_addr).upper()
        
        # Перевірка джерела
        for category, sources in self.source_addresses.items():
            for source in sources:
                if source.upper() in source_upper:
                    return category
        
        # Перевірка ключових слів
        if any(word in text_lower for word in ['баланс', 'карт', 'платіж', 'рахунок']):
            return 'banking'
        elif any(word in text_lower for word in ['посилка', 'відділення', 'доставка']):
            return 'delivery'
        elif any(word in text_lower for word in ['код', 'підтвердження', 'otp']):
            return 'authentication'
        elif any(word in text_lower for word in ['запис', 'лікар', 'прийом']):
            return 'medical'
        
        return 'other'
    
    def calculate_obfuscation_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахунок обфускації за допомогою ObfuscationDetector
        
        Args:
            df: DataFrame з повідомленнями
            
        Returns:
            DataFrame з доданою колонкою obfuscation_score
        """
        if df.empty:
            return df
            
        df = df.copy()
        self.logger.info("Розрахунок обфускації...")
        
        obfuscation_scores = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                self.logger.debug(f"Оброблено {idx}/{len(df)} записів")
            
            try:
                # Аналіз тексту та відправника
                result = self.obfuscation_detector.analyze_text_and_sender(
                    str(row['message_text']), 
                    str(row['source_addr'])
                )
                
                obfuscation_scores.append(result.get('combined_obfuscation', 0.0))
            except Exception as e:
                self.logger.warning(f"Помилка обфускації для запису {idx}: {e}")
                obfuscation_scores.append(0.0)
        
        df['obfuscation_score'] = obfuscation_scores
        
        if obfuscation_scores:
            avg_score = np.mean(obfuscation_scores)
            self.logger.info(f"Обфускація розрахована. Середній бал: {avg_score:.3f}")
        
        return df
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Екстракція всіх ознак для записів
        
        Args:
            df: DataFrame з базовими даними
            
        Returns:
            DataFrame з усіма ознаками
        """
        if df.empty:
            return df
            
        self.logger.info("Екстракція всіх ознак...")
        
        # Очищення даних
        df = self._clean_data(df)
        
        # Підготовка частотного аналізу
        self._calculate_frequency_data(df)
        
        # 1. Класифікація категорій
        if 'category' not in df.columns or df['category'].isna().all():
            df = self.classify_messages(df)
            df['category'] = df['predicted_category']
        
        # 2. Обфускація
        df = self.calculate_obfuscation_scores(df)
        
        # 3. Протокольні аномалії
        df['message_length'] = df['message_text'].astype(str).apply(len)
        df['source_addr_length'] = df['source_addr'].astype(str).apply(len)
        df['source_is_numeric'] = df['source_addr'].apply(lambda x: 1 if str(x).isdigit() else 0)
        df['dest_is_valid'] = df['dest_addr'].apply(self._validate_phone_number)
        df['message_parts'] = df['message_length'].apply(lambda x: max(1, (x + 139) // 140))
        df['encoding_issues'] = df['message_text'].apply(self._detect_encoding_issues)
        df['empty_message'] = (df['message_length'] == 0).astype(int)
        df['excessive_length'] = (df['message_length'] > 1000).astype(int)
        
        # 4. Частотні аномалії
        df['sender_frequency'] = df['source_addr'].map(df['source_addr'].value_counts())
        df['recipient_frequency'] = df['dest_addr'].map(df['dest_addr'].value_counts())
        df['sender_burst'] = df.apply(lambda x: self._calculate_sender_burst(x['source_addr'], x['timestamp']), axis=1)
        df['recipient_burst'] = df.apply(lambda x: self._calculate_recipient_burst(x['dest_addr'], x['timestamp']), axis=1)
        df['high_sender_frequency'] = (df['sender_frequency'] > 30).astype(int)
        df['high_recipient_frequency'] = (df['recipient_frequency'] > 15).astype(int)
        
        # 5. Семантичні аномалії
        df['suspicious_word_count'] = df['message_text'].apply(self._count_suspicious_words)
        df['url_count'] = df['message_text'].apply(self._count_urls)
        df['suspicious_url'] = df['message_text'].apply(self._has_suspicious_url)
        df['urgency_score'] = df['message_text'].apply(self._calculate_urgency_score)
        df['financial_patterns'] = df['message_text'].apply(self._count_financial_patterns)
        df['phone_numbers'] = df['message_text'].apply(self._count_phone_numbers)
        df['message_entropy'] = df['message_text'].apply(self._calculate_entropy)
        df['social_engineering'] = df['message_text'].apply(self._detect_social_engineering)
        df['typosquatting'] = df.apply(lambda x: self._detect_typosquatting(x['source_addr'], x['category']), axis=1)
        
        # 6. Поведінкові аномалії
        df['night_time'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        df['weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
        df['source_category_mismatch'] = df.apply(lambda x: self._check_source_category_mismatch(x['source_addr'], x['category']), axis=1)
        df['time_category_anomaly'] = df.apply(lambda x: self._check_time_category_anomaly(x['category'], x['hour']), axis=1)
        df['unusual_sender_pattern'] = df['source_addr'].apply(self._detect_unusual_sender_pattern)
        df['category_time_mismatch'] = df.apply(lambda x: self._check_category_time_mismatch(x['category'], x['hour'], x['day_of_week']), axis=1)
        df['sender_legitimacy'] = df.apply(lambda x: self._calculate_sender_legitimacy(x['source_addr'], x['category']), axis=1)
        
        # 7. Розрахунок агрегованого балу протокольних аномалій
        protocol_features = ['encoding_issues', 'empty_message', 'excessive_length', 
                           'source_is_numeric', 'dest_is_valid']
        df['protocol_anomaly_score'] = df[protocol_features].mean(axis=1)
        
        # Позначка про екстракцію
        df['features_extracted'] = 1
        df['feature_extraction_version'] = self.version
        
        self.logger.info("Екстракція ознак завершена")
        return df
    
    def update_database(self, df: pd.DataFrame, update_columns: Optional[List[str]] = None):
        """
        Оновлення даних в БД
        
        Args:
            df: DataFrame з оновленими даними
            update_columns: Список колонок для оновлення (None = всі)
        """
        if df.empty:
            self.logger.info("Немає даних для оновлення")
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Визначаємо колонки для оновлення
        if update_columns is None:
            update_columns = [
                'category', 'encoding_issues', 'excessive_length', 'dest_is_valid',
                'source_addr_length', 'source_is_numeric', 'empty_message',
                'protocol_anomaly_score', 'sender_frequency', 'recipient_frequency',
                'sender_burst', 'recipient_burst', 'high_sender_frequency',
                'high_recipient_frequency', 'suspicious_word_count', 'url_count',
                'suspicious_url', 'urgency_score', 'financial_patterns',
                'phone_numbers', 'message_entropy', 'obfuscation_score',
                'social_engineering', 'typosquatting', 'night_time', 'weekend',
                'business_hours', 'source_category_mismatch', 'time_category_anomaly',
                'unusual_sender_pattern', 'category_time_mismatch', 'sender_legitimacy',
                'features_extracted', 'feature_extraction_version'
            ]
        
        # Перевіряємо які колонки існують в DataFrame
        existing_columns = [col for col in update_columns if col in df.columns]
        
        if not existing_columns:
            self.logger.warning("Жодна з колонок для оновлення не знайдена в DataFrame")
            conn.close()
            return
        
        # Оновлюємо по батчах
        batch_size = self.batch_size
        total_updated = 0
        
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            for _, row in batch.iterrows():
                # Формуємо SQL запит
                set_clause = ", ".join([f"{col} = ?" for col in existing_columns])
                values = [row[col] for col in existing_columns]
                values.append(row['id'])  # Для WHERE умови
                
                query = f"UPDATE smpp_messages SET {set_clause} WHERE id = ?"
                
                try:
                    cursor.execute(query, values)
                    total_updated += 1
                except Exception as e:
                    self.logger.error(f"Помилка оновлення запису {row['id']}: {e}")
            
            conn.commit()
            self.logger.info(f"Оновлено {end_idx}/{len(df)} записів")
        
        conn.close()
        self.logger.info(f"Всього оновлено {total_updated} записів в БД")
    
    def process_unprocessed_messages(self, batch_size: Optional[int] = None):
        """
        Обробка необроблених повідомлень з БД
        
        Args:
            batch_size: Розмір батчу для обробки
        """
        batch_size = batch_size or self.batch_size
        
        # Завантажуємо необроблені записи
        where_clause = "features_extracted = 0 OR features_extracted IS NULL"
        df = self.load_data_from_db(limit=batch_size, where_clause=where_clause)
        
        if df.empty:
            self.logger.info("Всі записи вже оброблені")
            return
        
        self.logger.info(f"Знайдено {len(df)} необроблених записів")
        
        # Екстракція ознак
        df_processed = self.extract_all_features(df)
        
        # Оновлення БД
        self.update_database(df_processed)
    
    def reprocess_all_messages(self, batch_size: Optional[int] = None):
        """
        Переобробка всіх повідомлень в БД
        
        Args:
            batch_size: Розмір батчу для обробки
        """
        batch_size = batch_size or self.batch_size
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Отримуємо загальну кількість записів
            total_count = pd.read_sql_query("SELECT COUNT(*) as cnt FROM smpp_messages", conn)['cnt'][0]
            conn.close()
            
            self.logger.info(f"Початок переобробки {total_count} записів")
            
            # Обробляємо батчами
            for offset in range(0, total_count, batch_size):
                where_clause = f"1=1 LIMIT {batch_size} OFFSET {offset}"
                df = self.load_data_from_db(limit=batch_size, where_clause=where_clause)
                
                if df.empty:
                    break
                
                # Екстракція ознак
                df_processed = self.extract_all_features(df)
                
                # Оновлення БД
                self.update_database(df_processed)
                
                self.logger.info(f"Оброблено {min(offset + batch_size, total_count)}/{total_count} записів")
                
        except Exception as e:
            self.logger.error(f"Помилка переобробки: {e}")
            conn.close()
    
    # === Методи для розрахунку окремих ознак ===
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищення даних"""
        df = df.copy()
        
        # Заповнення пустих значень
        df['message_text'] = df['message_text'].fillna('')
        df['source_addr'] = df['source_addr'].fillna('')
        df['dest_addr'] = df['dest_addr'].fillna('')
        df['category'] = df['category'].fillna('unknown')
        
        # Перевірка та заповнення час/день тижня
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        return df
    
    def _calculate_frequency_data(self, df: pd.DataFrame):
        """Підготовка даних для частотного аналізу"""
        df_sorted = df.sort_values('timestamp')
        
        for _, row in df_sorted.iterrows():
            self.sender_counter[row['source_addr']].append(row['timestamp'])
            self.recipient_counter[row['dest_addr']].append(row['timestamp'])
    
    def _validate_phone_number(self, phone: str) -> int:
        """Валідація номера телефону"""
        if not phone:
            return 0
        pattern = r'^380[0-9]{9}$'
        return 1 if re.match(pattern, str(phone)) else 0
    
    def _detect_encoding_issues(self, text: str) -> int:
        """Виявлення проблем з кодуванням"""
        if not text:
            return 0
        weird_chars = ['�', '◊', '☺', '♦', '♣', '♠', '♥']
        return 1 if any(char in str(text) for char in weird_chars) else 0
    
    def _calculate_sender_burst(self, source_addr: str, current_time) -> int:
        """Розрахунок burst для відправника"""
        if not source_addr or source_addr not in self.sender_counter:
            return 0
        
        times = self.sender_counter[source_addr]
        window_start = current_time - pd.Timedelta(minutes=3)
        
        count = sum(1 for t in times if window_start <= t <= current_time)
        return 1 if count > 30 else 0
    
    def _calculate_recipient_burst(self, dest_addr: str, current_time) -> int:
        """Розрахунок burst для отримувача"""
        if not dest_addr or dest_addr not in self.recipient_counter:
            return 0
        
        times = self.recipient_counter[dest_addr]
        window_start = current_time - pd.Timedelta(minutes=3)
        
        count = sum(1 for t in times if window_start <= t <= current_time)
        return 1 if count > 15 else 0
    
    def _count_suspicious_words(self, text: str) -> int:
        """Підрахунок підозрілих слів"""
        if not text:
            return 0
        
        count = 0
        text_lower = str(text).lower()
        
        for lang_words in self.suspicious_words.values():
            for word in lang_words:
                if word in text_lower:
                    count += 1
        
        return min(count, 10)
    
    def _count_urls(self, text: str) -> int:
        """Підрахунок URL"""
        if not text:
            return 0
        
        pattern = r'https?://[^\s]+|www\.[^\s]+|[^\s]+\.[a-z]{2,}'
        return len(re.findall(pattern, str(text).lower()))
    
    def _has_suspicious_url(self, text: str) -> int:
        """Перевірка підозрілих URL"""
        if not text:
            return 0
        
        text_lower = str(text).lower()
        return 1 if any(shortener in text_lower for shortener in self.shorteners) else 0
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Розрахунок терміновості"""
        if not text:
            return 0.0
        
        urgency_words = ['термінов', 'негайно', 'зараз', 'швидко', 'останн', 'увага']
        score = 0
        text_lower = str(text).lower()
        
        for word in urgency_words:
            if word in text_lower:
                score += 1
        
        if sum(c.isupper() for c in str(text)) > len(str(text)) * 0.3:
            score += 1
        
        score += min(str(text).count('!'), 3)
        
        return min(score / 5.0, 1.0)
    
    def _count_financial_patterns(self, text: str) -> int:
        """Підрахунок фінансових патернів"""
        if not text:
            return 0
        
        patterns = [
            r'\d+\s*(?:грн|₴|\$|USD|EUR)',
            r'рахунок\s*\d+',
            r'карт[аи]\s*\d+',
            r'баланс',
            r'переказ'
        ]
        
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, str(text).lower()))
        
        return min(count, 5)
    
    def _count_phone_numbers(self, text: str) -> int:
        """Підрахунок номерів телефонів"""
        if not text:
            return 0
        
        pattern = r'\+?3?8?0[0-9]{9}'
        return len(re.findall(pattern, str(text)))
    
    def _calculate_entropy(self, text: str) -> float:
        """Розрахунок ентропії"""
        if not text:
            return 0.0
        
        char_counts = defaultdict(int)
        for char in str(text):
            char_counts[char] += 1
        
        text_len = len(str(text))
        entropy = 0.0
        
        for count in char_counts.values():
            p = count / text_len
            if p > 0:
                entropy -= p * np.log2(p)
        
        return min(entropy / 5.0, 1.0)
    
    def _detect_social_engineering(self, text: str) -> int:
        """Виявлення соціальної інженерії"""
        if not text:
            return 0
        
        patterns = [
            'виграли', 'безкоштов', 'подзвон', 'перекин',
            'заблоков', 'попередж', 'втрат', 'призупин'
        ]
        
        text_lower = str(text).lower()
        return 1 if any(pattern in text_lower for pattern in patterns) else 0
    
    def _detect_typosquatting(self, source_addr: str, category: str) -> float:
        """Виявлення typosquatting"""
        if not source_addr or not category:
            return 0.0
        
        legit_sources = self.source_addresses.get(category, [])
        if not legit_sources:
            return 0.0
        
        source_lower = str(source_addr).lower()
        
        for legit in legit_sources:
            legit_lower = legit.lower()
            
            if source_lower == legit_lower:
                return 0.0
            
            if len(source_lower) == len(legit_lower):
                diff_count = sum(a != b for a, b in zip(source_lower, legit_lower))
                if 1 <= diff_count <= 2:
                    return 1.0
        
        return 0.0
    
    def _check_source_category_mismatch(self, source_addr: str, category: str) -> int:
        """Перевірка невідповідності джерела категорії"""
        if not source_addr or not category:
            return 0
        
        expected_sources = self.source_addresses.get(category, [])
        if not expected_sources:
            return 0
        
        source_upper = str(source_addr).upper()
        
        for expected in expected_sources:
            if expected.upper() in source_upper or source_upper in expected.upper():
                return 0
        
        return 1
    
    def _check_time_category_anomaly(self, category: str, hour: int) -> float:
        """Перевірка аномальності часу для категорії"""
        time_rules = {
            'banking': (8, 20),      # Банки працюють 8-20
            'government': (9, 18),   # Держустанови 9-18
            'medical': (7, 21),      # Медицина 7-21
            'delivery': (6, 22),     # Доставка 6-22
        }
        
        if category not in time_rules:
            return 0.0
        
        start_hour, end_hour = time_rules[category]
        
        if start_hour <= hour <= end_hour:
            return 0.0
        else:
            return 1.0
    
    def _detect_unusual_sender_pattern(self, source_addr: str) -> int:
        """Виявлення незвичайних патернів відправника"""
        if not source_addr:
            return 0
        
        source_str = str(source_addr)
        
        # Змішані літери/цифри в короткому коді
        if len(source_str) <= 6:
            has_letters = any(c.isalpha() for c in source_str)
            has_digits = any(c.isdigit() for c in source_str)
            if has_letters and has_digits:
                return 1
        
        # Повторювані символи
        if len(set(source_str)) < len(source_str) / 2:
            return 1
        
        return 0
    
    def _check_category_time_mismatch(self, category: str, hour: int, day_of_week: int) -> int:
        """Перевірка невідповідності категорії часу"""
        critical_categories = ['banking', 'government']
            
        if category not in critical_categories:
            return 0
            
        # Нічний час (0-6)
        if 0 <= hour <= 6:
            return 1
            
        # Вихідні для держустанов
        if category == 'government' and day_of_week >= 5:
            return 1
            
        return 0
        
    def _calculate_sender_legitimacy(self, source_addr: str, category: str) -> float:
        """Розрахунок легітимності відправника"""
        if not source_addr or not category:
            return 0.0
            
        legit_sources = self.source_addresses.get(category, [])
        if not legit_sources:
            return 0.5
            
        source_upper = str(source_addr).upper()
            
        # Точне співпадіння
        if source_upper in [s.upper() for s in legit_sources]:
            return 1.0
            
        # Часткове співпадіння
        for legit in legit_sources:
            if legit.upper() in source_upper:
                return 0.7
            
        return 0.0
    
    def generate_statistics_report(self, df: pd.DataFrame) -> Dict:
        """
        Генерація статистичного звіту
        
        Args:
            df: DataFrame з обробленими даними
            
        Returns:
            Словник зі статистикою
        """
        if df.empty:
            return {
                'total_messages': 0,
                'processed_messages': 0,
                'categories': {},
                'anomaly_statistics': {},
                'average_scores': {}
            }
        
        # Забезпечуємо наявність необхідних колонок
        required_columns = {
            'features_extracted': 0,
            'category': 'unknown',
            'obfuscation_score': 0.0,
            'sender_burst': 0,
            'recipient_burst': 0,
            'suspicious_url': 0,
            'social_engineering': 0,
            'night_time': 0,
            'weekend': 0,
            'urgency_score': 0.0,
            'message_entropy': 0.0,
            'sender_legitimacy': 0.0
        }
        
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
        
        stats = {
            'total_messages': len(df),
            'processed_messages': int(df['features_extracted'].sum()) if 'features_extracted' in df.columns else 0,
            'categories': df['category'].value_counts().to_dict(),
            'anomaly_statistics': {
                'high_obfuscation': int((df['obfuscation_score'] > 0.4).sum()),
                'sender_bursts': int(df['sender_burst'].sum()),
                'recipient_bursts': int(df['recipient_burst'].sum()),
                'suspicious_urls': int(df['suspicious_url'].sum()),
                'social_engineering': int(df['social_engineering'].sum()),
                'night_time_messages': int(df['night_time'].sum()),
                'weekend_messages': int(df['weekend'].sum())
            },
            'average_scores': {
                'obfuscation': float(df['obfuscation_score'].mean()),
                'urgency': float(df['urgency_score'].mean()),
                'entropy': float(df['message_entropy'].mean()),
                'sender_legitimacy': float(df['sender_legitimacy'].mean())
            }
        }
        
        return stats


def main():
    """Основна функція для повної обробки всіх повідомлень"""
    
    try:
        # Показуємо поточний робочий каталог
        print(f"Поточний робочий каталог: {os.getcwd()}")
        print(f"Файли в поточному каталозі: {os.listdir('.')}")
        
        # Ініціалізація екстрактора з діагностикою
        print("\n=== Ініціалізація екстрактора ===")
        extractor = SMPPFeatureExtractor()
        
        # Параметри обробки
        BATCH_SIZE = 1000  # <-- ТУТ МОЖНА ЗМІНИТИ КІЛЬКІСТЬ ЗАПИСІВ ЗА РАЗ
        TOTAL_LIMIT = None  # <-- ТУТ МОЖНА ВКАЗАТИ ЗАГАЛЬНЕ ОБМЕЖЕННЯ (None = всі записи)
        
        print(f"Налаштування обробки:")
        print(f"  Розмір батчу: {BATCH_SIZE}")
        print(f"  Загальне обмеження: {TOTAL_LIMIT if TOTAL_LIMIT else 'Всі записи'}")
        
        # Отримуємо загальну кількість записів
        conn = sqlite3.connect(extractor.db_path)
        total_count_query = "SELECT COUNT(*) as cnt FROM smpp_messages"
        if TOTAL_LIMIT:
            total_count_query += f" LIMIT {TOTAL_LIMIT}"
        
        total_count = pd.read_sql_query(total_count_query, conn)['cnt'][0]
        conn.close()
        
        print(f"\nВсього записів для обробки: {total_count}")
        
        if total_count == 0:
            print("База даних порожня")
            return
        
        # ЕТАП 1: КЛАСИФІКАЦІЯ ВСІХ ПОВІДОМЛЕНЬ
        print(f"\n{'='*50}")
        print("ЕТАП 1: КЛАСИФІКАЦІЯ ВСІХ ПОВІДОМЛЕНЬ")
        print(f"{'='*50}")
        
        processed_classification = 0
        for offset in range(0, total_count, BATCH_SIZE):
            print(f"\nОбробка класифікації: записи {offset+1}-{min(offset+BATCH_SIZE, total_count)} з {total_count}")
            
            # Завантажуємо батч - правильний синтаксис для SQLite
            where_clause = f"1=1 LIMIT {BATCH_SIZE} OFFSET {offset}"
            df_batch = extractor.load_data_from_db(limit=None, where_clause=where_clause)
            
            if df_batch.empty:
                break
            
            # Класифікуємо
            df_classified = extractor.classify_messages(df_batch)
            
            # Оновлюємо тільки категорії в БД
            if 'predicted_category' in df_classified.columns:
                df_classified['category'] = df_classified['predicted_category']
                extractor.update_database(df_classified, update_columns=['category'])
            
            processed_classification += len(df_classified)
            print(f"  Класифіковано: {len(df_classified)} записів")
            
            # Показуємо розподіл по категоріях для цього батчу
            if 'category' in df_classified.columns:
                category_counts = df_classified['category'].value_counts()
                print(f"  Розподіл по категоріях в батчі: {dict(category_counts)}")
        
        print(f"\nКЛАСИФІКАЦІЯ ЗАВЕРШЕНА: оброблено {processed_classification} записів")
        
        # ЕТАП 2: РОЗРАХУНОК ОБФУСКАЦІЇ ДЛЯ ВСІХ ПОВІДОМЛЕНЬ
        print(f"\n{'='*50}")
        print("ЕТАП 2: РОЗРАХУНОК ОБФУСКАЦІЇ ДЛЯ ВСІХ ПОВІДОМЛЕНЬ")
        print(f"{'='*50}")
        
        processed_obfuscation = 0
        for offset in range(0, total_count, BATCH_SIZE):
            print(f"\nОбробка обфускації: записи {offset+1}-{min(offset+BATCH_SIZE, total_count)} з {total_count}")
            
            # Завантажуємо батч - правильний синтаксис для SQLite
            where_clause = f"1=1 LIMIT {BATCH_SIZE} OFFSET {offset}"
            df_batch = extractor.load_data_from_db(limit=None, where_clause=where_clause)
            
            if df_batch.empty:
                break
            
            # Розраховуємо обфускацію
            df_obfuscated = extractor.calculate_obfuscation_scores(df_batch)
            
            # Оновлюємо тільки обфускацію в БД
            extractor.update_database(df_obfuscated, update_columns=['obfuscation_score'])
            
            processed_obfuscation += len(df_obfuscated)
            print(f"  Оброблено обфускацію: {len(df_obfuscated)} записів")
            
            # Показуємо статистику обфускації для батчу
            if 'obfuscation_score' in df_obfuscated.columns:
                avg_obfuscation = df_obfuscated['obfuscation_score'].mean()
                high_obfuscation = (df_obfuscated['obfuscation_score'] > 0.4).sum()
                print(f"  Середня обфускація в батчі: {avg_obfuscation:.3f}")
                print(f"  Висока обфускація (>0.4): {high_obfuscation} записів")
        
        print(f"\nОБФУСКАЦІЯ ЗАВЕРШЕНА: оброблено {processed_obfuscation} записів")
        
        # ЕТАП 3: ЕКСТРАКЦІЯ ВСІХ ОЗНАК
        print(f"\n{'='*50}")
        print("ЕТАП 3: ЕКСТРАКЦІЯ ВСІХ ОЗНАК")
        print(f"{'='*50}")
        
        processed_features = 0
        for offset in range(0, total_count, BATCH_SIZE):
            print(f"\nЕкстракція ознак: записи {offset+1}-{min(offset+BATCH_SIZE, total_count)} з {total_count}")
            
            # Завантажуємо батч з уже класифікованими та обфускованими даними
            where_clause = f"1=1 LIMIT {BATCH_SIZE} OFFSET {offset}"
            df_batch = extractor.load_data_from_db(limit=None, where_clause=where_clause)
            
            if df_batch.empty:
                break
            
            # Екстракція всіх ознак (окрім класифікації та обфускації, які вже є)
            df_features = extractor.extract_all_features(df_batch)
            
            # Оновлюємо всі ознаки в БД
            extractor.update_database(df_features)
            
            processed_features += len(df_features)
            print(f"  Ознаки екстраговано: {len(df_features)} записів")
        
        print(f"\nЕКСТРАКЦІЯ ОЗНАК ЗАВЕРШЕНА: оброблено {processed_features} записів")
        
        # ЕТАП 4: ФІНАЛЬНА СТАТИСТИКА
        print(f"\n{'='*50}")
        print("ЕТАП 4: ФІНАЛЬНА СТАТИСТИКА")
        print(f"{'='*50}")
        
        # Завантажуємо всі оброблені дані для статистики
        final_df = extractor.load_data_from_db(limit=total_count)
        
        if not final_df.empty:
            stats = extractor.generate_statistics_report(final_df)
            
            print("\n=== ПІДСУМКОВИЙ СТАТИСТИЧНИЙ ЗВІТ ===")
            print(f"Всього повідомлень: {stats['total_messages']}")
            print(f"Повністю оброблених: {stats['processed_messages']}")
            
            print(f"\nРозподіл по категоріях:")
            for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_messages']) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
            
            print(f"\nСтатистика аномалій:")
            for anomaly, count in stats['anomaly_statistics'].items():
                percentage = (count / stats['total_messages']) * 100 if stats['total_messages'] > 0 else 0
                print(f"  {anomaly}: {count} ({percentage:.1f}%)")
            
            print(f"\nСередні показники:")
            for metric, value in stats['average_scores'].items():
                print(f"  {metric}: {value:.3f}")
            
            # Додаткова статистика
            print(f"\nДодаткова інформація:")
            if 'obfuscation_score' in final_df.columns:
                high_obf = (final_df['obfuscation_score'] > 0.4).sum()
                medium_obf = ((final_df['obfuscation_score'] > 0.2) & (final_df['obfuscation_score'] <= 0.4)).sum()
                low_obf = (final_df['obfuscation_score'] <= 0.2).sum()
                print(f"  Висока обфускація (>0.4): {high_obf}")
                print(f"  Середня обфускація (0.2-0.4): {medium_obf}")
                print(f"  Низька обфускація (≤0.2): {low_obf}")
        
        print(f"\n{'='*50}")
        print("ПОВНА ОБРОБКА ЗАВЕРШЕНА УСПІШНО!")
        print(f"{'='*50}")
        print(f"Оброблено:")
        print(f"  - Класифікація: {processed_classification} записів")
        print(f"  - Обфускація: {processed_obfuscation} записів") 
        print(f"  - Ознаки: {processed_features} записів")
        
    except Exception as e:
        print(f"Помилка під час виконання: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    main()