"""
Enhanced SMPP Dataset Generator with SQLite Database
Генератор датасету з типологізованими аномаліями для SQLite
"""

import random
import datetime
import logging
import json
import os
import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, Tuple, List
import hashlib
from collections import defaultdict
import string

# Створення папок
os.makedirs("logs", exist_ok=True)
os.makedirs("data/db", exist_ok=True)

# Налаштування логування
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/generator_e_dataset_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

class EnhancedSMPPDatasetGenerator:
    """Покращений генератор SMPP датасету з типологізованими аномаліями та SQLite збереженням"""
    
    def __init__(self, seed=42, config_path="data/data_config.json", db_path="data/db/smpp_dataset.db"):
        random.seed(seed)
        np.random.seed(seed)
        
        self.db_path = db_path
        
        # Завантаження конфігурації
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.value_config = config.get("value_config", {})
        self.value_lists = config.get("value_lists", {})
        self.shorteners = config.get("shorteners", [])
        
        # Шаблони повідомлень
        self.templates = {
            cat: self._load_templates(path) for cat, path in config["template_files"].items()
        }
        self.fraud_templates = self._load_templates(config["fraud_templates"])
        
        # Джерела відправників
        self.source_addresses = config["source_addresses"]
        
        # Оператори
        self.ua_operators = config.get("ua_operators", [])
        
        # Словник для обфускації (латиниця -> кирилиця)
        self.obfuscation_map = {
            'a': 'а', 'A': 'А',
            'e': 'е', 'E': 'Е', 
            'i': 'і', 'I': 'І',
            'o': 'о', 'O': 'О',
            'p': 'р', 'P': 'Р',
            'c': 'с', 'C': 'С',
            'y': 'у', 'Y': 'У',
            'x': 'х', 'X': 'Х',
            'H': 'Н', 'B': 'В',
            'M': 'М', 'T': 'Т',
            'K': 'К'
        }
        
        # Типи аномалій
        self.anomaly_types = [
            "structural",
            "frequency", 
            "semantic",
            "behavioral",
            "obfuscation",
            "sender_mismatch"
        ]
        
        # Лічильники для частотних аномалій
        self.sender_messages = defaultdict(list)
        self.recipient_messages = defaultdict(list)
        
        # Статистика
        self.anomaly_stats = defaultdict(int)
        
        # Ініціалізація бази даних
        self._init_database()
        
        logging.info("Enhanced SMPP Dataset Generator with SQLite initialized")
    
    def _init_database(self):
        """Ініціалізація SQLite бази даних"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Створення таблиці для SMPP повідомлень
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS smpp_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    source_addr TEXT NOT NULL,
                    dest_addr TEXT NOT NULL,
                    submit_time DATETIME NOT NULL,
                    message_text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    is_anomaly BOOLEAN NOT NULL,
                    anomaly_type TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(message_id)
                )
            ''')
            
            # Створення індексів для оптимізації запитів
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_submit_time ON smpp_messages(submit_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_addr ON smpp_messages(source_addr)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dest_addr ON smpp_messages(dest_addr)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_anomaly ON smpp_messages(is_anomaly)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomaly_type ON smpp_messages(anomaly_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON smpp_messages(category)')
            
            # Створення таблиці для статистики генерації
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generation_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER,
                    normal_messages INTEGER,
                    anomaly_messages INTEGER,
                    anomaly_rate REAL,
                    anomaly_breakdown TEXT,
                    notes TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logging.info(f"База даних ініціалізована: {self.db_path}")
            
        except Exception as e:
            logging.error(f"Помилка ініціалізації бази даних: {e}")
            raise
    
    def _save_to_database(self, records: List[Dict], batch_size: int = 1000):
        """Збереження записів до SQLite бази даних пакетами"""
        if not records:
            logging.warning("Немає записів для збереження")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Підготовка даних для вставки
            insert_query = '''
                INSERT OR IGNORE INTO smpp_messages 
                (message_id, source_addr, dest_addr, submit_time, message_text, 
                 category, is_anomaly, anomaly_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            inserted_count = 0
            
            # Вставка пакетами для кращої продуктивності
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_data = []
                
                for record in batch:
                    batch_data.append((
                        record['message_id'],
                        record['source_addr'],
                        record['dest_addr'],
                        record['submit_time'].isoformat(),
                        record['message_text'],
                        record['category'],
                        record['is_anomaly'],
                        record['anomaly_type']
                    ))
                
                cursor.executemany(insert_query, batch_data)
                inserted_count += cursor.rowcount
                
                # Логування прогресу
                if i % (batch_size * 10) == 0:
                    logging.info(f"Збережено {i + len(batch)}/{len(records)} записів...")
            
            conn.commit()
            conn.close()
            
            logging.info(f"Успішно збережено {inserted_count} нових записів до бази даних")
            return inserted_count
            
        except Exception as e:
            logging.error(f"Помилка збереження до бази даних: {e}")
            raise
    
    def _save_generation_stats(self, df: pd.DataFrame, notes: str = ""):
        """Збереження статистики генерації"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            total_messages = len(df)
            normal_messages = len(df[~df['is_anomaly']])
            anomaly_messages = len(df[df['is_anomaly']])
            anomaly_rate = anomaly_messages / total_messages if total_messages > 0 else 0
            
            # Розподіл аномалій по типах
            anomaly_breakdown = df[df['is_anomaly']]['anomaly_type'].value_counts().to_dict()
            anomaly_breakdown_json = json.dumps(anomaly_breakdown, ensure_ascii=False)
            
            cursor.execute('''
                INSERT INTO generation_stats 
                (total_messages, normal_messages, anomaly_messages, anomaly_rate, anomaly_breakdown, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (total_messages, normal_messages, anomaly_messages, anomaly_rate, anomaly_breakdown_json, notes))
            
            conn.commit()
            conn.close()
            
            logging.info("Статистика генерації збережена")
            
        except Exception as e:
            logging.error(f"Помилка збереження статистики: {e}")
    
    def get_database_stats(self):
        """Отримання статистики з бази даних"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Загальна статистика
            total_query = "SELECT COUNT(*) FROM smpp_messages"
            total_messages = pd.read_sql_query(total_query, conn).iloc[0, 0]
            
            # Статистика аномалій
            anomaly_query = """
                SELECT is_anomaly, COUNT(*) as count 
                FROM smpp_messages 
                GROUP BY is_anomaly
            """
            anomaly_stats = pd.read_sql_query(anomaly_query, conn)
            
            # Розподіл по типах аномалій
            anomaly_type_query = """
                SELECT anomaly_type, COUNT(*) as count 
                FROM smpp_messages 
                WHERE is_anomaly = 1 
                GROUP BY anomaly_type
            """
            anomaly_type_stats = pd.read_sql_query(anomaly_type_query, conn)
            
            # Розподіл по категоріях
            category_query = """
                SELECT category, is_anomaly, COUNT(*) as count 
                FROM smpp_messages 
                GROUP BY category, is_anomaly
            """
            category_stats = pd.read_sql_query(category_query, conn)
            
            # Часовий діапазон
            time_range_query = """
                SELECT MIN(submit_time) as min_time, MAX(submit_time) as max_time 
                FROM smpp_messages
            """
            time_range = pd.read_sql_query(time_range_query, conn)
            
            conn.close()
            
            return {
                'total_messages': total_messages,
                'anomaly_stats': anomaly_stats,
                'anomaly_type_stats': anomaly_type_stats,
                'category_stats': category_stats,
                'time_range': time_range
            }
            
        except Exception as e:
            logging.error(f"Помилка отримання статистики: {e}")
            return None
    
    def export_to_csv(self, output_file: str, limit: int = None):
        """Експорт даних з бази даних до CSV файлу"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM smpp_messages ORDER BY submit_time"
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"Дані експортовано до {output_file}")
            
        except Exception as e:
            logging.error(f"Помилка експорту до CSV: {e}")
    
    def _load_templates(self, filename):
        """Завантаження шаблонів з файлу"""
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    
    def generate_phone_number(self) -> str:
        """Генерує український номер телефону"""
        prefix = "380"
        operator = random.choice(self.ua_operators)
        number = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        return f"{prefix}{operator}{number}"
    
    def generate_source_addr(self, category: str, is_fraud: bool = False) -> str:
        """Генерує адресу відправника"""
        if is_fraud and random.random() > 0.3:
            # Для фроду - спотворені адреси
            legit = random.choice(self.source_addresses.get(category, ["UNKNOWN"]))
            mutations = [
                lambda x: x.lower(),
                lambda x: x.replace('I', '1'),
                lambda x: x.replace('O', '0'), 
                lambda x: x.replace('A', '4'),
                lambda x: x + "24",
                lambda x: x + "-INFO",
                lambda x: "SMS-" + x,
                lambda x: self.obfuscate_text(x)
            ]
            return random.choice(mutations)(legit)
        
        return random.choice(self.source_addresses.get(category, ["INFO"]))
    
    def obfuscate_text(self, text: str) -> str:
        """Обфускація тексту - заміна латинських літер на схожі кириличні"""
        obfuscated = ""
        for char in text:
            if char in self.obfuscation_map and random.random() > 0.5:
                obfuscated += self.obfuscation_map[char]
            else:
                obfuscated += char
        return obfuscated
    
    def generate_normal_message(self, category: str, submit_time: datetime.datetime) -> Dict:
        """Генерація нормального повідомлення"""
        template = random.choice(self.templates[category])
        
        # Генерація значень для підстановки
        values = self._generate_template_values()
        message = self._fill_template(template, values)
        
        source_addr = self.generate_source_addr(category, is_fraud=False)
        dest_addr = self.generate_phone_number()
        
        return {
            'message_id': self._generate_message_id(source_addr, dest_addr, submit_time, message),
            'source_addr': source_addr,
            'dest_addr': dest_addr,
            'submit_time': submit_time,
            'message_text': message,
            'category': category,
            'is_anomaly': False,
            'anomaly_type': None
        }
    
    def generate_structural_anomaly(self, category: str, submit_time: datetime.datetime) -> Dict:
        """Генерація структурної аномалії"""
        anomaly_type = random.choice([
            'excessive_length',
            'invalid_timestamp',
            'encoding_error',
            'missing_fields'
        ])
        
        template = random.choice(self.fraud_templates)
        values = self._generate_template_values()
        message = self._fill_template(template, values)
        
        if anomaly_type == 'excessive_length':
            # Надмірна довжина повідомлення
            message = message + " " + "".join(random.choices(string.ascii_letters + string.digits, k=500))
        
        elif anomaly_type == 'invalid_timestamp':
            # Некоректна мітка часу
            submit_time = datetime.datetime(2099, 12, 31, 23, 59, 59)
        
        elif anomaly_type == 'encoding_error':
            # Додаємо дивні символи
            weird_chars = ['�', '◊', '☺', '♦', '♣', '♠', '♥']
            message = message + " " + "".join(random.choices(weird_chars, k=10))
        
        elif anomaly_type == 'missing_fields':
            # Пусте повідомлення
            message = ""
        
        source_addr = self.generate_source_addr(category, is_fraud=True)
        dest_addr = self.generate_phone_number()
        
        self.anomaly_stats['structural'] += 1
        
        return {
            'message_id': self._generate_message_id(source_addr, dest_addr, submit_time, message),
            'source_addr': source_addr,
            'dest_addr': dest_addr,
            'submit_time': submit_time,
            'message_text': message,
            'category': category,
            'is_anomaly': True,
            'anomaly_type': 'structural'
        }
    
    def generate_frequency_anomaly(self, category: str, base_time: datetime.datetime) -> List[Dict]:
        """Генерація частотної аномалії - багато повідомлень за короткий час"""
        messages = []

        sensitive_categories = ['bank', 'government', 'otp', 'security']
         # Якщо категорія чутлива — можемо випадково обрати any subtype
        # Якщо ні — only 'same_recipient'
        if category in sensitive_categories:
            anomaly_subtype = random.choice(['same_sender', 'same_recipient'])
        else:
            anomaly_subtype = 'same_recipient'
        
        if anomaly_subtype == 'same_sender':
            # >30 повідомлень від одного відправника за 3 хвилини
            source_addr = self.generate_source_addr(category, is_fraud=True)
            message_count = random.randint(31, 40)
            template = random.choice(self.fraud_templates)
            
            for _ in range(message_count):
                submit_time = base_time + datetime.timedelta(seconds=random.randint(0, 180))
                dest_addr = self.generate_phone_number()
                
                values = self._generate_template_values()
                message = self._fill_template(template, values)
                
                record = {
                    'message_id': self._generate_message_id(source_addr, dest_addr, submit_time, message),
                    'source_addr': source_addr,
                    'dest_addr': dest_addr,
                    'submit_time': submit_time,
                    'message_text': message,
                    'category': category,
                    'is_anomaly': True,
                    'anomaly_type': 'frequency'
                }
                messages.append(record)
        
        else:  # same_recipient
            # >15 повідомлень на один номер за 3 хвилини
            dest_addr = self.generate_phone_number()
            message_count = random.randint(16, 20)
            
            for _ in range(message_count):
                submit_time = base_time + datetime.timedelta(seconds=random.randint(0, 180))
                source_addr = self.generate_source_addr(category, is_fraud=True)
                
                template = random.choice(self.fraud_templates)
                values = self._generate_template_values()
                message = self._fill_template(template, values)
                
                record = {
                    'message_id': self._generate_message_id(source_addr, dest_addr, submit_time, message),
                    'source_addr': source_addr,
                    'dest_addr': dest_addr,
                    'submit_time': submit_time,
                    'message_text': message,
                    'category': category,
                    'is_anomaly': True,
                    'anomaly_type': 'frequency'
                }
                messages.append(record)

        self.anomaly_stats['frequency'] += len(messages)
        return messages
    
    def generate_semantic_anomaly(self, category: str, submit_time: datetime.datetime) -> Dict:
        """Генерація семантичної аномалії - фішингові повідомлення"""
        template = random.choice(self.fraud_templates)
        values = self._generate_template_values()
        
        # Генерація підозрілих URL
        suspicious_domains = [
            'bit.ly/secure', 'tinyurl.com/bank', 'clck.ru/pay',
            'privat24-bank.com', 'monobank-ua.net', 'oschadbank.click',
            'secure-payment.link', 'verify-account.info'
        ]
        
        values['url'] = f"https://{random.choice(suspicious_domains)}/{self._random_string(8)}"
        
        message = self._fill_template(template, values)
        
        # Додаємо терміновість
        urgency_phrases = [
            "ТЕРМІНОВО!", "УВАГА!", "ОСТАННІЙ ДЕНЬ!", 
            "Діє лише 24 години!", "Негайна дія потрібна!"
        ]
        message = random.choice(urgency_phrases) + " " + message
        
        # Обфусковані джерела
        source_addr = self.obfuscate_text(
            random.choice(self.source_addresses.get(category, ["BANK"]))
        )
        dest_addr = self.generate_phone_number()
        
        self.anomaly_stats['semantic'] += 1
        
        return {
            'message_id': self._generate_message_id(source_addr, dest_addr, submit_time, message),
            'source_addr': source_addr,
            'dest_addr': dest_addr,
            'submit_time': submit_time,
            'message_text': message,
            'category': category,
            'is_anomaly': True,
            'anomaly_type': 'semantic'
        }
    
    def generate_behavioral_anomaly(self, category: str) -> Dict:
        """Генерація поведінкової аномалії - повідомлення в неробочий час"""
        # Визначаємо категорії з обмеженнями часу
        time_sensitive_categories = ['banking', 'government']
        
        if category not in time_sensitive_categories:
            category = random.choice(time_sensitive_categories)
        
        # Генеруємо час: або ніч (0-6), або вихідні
        if random.random() > 0.5:
            # Нічний час
            hour = random.randint(0, 5)
            day_offset = random.randint(0, 6)  # Будь-який день
        else:
            # Вихідні
            hour = random.randint(8, 20)
            # Субота (5) або неділя (6)
            days_until_weekend = (5 - datetime.datetime.now().weekday()) % 7
            if days_until_weekend == 0:
                days_until_weekend = 7
            day_offset = days_until_weekend + random.randint(0, 1)
        
        submit_time = datetime.datetime.now() + datetime.timedelta(days=day_offset)
        submit_time = submit_time.replace(hour=hour, minute=random.randint(0, 59))
        
        # Генеруємо офіційне повідомлення
        template = random.choice(self.templates[category])
        values = self._generate_template_values()
        message = self._fill_template(template, values)
        
        source_addr = self.generate_source_addr(category, is_fraud=True)
        dest_addr = self.generate_phone_number()
        
        self.anomaly_stats['behavioral'] += 1
        
        return {
            'message_id': self._generate_message_id(source_addr, dest_addr, submit_time, message),
            'source_addr': source_addr,
            'dest_addr': dest_addr,
            'submit_time': submit_time,
            'message_text': message,
            'category': category,
            'is_anomaly': True,
            'anomaly_type': 'behavioral'
        }
    
    def generate_obfuscation_anomaly(self, category: str, submit_time: datetime.datetime) -> Dict:
        """Генерація обфускаційної аномалії - змішані символи"""
        template = random.choice(self.fraud_templates)
        values = self._generate_template_values()
        message = self._fill_template(template, values)
        
        # Обфускуємо частину тексту
        words = message.split()
        obfuscated_words = []
        
        for word in words:
            if random.random() > 0.3:  # 70% шанс обфускації слова
                obfuscated_words.append(self.obfuscate_text(word))
            else:
                obfuscated_words.append(word)
        
        message = " ".join(obfuscated_words)
        
        # Додаємо Unicode хитрощі
        unicode_tricks = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
        ]
        
        # Вставляємо невидимі символи
        for _ in range(random.randint(1, 3)):
            pos = random.randint(0, len(message))
            trick = random.choice(unicode_tricks)
            message = message[:pos] + trick + message[pos:]
        
        source_addr = self.obfuscate_text(self.generate_source_addr(category, is_fraud=True))
        dest_addr = self.generate_phone_number()
        
        self.anomaly_stats['obfuscation'] += 1
        
        return {
            'message_id': self._generate_message_id(source_addr, dest_addr, submit_time, message),
            'source_addr': source_addr,
            'dest_addr': dest_addr,
            'submit_time': submit_time,
            'message_text': message,
            'category': category,
            'is_anomaly': True,
            'anomaly_type': 'obfuscation'
        }
    
    def generate_sender_mismatch_anomaly(self, submit_time: datetime.datetime) -> Dict:
        """Генерація аномалії невідповідності відправника"""
        # Вибираємо категорію повідомлення
        message_category = random.choice(list(self.templates.keys()))
        
        # Але відправника з іншої категорії
        sender_categories = [cat for cat in self.source_addresses.keys() if cat != message_category]
        sender_category = random.choice(sender_categories)
        
        # Генеруємо повідомлення однієї категорії
        template = random.choice(self.templates[message_category])
        values = self._generate_template_values()
        message = self._fill_template(template, values)
        
        # Але відправник з іншої
        source_addr = random.choice(self.source_addresses[sender_category])
        dest_addr = self.generate_phone_number()
        
        self.anomaly_stats['sender_mismatch'] += 1
        
        return {
            'message_id': self._generate_message_id(source_addr, dest_addr, submit_time, message),
            'source_addr': source_addr,
            'dest_addr': dest_addr,
            'submit_time': submit_time,
            'message_text': message,
            'category': message_category,
            'is_anomaly': True,
            'anomaly_type': 'sender_mismatch'
        }
    
    def generate_weekly_traffic(self, messages_per_day: int = 5000, anomaly_rate: float = 0.15, save_to_db: bool = True) -> pd.DataFrame:
        """Генерація тижневого трафіку з реалістичним розподілом"""
        records = []
        
        # Початок тижня (понеділок)
        start_date = datetime.datetime.now() - datetime.timedelta(days=datetime.datetime.now().weekday())
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        logging.info(f"Генерація тижневого трафіку з {start_date}")
        
        # Розподіл трафіку по днях тижня (менше у вихідні)
        daily_weights = [1.0, 1.0, 1.0, 1.0, 0.9, 0.5, 0.4]  # Пн-Нд
        
        # Розподіл трафіку по годинах (пік вдень)
        hourly_weights = np.exp(-0.5 * ((np.arange(24) - 14) / 6) ** 2)
        hourly_weights = hourly_weights / hourly_weights.sum()
        
        categories = list(self.templates.keys())
        
        for day in range(7):
            current_date = start_date + datetime.timedelta(days=day)
            day_messages = int(messages_per_day * daily_weights[day])
            
            logging.info(f"День {day+1}/7: {current_date.strftime('%Y-%m-%d')} - {day_messages} повідомлень")
            
            # Розподіл повідомлень по годинах
            for hour in range(24):
                hour_messages = int(day_messages * hourly_weights[hour])
                
                for _ in range(hour_messages):
                    # Випадковий час в межах години
                    submit_time = current_date.replace(hour=hour) + datetime.timedelta(
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59)
                    )
                    
                    # Визначаємо чи це аномалія
                    if random.random() < anomaly_rate:
                        # Частотна аномалія буде мати у 20 разів меншу ймовірність
                        anomaly_weights = {
                            'structural': 1,
                            'semantic': 1,
                            'behavioral': 1,
                            'obfuscation': 1,
                            'sender_mismatch': 1,
                            'frequency': 0.05  # у 20 разів менше, ніж інші
                        }

                        # Створюємо список типів і ваг
                        types, weights = zip(*anomaly_weights.items())
                        anomaly_type = random.choices(types, weights=weights, k=1)[0]

                        category = random.choice(categories)
                    
                        if anomaly_type == 'structural':
                            record = self.generate_structural_anomaly(category, submit_time)
                            records.append(record)
                        
                        elif anomaly_type == 'frequency':
                            # Частотні аномалії генерують багато записів
                            freq_records = self.generate_frequency_anomaly(category, submit_time)
                            records.extend(freq_records)
                        
                        elif anomaly_type == 'semantic':
                            record = self.generate_semantic_anomaly(category, submit_time)
                            records.append(record)
                        
                        elif anomaly_type == 'behavioral':
                            record = self.generate_behavioral_anomaly(category)
                            records.append(record)
                        
                        elif anomaly_type == 'obfuscation':
                            record = self.generate_obfuscation_anomaly(category, submit_time)
                            records.append(record)
                        
                        elif anomaly_type == 'sender_mismatch':
                            record = self.generate_sender_mismatch_anomaly(submit_time)
                            records.append(record)
                    
                    else:
                        # Нормальне повідомлення
                        category = random.choice(categories)
                        record = self.generate_normal_message(category, submit_time)
                        records.append(record)
        
        # Створюємо DataFrame
        df = pd.DataFrame(records)
        
        # Сортуємо за часом
        df = df.sort_values('submit_time').reset_index(drop=True)
        
        # Збереження до бази даних
        if save_to_db:
            inserted_count = self._save_to_database(records)
            self._save_generation_stats(df, f"Weekly traffic generation: {messages_per_day} msg/day, {anomaly_rate:.1%} anomaly rate")
            logging.info(f"Збережено {inserted_count} записів до бази даних")
        
        # Логування статистики
        self._log_statistics(df)
        
        return df
    
    def _generate_template_values(self) -> Dict:
        """Генерація значень для шаблонів"""
        return {
            "amount": random.randint(*self.value_config.get("amount_range", [10, 50000])),
            "balance": random.randint(*self.value_config.get("balance_range", [0, 100000])),
            "card": ''.join([str(random.randint(0, 9)) for _ in range(4)]),
            "code": ''.join([str(random.randint(0, 9)) for _ in range(random.randint(4, 6))]),
            "date": (datetime.datetime.now() + datetime.timedelta(days=random.randint(0, 30))).strftime("%d.%m"),
            "time": f"{random.randint(8, 20)}:{random.randint(0, 59):02d}",
            "tracking": ''.join([str(random.randint(0, 9)) for _ in range(14)]),
            "order_id": ''.join([str(random.randint(0, 9)) for _ in range(8)]),
            "branch": random.randint(1, 500),
            "merchant": random.choice(self.value_lists.get("merchant", ["АТБ"])),
            "sender": random.choice(self.value_lists.get("sender", ["Іван"])),
            "service": random.choice(self.value_lists.get("service", ["Gmail"])),
            "ticket": ''.join([str(random.randint(0, 9)) for _ in range(10)]),
            "driver": random.choice(self.value_lists.get("driver", ["Олександр"])),
            "car_number": f"AA{random.randint(1000, 9999)}AA",
            "number": self.generate_phone_number(),
            "minutes": random.randint(2, 15),
            "passenger": random.choice(self.value_lists.get("passenger", ["Оксана"])),
            "flight": f"PS{random.randint(100, 999)}",
            "train": f"{random.randint(1, 200)}К",
            "doctor": random.choice(self.value_lists.get("doctor", ["Петренко І.І."])),
            "discount": random.choice([10, 20, 30, 50]),
            "promo": ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6)),
            "shop": random.choice(self.value_lists.get("shop", ["ZARA"])),
            "location": random.choice(self.value_lists.get("location", ["Київ"])),
            "device": random.choice(self.value_lists.get("device", ["iPhone"])),
            "subject": random.choice(self.value_lists.get("subject", ["Математика"])),
            "course": random.choice(self.value_lists.get("course", ["Python"])),
            "queue": random.randint(1, 999),
            "booking_id": ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8)),
            "destination": random.choice(self.value_lists.get("destination", ["Париж"])),
            "host": random.choice(self.value_lists.get("host", ["John"])),
            "hotel": random.choice(self.value_lists.get("hotel", ["Hilton"])),
            "url": ""  # Буде заповнено для аномалій
        }
    
    def _fill_template(self, template: str, values: Dict) -> str:
        """Заповнення шаблону значеннями"""
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            if placeholder in template:
                template = template.replace(placeholder, str(value))
        return template
    
    def _generate_message_id(self, source: str, dest: str, time: datetime.datetime, message: str) -> str:
        """Генерація унікального ID повідомлення"""
        data = f"{source}{dest}{time}{message}{random.random()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _random_string(self, length: int) -> str:
        """Генерація випадкового рядка"""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    def _log_statistics(self, df: pd.DataFrame):
        """Логування статистики датасету"""
        logging.info("\n" + "="*50)
        logging.info("СТАТИСТИКА ЗГЕНЕРОВАНОГО ДАТАСЕТУ")
        logging.info("="*50)
        
        logging.info(f"Загальна кількість записів: {len(df)}")
        logging.info(f"Нормальних повідомлень: {len(df[~df['is_anomaly']])}")
        logging.info(f"Аномальних повідомлень: {len(df[df['is_anomaly']])}")
        logging.info(f"Відсоток аномалій: {len(df[df['is_anomaly']]) / len(df) * 100:.2f}%")
        
        logging.info("\nРозподіл по категоріях:")
        category_dist = df.groupby(['category', 'is_anomaly']).size().unstack(fill_value=0)
        logging.info(f"\n{category_dist}")
        
        logging.info("\nРозподіл аномалій по типах:")
        anomaly_dist = df[df['is_anomaly']]['anomaly_type'].value_counts()
        for atype, count in anomaly_dist.items():
            logging.info(f"  {atype}: {count} ({count/len(df[df['is_anomaly']])*100:.1f}%)")
        
        logging.info("\nРозподіл по днях тижня:")
        df['weekday'] = pd.to_datetime(df['submit_time']).dt.day_name()
        weekday_dist = df['weekday'].value_counts()
        logging.info(f"\n{weekday_dist}")
        
        logging.info("\nТоп-10 відправників з найбільшою кількістю повідомлень:")
        top_senders = df['source_addr'].value_counts().head(10)
        for sender, count in top_senders.items():
            logging.info(f"  {sender}: {count} повідомлень")
        
        logging.info("\nЧасовий діапазон:")
        logging.info(f"  Від: {df['submit_time'].min()}")
        logging.info(f"  До: {df['submit_time'].max()}")
        
        logging.info("="*50)


# Використання
if __name__ == "__main__":
    # Створення генератора з SQLite підтримкою
    db_path = f'data/db/smpp_dataset_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
    generator = EnhancedSMPPDatasetGenerator(db_path=db_path)
    
    # Генерація тижневого датасету
    logging.info("Початок генерації тижневого SMPP трафіку...")
    
    df = generator.generate_weekly_traffic(
        messages_per_day=5000,  # ~35000 повідомлень за тиждень
        anomaly_rate=0.12,      # 12% аномалій
        save_to_db=True         # Зберігати до бази даних
    )
    
    # Отримання статистики з бази даних
    logging.info("\nОтримання статистики з бази даних...")
    db_stats = generator.get_database_stats()
    if db_stats:
        logging.info(f"Загальна кількість записів у БД: {db_stats['total_messages']}")
        logging.info("Розподіл аномалій у БД:")
        logging.info(f"\n{db_stats['anomaly_stats']}")
    
    # Опціонально: експорт до CSV
    export_csv = input("\nЕкспортувати дані до CSV? (y/n): ").lower() == 'y'
    if export_csv:
        csv_filename = f'data/datasets/smpp_export_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        os.makedirs("data/datasets", exist_ok=True)
        generator.export_to_csv(csv_filename)
        logging.info(f"Дані експортовано до {csv_filename}")
    
    logging.info(f"\nБаза даних збережена: {db_path}")
    logging.info("Генерація завершена!")