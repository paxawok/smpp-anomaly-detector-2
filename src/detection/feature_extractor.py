import pandas as pd
import numpy as np
import re
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
import warnings
from detection.obfuscation.obfuscation_detector import ConfusablesObfuscationDetector

warnings.filterwarnings('ignore')

os.makedirs("logs", exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/generator_features_{timestamp}.log"

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

class SMPPFeatureExtractor:
    def __init__(self, config_path="data/data_config.json"):
        """Ініціалізація з конфігурацією"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.shorteners = self.config.get("shorteners", [])
        self.source_addresses = self.config.get("source_addresses", {})
        self.suspicious_words = self.config.get("suspicious_words", {})
        
        # Для частотного аналізу
        self.sender_counter = defaultdict(list)
        self.recipient_counter = defaultdict(list)
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Екстракція всіх 32 ключових ознак"""
        logging.info("Екстракція 32 ключових ознак...")
        
        # Очищення від NaN
        df = self._clean_data(df)
        
        # Підготовка часових даних
        df['submit_time'] = pd.to_datetime(df['submit_time'])
        df['hour'] = df['submit_time'].dt.hour
        df['day_of_week'] = df['submit_time'].dt.dayofweek
        
        # Підготовка текстових даних
        df['message_text'] = df['message_text'].fillna('').astype(str)
        df['source_addr'] = df['source_addr'].fillna('').astype(str)
        df['dest_addr'] = df['dest_addr'].fillna('').astype(str)
        df['category'] = df['category'].fillna('unknown').astype(str)
        
        # Рахуємо частотні аномалії
        self._calculate_frequency_anomalies(df)
        
        # 32 ключові ознаки
        features = {}
        
        # ===== ПРОТОКОЛЬНІ АНОМАЛІЇ (8 ознак) =====
        features['message_length'] = df['message_text'].apply(len)
        features['source_addr_length'] = df['source_addr'].apply(len)
        features['source_is_numeric'] = df['source_addr'].apply(lambda x: 1 if x.isdigit() else 0)
        features['dest_is_valid'] = df['dest_addr'].apply(self._validate_phone_number)
        features['message_parts'] = features['message_length'].apply(lambda x: max(1, (x + 139) // 140))
        features['encoding_issues'] = df['message_text'].apply(self._detect_encoding_issues)
        features['empty_message'] = (features['message_length'] == 0).astype(int)
        features['excessive_length'] = (features['message_length'] > 1000).astype(int)
        
        # ===== ЧАСТОТНІ АНОМАЛІЇ (6 ознак) =====
        features['sender_frequency'] = df['source_addr'].map(df['source_addr'].value_counts())
        features['recipient_frequency'] = df['dest_addr'].map(df['dest_addr'].value_counts())
        features['sender_burst'] = df.apply(lambda x: self._calculate_sender_burst(x['source_addr'], x['submit_time']), axis=1)
        features['recipient_burst'] = df.apply(lambda x: self._calculate_recipient_burst(x['dest_addr'], x['submit_time']), axis=1)
        features['high_sender_frequency'] = (features['sender_frequency'] > 30).astype(int)
        features['high_recipient_frequency'] = (features['recipient_frequency'] > 15).astype(int)
        
        # ===== СЕМАНТИЧНІ АНОМАЛІЇ (10 ознак) =====
        features['suspicious_word_count'] = df['message_text'].apply(self._count_suspicious_words)
        features['url_count'] = df['message_text'].apply(self._count_urls)
        features['suspicious_url'] = df['message_text'].apply(self._has_suspicious_url)
        features['urgency_score'] = df['message_text'].apply(self._calculate_urgency_score)
        features['financial_patterns'] = df['message_text'].apply(self._count_financial_patterns)
        features['phone_numbers'] = df['message_text'].apply(self._count_phone_numbers)
        features['message_entropy'] = df['message_text'].apply(self._calculate_entropy)
        
        # Покращена обфускація з confusable-homoglyphs
        if self.has_advanced_obfuscation and self.obfuscation_detector:
            # Використовуємо покращений детектор для повного аналізу
            obfuscation_results = df.apply(
                lambda row: self.obfuscation_detector.analyze_text_and_sender(
                    str(row['message_text']), str(row['source_addr'])
                ), axis=1
            )
            # Основна ознака - комбінований score
            features['obfuscation_score'] = obfuscation_results.apply(lambda x: x['combined_obfuscation'])
            
            logging.info("✓ Використано покращену обфускацію з confusable-homoglyphs")
        else:
            # Fallback до простого методу
            features['obfuscation_score'] = df['message_text'].apply(self._calculate_obfuscation)
            
            logging.warning("✗ Використано базову обфускацію. Встановіть confusable-homoglyphs для кращої точності")
        
        features['social_engineering'] = df['message_text'].apply(self._detect_social_engineering)
        features['typosquatting'] = df.apply(lambda x: self._detect_typosquatting(x['source_addr'], x['category']), axis=1)
        
        # ===== ПОВЕДІНКОВІ АНОМАЛІЇ (8 ознак) =====
        features['night_time'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        features['weekend'] = (df['day_of_week'] >= 5).astype(int)
        features['business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        features['source_category_mismatch'] = df.apply(lambda x: self._check_source_category_mismatch(x['source_addr'], x['category']), axis=1)
        features['time_category_anomaly'] = df.apply(lambda x: self._check_time_category_anomaly(x['category'], x['hour']), axis=1)
        features['unusual_sender_pattern'] = df['source_addr'].apply(self._detect_unusual_sender_pattern)
        features['category_time_mismatch'] = df.apply(lambda x: self._check_category_time_mismatch(x['category'], x['hour'], x['day_of_week']), axis=1)
        features['sender_legitimacy'] = df.apply(lambda x: self._calculate_sender_legitimacy(x['source_addr'], x['category']), axis=1)
        
        # Створення результуючого DataFrame
        result_df = df.copy()
        for feature_name, feature_values in features.items():
            result_df[feature_name] = feature_values
        
        # Очищення від NaN в ознаках
        feature_columns = list(features.keys())
        for col in feature_columns:
            result_df[col] = result_df[col].fillna(0)
            # Конвертуємо в числові типи
            if result_df[col].dtype == 'object':
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
        
        logging.info(f"Екстраговано {len(feature_columns)} ознак")
        return result_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищення даних від NaN"""
        df = df.copy()
        
        # Заповнення пустих значень
        string_columns = ['source_addr', 'dest_addr', 'message_text', 'category']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Заповнення часових значень
        if 'submit_time' in df.columns:
            df['submit_time'] = pd.to_datetime(df['submit_time'], errors='coerce')
            df['submit_time'] = df['submit_time'].fillna(datetime.now())
        
        return df
    
    def _calculate_frequency_anomalies(self, df: pd.DataFrame):
        """Попередній розрахунок частотних аномалій"""
        df_sorted = df.sort_values('submit_time')
        
        for idx, row in df_sorted.iterrows():
            source = row['source_addr']
            dest = row['dest_addr']
            time = row['submit_time']
            
            self.sender_counter[source].append(time)
            self.recipient_counter[dest].append(time)
    
    def _validate_phone_number(self, phone: str) -> int:
        """Валідація номера телефону"""
        if not phone:
            return 0
        # Український номер: 380XXXXXXXXX
        pattern = r'^380[0-9]{9}$'
        return 1 if re.match(pattern, phone) else 0
    
    def _detect_encoding_issues(self, text: str) -> int:
        """Виявлення проблем з кодуванням"""
        if not text:
            return 0
        
        # Пошук незвичайних символів
        weird_chars = ['�', '◊', '☺', '♦', '♣', '♠', '♥']
        return 1 if any(char in text for char in weird_chars) else 0
    
    def _calculate_sender_burst(self, source_addr: str, current_time: datetime) -> int:
        """Розрахунок burst для відправника (>30 повідомлень за 3 хвилини)"""
        if not source_addr or source_addr not in self.sender_counter:
            return 0
        
        times = self.sender_counter[source_addr]
        window_start = current_time - pd.Timedelta(minutes=3)
        
        count = sum(1 for t in times if window_start <= t <= current_time)
        return 1 if count > 30 else 0
    
    def _calculate_recipient_burst(self, dest_addr: str, current_time: datetime) -> int:
        """Розрахунок burst для отримувача (>15 повідомлень за 3 хвилини)"""
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
        text_lower = text.lower()
        
        for lang_words in self.suspicious_words.values():
            for word in lang_words:
                if word in text_lower:
                    count += 1
        
        return min(count, 10)  # Максимум 10
    
    def _count_urls(self, text: str) -> int:
        """Підрахунок URL"""
        if not text:
            return 0
        
        pattern = r'https?://[^\s]+|www\.[^\s]+|[^\s]+\.[a-z]{2,}'
        return len(re.findall(pattern, text.lower()))
    
    def _has_suspicious_url(self, text: str) -> int:
        """Перевірка підозрілих URL"""
        if not text:
            return 0
        
        text_lower = text.lower()
        return 1 if any(shortener in text_lower for shortener in self.shorteners) else 0
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Розрахунок поспішання"""
        if not text:
            return 0.0
        
        urgency_words = ['термінов', 'негайно', 'зараз', 'швидко', 'останн', 'увага']
        score = 0
        text_lower = text.lower()
        
        for word in urgency_words:
            if word in text_lower:
                score += 1
        
        # Додаткові бали за великі літери
        if sum(c.isupper() for c in text) > len(text) * 0.3:
            score += 1
        
        # Додаткові бали за знаки оклику
        score += min(text.count('!'), 3)
        
        return min(score / 5.0, 1.0)  # Нормалізуємо до [0, 1]
    
    def _count_financial_patterns(self, text: str) -> int:
        """Підрахунок фінансових патернів"""
        if not text:
            return 0
        
        patterns = [
            r'\d+\s*(?:грн|₴|\$|USD|EUR)',  # Валюти
            r'рахунок\s*\d+',  # Номери рахунків
            r'карт[аи]\s*\d+',  # Номери карт
            r'баланс',  # Баланс
            r'переказ'  # Переказ
        ]
        
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text.lower()))
        
        return min(count, 5)
    
    def _count_phone_numbers(self, text: str) -> int:
        """Підрахунок номерів телефонів"""
        if not text:
            return 0
        
        pattern = r'\+?3?8?0[0-9]{9}'
        return len(re.findall(pattern, text))
    
    def _calculate_entropy(self, text: str) -> float:
        """Розрахунок ентропії Шеннона"""
        if not text:
            return 0.0
        
        # Підрахунок частоти символів
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        # Розрахунок ентропії
        text_len = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            p = count / text_len
            if p > 0:
                entropy -= p * np.log2(p)
        
        return min(entropy / 5.0, 1.0)  # Нормалізуємо
        
    def _detect_social_engineering(self, text: str) -> int:
        """Виявлення соціальної інженерії"""
        if not text:
            return 0
        
        patterns = [
            'виграли',
            'безкоштов',
            'подзвон',
            'перекин',
            'заблоков',
            'попередж',
            'втрат',
            'призупин'
        ]
        
        text_lower = text.lower()
        return 1 if any(pattern in text_lower for pattern in patterns) else 0
    
    def _detect_typosquatting(self, source_addr: str, category: str) -> float:
        """Виявлення typosquatting"""
        if not source_addr or not category:
            return 0.0
        
        legit_sources = self.source_addresses.get(category, [])
        if not legit_sources:
            return 0.0
        
        source_lower = source_addr.lower()
        
        for legit in legit_sources:
            legit_lower = legit.lower()
            
            # Точне співпадіння
            if source_lower == legit_lower:
                return 0.0
            
            # Перевірка схожості (Levenshtein distance спрощено)
            if len(source_lower) == len(legit_lower):
                diff_count = sum(a != b for a, b in zip(source_lower, legit_lower))
                if 1 <= diff_count <= 2:  # 1-2 відмінності
                    return 1.0
        
        return 0.0
    
    def _check_source_category_mismatch(self, source_addr: str, category: str) -> int:
        """Перевірка невідповідності джерела категорії"""
        if not source_addr or not category:
            return 0
        
        expected_sources = self.source_addresses.get(category, [])
        if not expected_sources:
            return 0
        
        source_upper = source_addr.upper()
        
        # Точне або часткове співпадіння
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
        
        # Змішані літери/цифри в короткому коді
        if len(source_addr) <= 6:
            has_letters = any(c.isalpha() for c in source_addr)
            has_digits = any(c.isdigit() for c in source_addr)
            if has_letters and has_digits:
                return 1
        
        # Повторювані символи
        if len(set(source_addr)) < len(source_addr) / 2:
            return 1
        
        return 0
    
    def _check_category_time_mismatch(self, category: str, hour: int, day_of_week: int) -> int:
        """Перевірка невідповідності категорії часу"""
        # Критичні категорії, які не повинні працювати вночі/у вихідні
        critical_categories = ['banking', 'government']
        
        if category not in critical_categories:
            return 0
        
        # Нічний час (0-6)
        if 0 <= hour <= 6:
            return 1
        
        # Вихідні для держустанов
        if category == 'government' and day_of_week >= 5:  # Субота/неділя
            return 1
        
        return 0
    
    def _calculate_sender_legitimacy(self, source_addr: str, category: str) -> float:
        """Розрахунок легітимності відправника"""
        if not source_addr or not category:
            return 0.0
        
        legit_sources = self.source_addresses.get(category, [])
        if not legit_sources:
            return 0.5  # Нейтральна оцінка для невідомих категорій
        
        source_upper = source_addr.upper()
        
        # Точне співпадіння
        if source_upper in [s.upper() for s in legit_sources]:
            return 1.0
        
        # Часткове співпадіння
        for legit in legit_sources:
            if legit.upper() in source_upper:
                return 0.7
        
        return 0.0
    
    def get_feature_names(self) -> List[str]:
        """Отримання списку імен ознак (рівно 32)"""
        return [
            # Протокольні аномалії (8)
            'message_length', 'source_addr_length', 'source_is_numeric', 'dest_is_valid',
            'message_parts', 'encoding_issues', 'empty_message', 'excessive_length',
            
            # Частотні аномалії (6)
            'sender_frequency', 'recipient_frequency', 'sender_burst', 'recipient_burst',
            'high_sender_frequency', 'high_recipient_frequency',
            
            # Семантичні аномалії (10)
            'suspicious_word_count', 'url_count', 'suspicious_url', 'urgency_score',
            'financial_patterns', 'phone_numbers', 'message_entropy', 'obfuscation_score',
            'social_engineering', 'typosquatting',
            
            # Поведінкові аномалії (8)
            'night_time', 'weekend', 'business_hours', 'source_category_mismatch',
            'time_category_anomaly', 'unusual_sender_pattern', 'category_time_mismatch',
            'sender_legitimacy'
        ]
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Статистичний аналіз датасету"""
        logging.info("Проведення статистичного аналізу...")
        
        analysis = {
            'total_messages': len(df),
            'normal_messages': len(df[~df['is_anomaly']]),
            'anomaly_messages': len(df[df['is_anomaly']]),
            'anomaly_rate': df['is_anomaly'].mean()
        }
        
        return analysis
    
    def visualize_analysis(self, df: pd.DataFrame):
        """Візуалізація аналізу"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Розподіл аномалій по типах
        ax = axes[0, 0]
        if 'anomaly_type' in df.columns:
            anomaly_counts = df[df['is_anomaly']]['anomaly_type'].value_counts()
            anomaly_counts.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Розподіл аномалій по типах')
            ax.set_xlabel('Тип аномалії')
            ax.set_ylabel('Кількість')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Часовий розподіл
        ax = axes[0, 1]
        hourly_normal = df[~df['is_anomaly']]['hour'].value_counts().sort_index()
        hourly_anomalies = df[df['is_anomaly']]['hour'].value_counts().sort_index()
        
        x = range(24)
        ax.bar(x, hourly_normal.reindex(x, fill_value=0), alpha=0.7, label='Нормальні')
        ax.bar(x, hourly_anomalies.reindex(x, fill_value=0), alpha=0.7, label='Аномалії')
        ax.set_xlabel('Година')
        ax.set_ylabel('Кількість')
        ax.set_title('Розподіл по годинах')
        ax.legend()
        
        # 3. Топ аномальних ознак
        ax = axes[1, 0]
        feature_names = self.get_feature_names()
        if all(f in df.columns for f in feature_names[:5]):
            feature_corr = df[feature_names[:5]].corrwith(df['is_anomaly']).abs()
            feature_corr.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Кореляція ознак з аномаліями')
            ax.set_xlabel('Ознака')
            ax.set_ylabel('Кореляція')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Розподіл довжини повідомлень
        ax = axes[1, 1]
        normal_lengths = df[~df['is_anomaly']]['message_length']
        anomaly_lengths = df[df['is_anomaly']]['message_length']
        
        ax.hist(normal_lengths, bins=50, alpha=0.7, label='Нормальні', density=True)
        ax.hist(anomaly_lengths, bins=50, alpha=0.7, label='Аномалії', density=True)
        ax.set_xlabel('Довжина повідомлення')
        ax.set_ylabel('Щільність')
        ax.set_title('Розподіл довжини повідомлень')
        ax.legend()
        
        plt.tight_layout()
        
        # Створення директорії та збереження
        os.makedirs('plot', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'plot/visual_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Графік збережено у {filename}")
        plt.close()
    
    def process_dataset(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Повна обробка датасету"""
        logging.info(f"Завантаження датасету з {input_file}...")
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        
        # Екстракція ознак
        df_processed = self.extract_features(df)
        
        # Аналіз
        analysis_results = self.analyze_dataset(df_processed)
        logging.info(f"Аналіз завершено:")
        logging.info(f"  - Всього повідомлень: {analysis_results['total_messages']}")
        logging.info(f"  - Нормальних: {analysis_results['normal_messages']}")
        logging.info(f"  - Аномальних: {analysis_results['anomaly_messages']}")
        logging.info(f"  - Частка аномалій: {analysis_results['anomaly_rate']:.2%}")
        
        # Візуалізація
        self.visualize_analysis(df_processed)
        
        # Збереження
        if output_file:
            df_processed.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"Оброблений датасет збережено в {output_file}")
        
        # Виведення інформації про ознаки
        feature_names = self.get_feature_names()
        logging.info(f"\nВсього ознак: {len(feature_names)}")
        logging.info("\nОзнаки для навчання моделі:")
        for i, feat in enumerate(feature_names, 1):
            logging.info(f"  {i:2d}. {feat}")
        
        return df_processed


if __name__ == "__main__":
    # Використання
    extractor = SMPPFeatureExtractor()
    
    # Перевірка доступності покращеної обфускації
    if extractor.has_advanced_obfuscation:
        print("✓ Покращена обфускація активна")
        
        # Тест нової обфускації
        test_cases = [
            ("Ваш баланс: 1500 грн", "Нормальне повідомлення"),
            ("Ваш bаланс: 1500 грн", "Обфускація: b замість б"),
            ("Рrivаt24 повідомлення", "Обфускація: P, r, а"),
            ("iPhone доступний", "Нормальне змішування"),
            ("www.рrivаt24.com", "Обфускований домен")
        ]
        
        print("\nТест покращеної обфускації:")
        print("-" * 60)
        for text, description in test_cases:
            score = extractor.obfuscation_detector.calculate_obfuscation_score(text)
            print(f"{description:<25}: {score:.3f} | '{text}'")
    else:
        print("✗ Покращена обфускація недоступна")
        print("Встановіть: pip install confusable-homoglyphs")
    
    # Обробка датасету (якщо файл існує)
    try:
        df_processed = extractor.process_dataset(
            input_file='datasets/smpp_weekly_dataset_20250604_223745.csv',
            output_file='datasets/smpp_weekly_dataset_features_optimized.csv'
        )
        
        # Виведення статистики по ознаках
        feature_names = extractor.get_feature_names()
        print(f"\nГотово! Екстраговано {len(feature_names)} ознак:")
        
        print("\n=== ПРОТОКОЛЬНІ АНОМАЛІЇ (8) ===")
        for feat in feature_names[:8]:
            print(f"  • {feat}")
        
        print("\n=== ЧАСТОТНІ АНОМАЛІЇ (6) ===")
        for feat in feature_names[8:14]:
            print(f"  • {feat}")
        
        print("\n=== СЕМАНТИЧНІ АНОМАЛІЇ (10) ===")
        for feat in feature_names[14:24]:
            print(f"  • {feat}")
        
        print("\n=== ПОВЕДІНКОВІ АНОМАЛІЇ (8) ===")
        for feat in feature_names[24:32]:
            print(f"  • {feat}")
        
        # Статистика обфускації
        if 'obfuscation_score' in df_processed.columns:
            obf_stats = df_processed['obfuscation_score'].describe()
            print(f"\n=== СТАТИСТИКА ОБФУСКАЦІЇ ===")
            print(f"Середній рівень: {obf_stats['mean']:.3f}")
            print(f"Максимальний: {obf_stats['max']:.3f}")
            print(f"Повідомлень з обфускацією >0.5: {(df_processed['obfuscation_score'] > 0.5).sum()}")
            print(f"Повідомлень з обфускацією >0.8: {(df_processed['obfuscation_score'] > 0.8).sum()}")
        
    except FileNotFoundError:
        print(f"\nУвага: Файл датасету не знайдено.")
        print("Створіть датасет спочатку за допомогою EnhancedSMPPDatasetGenerator")