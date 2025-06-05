import pandas as pd
import numpy as np
import re
import json
import logging
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os

class SMSCategoryClassifier:
    """
    Класифікатор SMS повідомлень за категоріями
    Навчається на розмічених нормальних даних і класифікує нові повідомлення
    """
    
    def __init__(self, config_path="src/detection/classification/classification_config.json"):
        """Ініціалізація класифікатора"""
        
        # Завантаження конфігурації
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Категорії та їх ключові слова
        self.category_keywords = self.config.get("category_keywords", {})
        self.source_addresses = self.config.get("source_addresses", {})
        self.extended_keywords = self.config.get("extended_keywords", {})
        
        # Моделі
        self.models = {
            'tfidf_nb': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                         stop_words=None, lowercase=True)),
                ('classifier', MultinomialNB(alpha=0.1))
            ]),
            
            'tfidf_lr': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                         stop_words=None, lowercase=True)),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            
            'tfidf_rf': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                         stop_words=None, lowercase=True)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        }
        
        self.best_model = None
        self.feature_extractor = None
        self.label_encoder = None
        
        # Налаштування логування
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, text: str) -> str:
        """Попередня обробка тексту"""
        if not isinstance(text, str):
            return ""
        
        # Приведення до нижнього регістру
        text = text.lower()
        
        # Видалення зайвих пробілів
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Заміна цифрових послідовностей на токен
        text = re.sub(r'\d{4,}', 'NUMBERS', text)
        
        # Заміна сум грошей на токен
        text = re.sub(r'\d+\s*(?:грн|₴|\$|usd|eur)', 'MONEY', text)
        
        # Заміна номерів телефонів на токен
        text = re.sub(r'\+?3?8?0[0-9]{9}', 'PHONE', text)
        
        # Заміна URL на токен
        text = re.sub(r'https?://[^\s]+|www\.[^\s]+', 'URL', text)
        
        return text
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Екстракція додаткових текстових ознак"""
        features_df = df.copy()
        
        # Довжинні ознаки
        features_df['text_length'] = df['message_text'].apply(len)
        features_df['word_count'] = df['message_text'].apply(lambda x: len(str(x).split()))
        features_df['avg_word_length'] = df['message_text'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
        )
        
        # Символьні ознаки
        features_df['digit_ratio'] = df['message_text'].apply(
            lambda x: sum(c.isdigit() for c in str(x)) / max(len(str(x)), 1)
        )
        features_df['upper_ratio'] = df['message_text'].apply(
            lambda x: sum(c.isupper() for c in str(x)) / max(len(str(x)), 1)
        )
        features_df['punct_count'] = df['message_text'].apply(
            lambda x: sum(1 for c in str(x) if c in '.,!?:;-()[]{}')
        )
        
        # Спеціальні патерни
        features_df['has_money'] = df['message_text'].apply(
            lambda x: 1 if re.search(r'\d+\s*(?:грн|₴|\$|usd|eur)', str(x).lower()) else 0
        )
        features_df['has_phone'] = df['message_text'].apply(
            lambda x: 1 if re.search(r'\+?3?8?0[0-9]{9}', str(x)) else 0
        )
        features_df['has_url'] = df['message_text'].apply(
            lambda x: 1 if re.search(r'https?://[^\s]+|www\.[^\s]+', str(x).lower()) else 0
        )
        features_df['has_numbers'] = df['message_text'].apply(
            lambda x: 1 if re.search(r'\d{4,}', str(x)) else 0
        )
        
        return features_df
    
    def extract_source_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Екстракція ознак джерела повідомлення"""
        features_df = df.copy()
        
        # Довжина адреси відправника
        features_df['source_length'] = df['source_addr'].apply(len)
        
        # Тип адреси (цифровий/текстовий)
        features_df['source_is_numeric'] = df['source_addr'].apply(lambda x: 1 if str(x).isdigit() else 0)
        
        # Відповідність джерела категорії
        features_df['source_category_match'] = df.apply(
            lambda row: self._calculate_source_match(row['source_addr'], row['category']), axis=1
        )
        
        return features_df
    
    def _calculate_source_match(self, source_addr: str, category: str) -> float:
        """Розрахунок відповідності джерела категорії"""
        if not source_addr or category not in self.source_addresses:
            return 0.0
        
        expected_sources = self.source_addresses[category]
        source_upper = str(source_addr).upper()
        
        # Точне співпадіння
        if source_upper in [s.upper() for s in expected_sources]:
            return 1.0
        
        # Часткове співпадіння
        for expected in expected_sources:
            if expected.upper() in source_upper or source_upper in expected.upper():
                return 0.7
        
        return 0.0
    
    def rule_based_prediction(self, text: str, source_addr: str = "") -> str:
        """Класифікація на основі правил (як fallback)"""
        text_lower = str(text).lower()
        source_upper = str(source_addr).upper()
        
        # Спочатку перевіряємо джерело
        for category, sources in self.source_addresses.items():
            for source in sources:
                if source.upper() in source_upper:
                    return category
        
        # Потім перевіряємо ключові слова
        category_scores = {}
        
        for category, keywords in self.extended_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            category_scores[category] = score
        
        # Повертаємо категорію з найвищим скором
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return "marketing"  # За замовчуванням
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Підготовка даних для навчання"""
        # Фільтруємо тільки нормальні повідомлення
        normal_df = df[~df['is_anomaly']].copy()
        
        # Попередня обробка тексту
        normal_df['processed_text'] = normal_df['message_text'].apply(self.preprocess_text)
        
        # Екстракція додаткових ознак
        normal_df = self.extract_text_features(normal_df)
        normal_df = self.extract_source_features(normal_df)
        
        # Підготовка цільової змінної
        y = normal_df['category']
        
        self.logger.info(f"Підготовлено {len(normal_df)} зразків для навчання")
        self.logger.info(f"Розподіл по категоріях:\n{y.value_counts()}")
        
        return normal_df, y
    
    def train_models(self, df: pd.DataFrame) -> Dict:
        """Навчання всіх моделей"""
        self.logger.info("Початок навчання моделей...")
        
        # Підготовка даних
        X_df, y = self.prepare_training_data(df)
        
        # Основний текст для TF-IDF
        X_text = X_df['processed_text']
        
        # Розділення на навчальну та тестову вибірки
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Навчання кожної моделі
        for model_name, model in self.models.items():
            self.logger.info(f"Навчання моделі: {model_name}")
            
            # Навчання
            model.fit(X_train, y_train)
            
            # Крос-валідація
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Оцінка на тестовій вибірці
            test_score = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            
            results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            self.logger.info(f"{model_name} - CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}, Test: {test_score:.3f}")
        
        # Вибір найкращої моделі
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_score'])
        self.best_model = self.models[best_model_name]
        
        self.logger.info(f"Найкраща модель: {best_model_name}")
        
        # Повторне навчання найкращої моделі на всіх даних
        self.best_model.fit(X_text, y)
        
        return results
    
    def predict_category(self, text: str, source_addr: str = "") -> Tuple[str, float]:
        """Передбачення категорії для одного повідомлення"""
        if self.best_model is None:
            # Якщо модель не навчена, використовуємо правила
            return self.rule_based_prediction(text, source_addr), 0.5
        
        # Попередня обробка
        processed_text = self.preprocess_text(text)
        
        # Передбачення
        prediction = self.best_model.predict([processed_text])[0]
        
        # Ймовірності (якщо модель підтримує)
        try:
            probabilities = self.best_model.predict_proba([processed_text])[0]
            confidence = probabilities.max()
        except:
            confidence = 0.8  # За замовчуванням
        
        return prediction, confidence
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Передбачення категорій для датафрейму"""
        result_df = df.copy()
        
        if self.best_model is None:
            # Використовуємо правила
            result_df['predicted_category'] = df.apply(
                lambda row: self.rule_based_prediction(row['message_text'], row.get('source_addr', '')), 
                axis=1
            )
            result_df['prediction_confidence'] = 0.5
        else:
            # Використовуємо навчену модель
            processed_texts = df['message_text'].apply(self.preprocess_text)
            
            predictions = self.best_model.predict(processed_texts)
            
            try:
                probabilities = self.best_model.predict_proba(processed_texts)
                confidences = probabilities.max(axis=1)
            except:
                confidences = [0.8] * len(predictions)
            
            result_df['predicted_category'] = predictions
            result_df['prediction_confidence'] = confidences
        
        return result_df
    
    def evaluate_on_anomalies(self, df: pd.DataFrame) -> Dict:
        """Оцінка роботи класифікатора на аномаліях"""
        anomaly_df = df[df['is_anomaly']].copy()
        
        if len(anomaly_df) == 0:
            return {"message": "Немає аномалій для оцінки"}
        
        # Передбачення для аномалій
        predictions_df = self.predict_batch(anomaly_df)
        
        # Аналіз розподілу категорій в аномаліях
        predicted_distribution = predictions_df['predicted_category'].value_counts()
        confidence_stats = predictions_df['prediction_confidence'].describe()
        
        return {
            'total_anomalies': len(anomaly_df),
            'predicted_distribution': predicted_distribution.to_dict(),
            'confidence_stats': confidence_stats.to_dict(),
            'low_confidence_count': sum(predictions_df['prediction_confidence'] < 0.6)
        }
    
    def save_model(self, filepath: str):
        """Збереження навченої моделі"""
        model_data = {
            'best_model': self.best_model,
            'config': self.config,
            'extended_keywords': self.extended_keywords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Модель збережено в {filepath}")
    
    def load_model(self, filepath: str):
        """Завантаження навченої моделі"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.config = model_data['config']
        self.extended_keywords = model_data['extended_keywords']
        
        self.logger.info(f"Модель завантажено з {filepath}")
    
    def visualize_results(self, results: Dict, df: pd.DataFrame):
        """Візуалізація результатів"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Порівняння моделей
        ax = axes[0, 0]
        model_names = list(results.keys())
        test_scores = [results[name]['test_score'] for name in model_names]
        cv_scores = [results[name]['cv_mean'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        ax.bar(x - width / 2, cv_scores, width, label='Cross-validation', alpha=0.8)
        ax.bar(x + width / 2, test_scores, width, label='Test score', alpha=0.8)
        ax.set_xlabel('Моделі')
        ax.set_ylabel('Accuracy')
        ax.set_title('Порівняння моделей')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()

        # 2. Розподіл категорій
        ax = axes[0, 1]
        normal_data = df[~df['is_anomaly']]
        category_counts = normal_data['category'].value_counts()
        category_counts.plot(kind='bar', ax=ax, color='lightblue')
        ax.set_title('Розподіл категорій (нормальні дані)')
        ax.set_xlabel('Категорія')
        ax.set_ylabel('Кількість')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Довжина повідомлень по категоріях
        ax = axes[1, 0]
        category_lengths = {}
        for category in normal_data['category'].unique()[:5]:  # Топ-5 категорій
            lengths = normal_data[normal_data['category'] == category]['message_text'].apply(len)
            category_lengths[category] = lengths

        ax.boxplot(category_lengths.values(), labels=category_lengths.keys())
        ax.set_title('Розподіл довжини повідомлень')
        ax.set_xlabel('Категорія')
        ax.set_ylabel('Довжина символів')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. Матриця плутанини для найкращої моделі
        ax = axes[1, 1]
        if self.best_model and len(normal_data) > 100:
            test_data = normal_data.sample(n=min(200, len(normal_data)))
            processed_texts = test_data['message_text'].apply(self.preprocess_text)
            y_true = test_data['category']
            y_pred = self.best_model.predict(processed_texts)

            top_categories = y_true.value_counts().head(6).index
            mask = y_true.isin(top_categories)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]

            cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_categories)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=top_categories, yticklabels=top_categories, ax=ax)
            ax.set_title('Матриця плутанини (топ-6 категорій)')
            ax.set_xlabel('Передбачена категорія')
            ax.set_ylabel('Справжня категорія')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

        plt.tight_layout()

        os.makedirs('data/plots/classification', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/plots/classification/sms_classifier_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Графіки збережено в {filename}")



# Приклад використання та демонстрація
if __name__ == "__main__":
    
    def demonstrate_classifier():
        """Демонстрація роботи класифікатора"""
        
        # Ініціалізація
        classifier = SMSCategoryClassifier()
        
        df = pd.read_csv('data/datasets/smpp_weekly_dataset_20250604_223745.csv', encoding='utf-8-sig')
        print(f"Завантажено {len(df)} записів")
            
        # Навчання моделей
        print("\n=== НАВЧАННЯ МОДЕЛЕЙ ===")
        results = classifier.train_models(df)
            
        # Виведення результатів
        print("\n=== РЕЗУЛЬТАТИ НАВЧАННЯ ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Cross-validation: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
            print(f"  Test accuracy: {metrics['test_score']:.3f}")
            
        # Оцінка на аномаліях
        print("\n=== ОЦІНКА НА АНОМАЛІЯХ ===")
        anomaly_results = classifier.evaluate_on_anomalies(df)
        print(f"Всього аномалій: {anomaly_results.get('total_anomalies', 0)}")
        print("Розподіл передбачених категорій:")
        for category, count in anomaly_results.get('predicted_distribution', {}).items():
            print(f"  {category}: {count}")
            
        print(f"Повідомлень з низькою впевненістю (<0.6): {anomaly_results.get('low_confidence_count', 0)}")
            
        # Візуалізація
        classifier.visualize_results(results, df)
            
        # Збереження моделі
        classifier.save_model('models/sms_classifier.pkl')
            
        # Тестування на окремих прикладах
        print("\n=== ТЕСТУВАННЯ НА ПРИКЛАДАХ ===")
        test_messages = [
            ("Ваш баланс: 1500 грн. Останній платіж: 250 грн в АТБ", "PRIVAT24"),
            ("Код підтвердження: 123456. Не повідомляйте його нікому", "AUTH"),
            ("Ваше замовлення №12345 готове до видачі у відділенні", "NOVAPOSHTA"),
            ("ТЕРМІННОВО! Ваш рахунок буде заблоковано!", "BANK-FAKE"),
            ("Запис до лікаря Іванова на завтра о 14:00", "CLINIC")
        ]
            
        for message, source in test_messages:
            category, confidence = classifier.predict_category(message, source)
            print(f"Повідомлення: {message[:50]}...")
            print(f"Джерело: {source}")
            print(f"Передбачена категорія: {category} (впевненість: {confidence:.2f})")
            print()
            
        
    # Запуск демонстрації
    demonstrate_classifier()