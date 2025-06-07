import pandas as pd
import numpy as np
import re
import json
import logging
import pickle
import sqlite3
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Екстрактор текстових ознак"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_str = str(text)
            feature_dict = {
                'length': len(text_str),
                'word_count': len(text_str.split()),
                'digit_ratio': sum(c.isdigit() for c in text_str) / max(len(text_str), 1),
                'upper_ratio': sum(c.isupper() for c in text_str) / max(len(text_str), 1),
                'has_money': 1 if re.search(r'\d+\s*(?:грн|₴|\$|uah|usd|eur)', text_str.lower()) else 0,
                'has_phone': 1 if re.search(r'\+?3?8?0?\d{9,}', text_str) else 0,
                'has_url': 1 if re.search(r'https?://|www\.', text_str.lower()) else 0,
                'has_code': 1 if re.search(r'(?:код|code|pin|otp).*\d{4,}', text_str.lower()) else 0,
                'exclamation_count': text_str.count('!'),
                'question_count': text_str.count('?'),
                'avg_word_length': np.mean([len(w) for w in text_str.split()]) if text_str.split() else 0
            }
            features.append(list(feature_dict.values()))
        return np.array(features)

class SourceFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, source_addresses):
        self.source_addresses = source_addresses
        self.source_encoder = LabelEncoder()

    def fit(self, X, y=None):
        # створюємо нормалізовану копію
        self.normalized_addresses = {
            k: [s.upper() for s in v] for k, v in self.source_addresses.items()
        }

        unique_sources = set(str(s).upper() for s in X if pd.notna(s))
        unique_sources.add('OTHER')
        self.source_encoder.fit(sorted(unique_sources))
        return self

    def transform(self, X):
        features = []

        for source in X:
            source_str = str(source).upper()

            feature_dict = {
                'source_length': len(source_str),
                'is_numeric': int(source_str.replace('+', '').isdigit()),
                'is_short': int(len(source_str) <= 6),
                'has_digits': int(any(c.isdigit() for c in source_str)),
                'known_source': int(self._is_known_source(source_str)),
            }

            encoded_source = self._encode_source(source_str)
            features.append(list(feature_dict.values()) + [encoded_source])

        return np.array(features)

    def _is_known_source(self, source):
        return any(source in sources for sources in self.source_addresses.values())

    def _encode_source(self, source):
        # Перевірка класів
        if source not in self.source_encoder.classes_:
            source = 'OTHER'
        return self.source_encoder.transform([source])[0]

class TextSelector(BaseEstimator, TransformerMixin):
    """Селектор для вибору конкретної колонки з DataFrame"""
    
    def __init__(self, field):
        self.field = field
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.field]
    
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Екстрактор часових ознак"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for timestamp in X:
            if isinstance(timestamp, str):
                dt = pd.to_datetime(timestamp)
            else:
                dt = timestamp
                
            feature_dict = {
                'hour': dt.hour,
                'day_of_week': dt.dayofweek,
                'is_weekend': 1 if dt.dayofweek >= 5 else 0,
                'is_night': 1 if dt.hour >= 22 or dt.hour < 6 else 0,
                'is_business_hours': 1 if 9 <= dt.hour <= 18 and dt.dayofweek < 5 else 0,
                'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
                'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
                'day_sin': np.sin(2 * np.pi * dt.dayofweek / 7),
                'day_cos': np.cos(2 * np.pi * dt.dayofweek / 7)
            }
            features.append(list(feature_dict.values()))
        return np.array(features)

class OptimizedSMSClassifier:
    """Оптимізований класифікатор SMS повідомлень"""
    
    def __init__(self, config_path="src/detection/classification/classification_config.json"):
        """Ініціалізація класифікатора"""
        
        # Завантаження конфігурації
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.source_addresses = self.config.get("source_addresses", {})
        self.extended_keywords = self.config.get("extended_keywords", {})
        
        # Ініціалізація компонентів
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.training_time = None
        self.model_metrics = {}
        
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, text: str) -> str:
        """Попередня обробка тексту"""
        if not isinstance(text, str):
            return ""
        
        # Приведення до нижнього регістру
        text = text.lower()
        
        # Очищення
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Нормалізація чисел
        text = re.sub(r'\b\d{4,}\b', 'LONGNUM', text)
        text = re.sub(r'\d+\s*(?:грн|₴|uah)', 'MONEY', text)
        text = re.sub(r'\+?3?8?0?\d{9,}', 'PHONE', text)
        text = re.sub(r'https?://[^\s]+', 'URL', text)
        text = re.sub(r'www\.[^\s]+', 'URL', text)
        text = re.sub(r'\b\d{4,6}\b', 'CODE', text)
        
        return text
    
    def create_feature_pipeline(self):
        """Створення пайплайну для екстракції ознак"""
        
        # TF-IDF для тексту
        tfidf = Pipeline([
            ('selector', TextSelector('processed_text')),
            ('vectorizer', TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 3),
                min_df=3,
                max_df=0.95,
                lowercase=True,
                analyzer='word',
                token_pattern=r'\b\w+\b'
            ))
        ])
        
        # Комбінований екстрактор ознак
        feature_union = FeatureUnion([
            ('tfidf', tfidf),
            ('text_features', Pipeline([
                ('selector', TextSelector('message_text')),
                ('extractor', TextFeatureExtractor())
            ])),
            ('source_features', Pipeline([
                ('selector', TextSelector('source_addr')),
                ('extractor', SourceFeatureExtractor(self.source_addresses))
            ])),
            ('time_features', Pipeline([
                ('selector', TextSelector('submit_time')),
                ('extractor', TimeFeatureExtractor())
            ]))
        ])
        
        return feature_union
    
    def create_models(self):
        """Створення моделей з оптимізованими гіперпараметрами"""
        
        feature_pipeline = self.create_feature_pipeline()
        
        models = {
            'logistic_regression': {
                'pipeline': Pipeline([
                    ('features', feature_pipeline),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('classifier', LogisticRegression(random_state=42))
                ]),
                'param_grid': {
                    'features__tfidf__vectorizer__max_features': [1000, 3000],
                    'classifier__C': [0.01, 0.1, 1, 10],
                    'classifier__penalty': ['l2'],
                    'classifier__max_iter': [1000],
                    'classifier__class_weight': ['balanced']
                }
            },
            'svm': {
                'pipeline': Pipeline([
                    ('features', self.create_feature_pipeline()),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('classifier', LinearSVC(random_state=42))
                ]),
                'param_grid': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__loss': ['hinge', 'squared_hinge'],
                    'classifier__max_iter': [1000]
                }
            }
        }
        
        return models
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Підготовка даних для навчання"""
        
        # Фільтруємо тільки нормальні повідомлення
        normal_df = df[~df['is_anomaly']].copy()
        
        # Попередня обробка тексту
        normal_df['processed_text'] = normal_df['message_text'].apply(self.preprocess_text)
        
        # Видалення рідкісних категорій
        category_counts = normal_df['category'].value_counts()
        valid_categories = category_counts[category_counts >= 10].index
        normal_df = normal_df[normal_df['category'].isin(valid_categories)]
        
        self.logger.info(f"Підготовлено {len(normal_df)} зразків")
        self.logger.info(f"Категорії: {list(valid_categories)}")
        
        return normal_df
    
    def train_and_optimize(self, df: pd.DataFrame) -> Dict:
        """Навчання та оптимізація моделей"""
        
        start_time = time.time()
        self.logger.info("Початок навчання моделей...")
        
        # Підготовка даних
        prepared_df = self.prepare_data(df)
        
        # Кодування цільової змінної
        y = self.label_encoder.fit_transform(prepared_df['category'])
        
        # Розділення на навчальну та тестову вибірки
        X_train, X_test, y_train, y_test = train_test_split(
            prepared_df, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = self.create_models()
        best_score = 0
        best_model_name = None
        results = {}
        
        for model_name, model_config in models.items():
            self.logger.info(f"\nНавчання моделі: {model_name}")
            
            # Grid Search для оптимізації гіперпараметрів
            grid_search = GridSearchCV(
                model_config['pipeline'],
                model_config['param_grid'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # Навчання
            grid_search.fit(X_train, y_train)
            
            # Оцінка
            y_pred = grid_search.predict(X_test)
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # ROC AUC для мультикласової класифікації
            try:
                y_pred_proba = grid_search.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = None
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'best_params': grid_search.best_params_,
                'classification_report': classification_report(
                    y_test, y_pred, 
                    target_names=self.label_encoder.classes_
                )
            }
            
            self.logger.info(f"{model_name} - F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
            
            # Оновлення найкращої моделі
            if f1 > best_score:
                best_score = f1
                best_model_name = model_name
                self.best_model = grid_search
                self.model_metrics = results[model_name]
        
        self.training_time = (time.time() - start_time) / 60  # в хвилинах
        self.logger.info(f"\nНайкраща модель: {best_model_name} (F1: {best_score:.3f})")
        self.logger.info(f"Час навчання: {self.training_time:.2f} хвилин")
        
        return results
    
    def predict(self, text: str, source_addr: str, submit_time: str) -> Tuple[str, float]:
        """Передбачення категорії для одного повідомлення"""
        
        if self.best_model is None:
            raise ValueError("Модель не навчена")
        
        # Створення DataFrame для передбачення
        data = pd.DataFrame([{
            'message_text': text,
            'source_addr': source_addr,
            'submit_time': submit_time,
            'processed_text': self.preprocess_text(text)
        }])
        
        # Передбачення
        prediction_encoded = self.best_model.predict(data)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Ймовірність
        try:
            probabilities = self.best_model.predict_proba(data)[0]
            confidence = probabilities.max()
        except:
            confidence = 0.9
        
        return prediction, confidence
    
    def save_model(self, base_path: str = "models") -> str:
        """Збереження моделі з таймштампом"""
        
        os.makedirs(base_path, exist_ok=True)
        
        # Генерація імені файлу з таймштампом
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"sms_classifier_{timestamp}.pkl"
        filepath = os.path.join(base_path, filename)
        
        # Збереження моделі
        model_data = {
            'best_model': self.best_model,
            'label_encoder': self.label_encoder,
            'source_addresses': self.source_addresses,
            'model_metrics': self.model_metrics,
            'training_time': self.training_time,
            'timestamp': timestamp
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Модель збережено: {filepath}")
        return filepath
    
    def save_to_database(self, filepath: str, db_path: str, training_size: int):
        """Збереження інформації про модель в БД"""
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Підготовка даних для вставки
        model_data = {
            'model_name': 'SMS Category Classifier',
            'model_type': 'Multiclass Classification',
            'version': datetime.now().strftime("%Y%m%d_%H%M"),
            'file_path': filepath,
            'config_json': json.dumps(self.config),
            'feature_names': json.dumps([
                'tfidf', 'text_features', 'source_features', 'time_features'
            ]),
            'accuracy': self.model_metrics.get('accuracy'),
            'precision_score': self.model_metrics.get('precision'),
            'recall': self.model_metrics.get('recall'),
            'f1_score': self.model_metrics.get('f1_score'),
            'roc_auc': self.model_metrics.get('roc_auc'),
            'training_dataset_size': training_size,
            'training_duration_minutes': int(self.training_time),
            'validation_score': self.model_metrics.get('f1_score'),
            'is_active': 1,
            'deployment_status': 'ready',
            'trained_by': 'OptimizedSMSClassifier',
            'training_notes': f"Best params: {json.dumps(self.best_model.best_params_ if hasattr(self.best_model, 'best_params_') else {})}"
        }
        
        # SQL запит
        columns = ', '.join(model_data.keys())
        placeholders = ', '.join(['?' for _ in model_data])
        query = f"INSERT INTO models ({columns}) VALUES ({placeholders})"
        
        cursor.execute(query, list(model_data.values()))
        conn.commit()
        conn.close()
        
        self.logger.info("Інформація про модель збережена в БД")
    
    def visualize_results(self, results: Dict, df: pd.DataFrame):
        """Візуалізація результатів навчання"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Порівняння моделей
        ax = axes[0, 0]
        metrics_df = pd.DataFrame({
            model: {
                'F1-Score': metrics['f1_score'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            }
            for model, metrics in results.items()
        }).T
        
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title('Порівняння моделей', fontsize=14)
        ax.set_ylabel('Значення метрики')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Розподіл категорій
        ax = axes[0, 1]
        normal_df = df[~df['is_anomaly']].copy()  # Створюємо копію, щоб уникнути попереджень
        category_dist = normal_df['category'].value_counts().head(10)
        category_dist.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Топ-10 категорій', fontsize=14)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Розподіл по годинах (виправлений блок)
        ax = axes[1, 0]
        try:
            # Автовизначення формату дати
            normal_df['hour'] = pd.to_datetime(
                normal_df['submit_time'],
                format='mixed'  # Автоматичне визначення формату
            ).dt.hour
        
            hour_dist = normal_df.groupby(['hour', 'category']).size().unstack(fill_value=0)
            top_categories = normal_df['category'].value_counts().head(5).index
            hour_dist[top_categories].plot(ax=ax, kind='line', marker='o')
            ax.set_title('Розподіл по годинах (топ-5 категорій)', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        except Exception as e:
            ax.text(0.5, 0.5, f'Помилка обробки дат:\n{str(e)}', 
                ha='center', va='center')
            self.logger.error(f"Помилка при обробці дат: {e}")
            
        # 4. Матриця важливості ознак (якщо доступна)
        ax = axes[1, 1]
        if hasattr(self.best_model, 'best_estimator_'):
            # Спробуємо отримати важливість ознак
            try:
                classifier = self.best_model.best_estimator_.named_steps['classifier']
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_[:20]  # Топ-20
                    indices = np.argsort(importances)[::-1]
                    ax.bar(range(len(importances)), importances[indices])
                    ax.set_title('Топ-20 важливих ознак', fontsize=14)
                    ax.set_xlabel('Індекс ознаки')
                    ax.set_ylabel('Важливість')
                else:
                    ax.text(0.5, 0.5, 'Важливість ознак недоступна\nдля цієї моделі', 
                           ha='center', va='center', transform=ax.transAxes)
            except:
                ax.text(0.5, 0.5, 'Не вдалося отримати\nважливість ознак', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Збереження
        os.makedirs('data/plots/classification', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'data/plots/classification/optimized_classifier_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Візуалізація збережена: {filename}")


def main():
    """Основна функція для навчання та збереження класифікатора"""
    
    # Параметри
    csv_path = 'data/datasets/smpp_weekly_dataset_20250607_053008.csv'
    db_path = 'data/db/smpp.sqlite'
    
    # Завантаження даних
    print(f"Завантаження даних з {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-16')
    print(f"Завантажено {len(df)} записів")
    
    # Ініціалізація класифікатора
    classifier = OptimizedSMSClassifier()
    
    # Навчання та оптимізація
    print("\nНавчання моделей...")
    results = classifier.train_and_optimize(df)
    
    # Виведення результатів
    print("\n=== РЕЗУЛЬТАТИ НАВЧАННЯ ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Збереження моделі
    filepath = classifier.save_model()
    
    # Збереження в БД
    training_size = len(df[~df['is_anomaly']])
    classifier.save_to_database(filepath, db_path, training_size)
    
    # Візуалізація
    classifier.visualize_results(results, df)
    
    # Тестування
    print("\n=== ТЕСТУВАННЯ НА ПРИКЛАДАХ ===")
    test_cases = [
        {
            'text': "Ваш баланс: 1500 грн. Останній платіж: 250 грн",
            'source': "PRIVAT24",
            'time': "2024-01-15 14:30:00"
        },
        {
            'text': "Код підтвердження: 123456",
            'source': "Google",
            'time': "2024-01-15 10:00:00"
        },
        {
            'text': "Ваше замовлення готове до отримання",
            'source': "NOVAPOSHTA",
            'time': "2024-01-15 18:00:00"
        }
    ]
    
    for test in test_cases:
        category, confidence = classifier.predict(
            test['text'], test['source'], test['time']
        )
        print(f"\nТекст: {test['text']}")
        print(f"Джерело: {test['source']}")
        print(f"Категорія: {category} (впевненість: {confidence:.2%})")
    
    print("\n✅ Класифікатор успішно навчений та збережений!")


if __name__ == "__main__":
    main()