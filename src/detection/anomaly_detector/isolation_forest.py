import matplotlib
matplotlib.use('Agg')

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (precision_recall_curve, classification_report, 
                           confusion_matrix, roc_auc_score, f1_score, 
                           precision_score, recall_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sqlite3
import time
import warnings
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')


class IsolationForestTrainer:
    """
    Клас для навчання ансамблю моделей Isolation Forest
    для виявлення аномалій у SMPP повідомленнях
    """
    
    def __init__(self, 
                 input_file: str = 'data/datasets/smpp_weekly_dataset_features_optimized.csv',
                 config_file: str = 'if_config.json',
                 db_path: str = 'data/db/smpp.sqlite',
                 anomaly_ratio: float = 0.05):
        """
        Ініціалізація тренера
        
        Args:
            input_file: Шлях до CSV файлу з даними
            config_file: Файл конфігурації
            db_path: Шлях до бази даних для збереження метаданих
            anomaly_ratio: Очікуваний відсоток аномалій для синтетичних даних
        """
        # Налаштування директорій
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = 'logs/isolation_forest'
        self.plot_dir = 'data/plots/isolation_forest'
        self.models_dir = 'models'
        self.output_dir = 'data/predictions'
        
        # Створення директорій
        for dir_path in [self.log_dir, self.plot_dir, self.models_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Налаштування логування
        self._setup_logging()
        
        # Параметри
        self.input_file = input_file
        self.db_path = db_path
        self.config_file = config_file
        self.anomaly_ratio = anomaly_ratio
        
        # Завантаження конфігурації
        self._load_config()
        
        # Ініціалізація компонентів
        self.scaler = None
        self.models = {}
        self.optimal_threshold = None
        self.feature_importance = None
        self.experiment_results = []
        self.full_dataset = None  # Зберігаємо повний датасет
        
        self.logger.info(f"Ініціалізація IsolationForestTrainer завершена")
        self.logger.info(f"Вхідний файл: {self.input_file}")
        
    def _setup_logging(self):
        """Налаштування системи логування"""
        log_file = os.path.join(self.log_dir, f'training_{self.timestamp}.log')
        
        # Створення форматера
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловий хендлер
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Консольний хендлер
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Налаштування логера
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _load_config(self):
        """Завантаження конфігурації з файлу"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.feature_columns = config.get("feature_columns", [])
            self.base_model_params = config.get("base_model_params", {})
            self.param_grid = config.get("param_grid", {})
            self.logger.info(f"Конфігурацію завантажено з {self.config_file}")
        except FileNotFoundError:
            self.logger.warning(f"Файл конфігурації {self.config_file} не знайдено. Використовуються параметри за замовчуванням")
            self._set_default_config()
        except Exception as e:
            self.logger.error(f"Помилка завантаження конфігурації: {e}")
            self._set_default_config()
            
    def _set_default_config(self):
        """Встановлення параметрів за замовчуванням"""
        # Всі доступні ознаки з датасету
        self.feature_columns = [
            'weekday', 'hour', 'day_of_week', 'message_length', 
            'source_addr_length', 'source_is_numeric', 'dest_is_valid', 
            'message_parts', 'encoding_issues', 'empty_message', 
            'excessive_length', 'sender_frequency', 'recipient_frequency',
            'sender_burst', 'recipient_burst', 'high_sender_frequency',
            'high_recipient_frequency', 'suspicious_word_count', 'url_count',
            'suspicious_url', 'urgency_score', 'financial_patterns',
            'phone_numbers', 'message_entropy', 'obfuscation_score',
            'social_engineering', 'typosquatting', 'night_time', 
            'weekend', 'business_hours', 'source_category_mismatch',
            'time_category_anomaly', 'unusual_sender_pattern',
            'category_time_mismatch', 'sender_legitimacy'
        ]
        
        self.base_model_params = {
            'n_jobs': -1,
            'random_state': 42,
            'warm_start': False
        }
        
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_samples': ['auto', 0.8, 0.9],
            'contamination': [0.01, 0.05, 0.1],
            'max_features': [0.8, 1.0],
            'bootstrap': [False, True]
        }
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Завантаження даних з CSV файлу і фільтрація за is_anomaly = False"""
        self.logger.info("Завантаження даних з файлу...")
        try:
            # Завантаження з різними кодуваннями
            encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.input_file, encoding=encoding)
                    self.logger.info(f"Файл успішно завантажено з кодуванням {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                raise ValueError("Не вдалося завантажити файл з жодним кодуванням")
                
            self.logger.info(f"Завантажено {len(df)} записів з файлу")
            self.logger.info(f"Колонки в датасеті: {list(df.columns)}")
            
            # Зберігаємо повний датасет для подальших передбачень
            self.full_dataset = df.copy()
            
            # Статистика по is_anomaly
            anomaly_stats = df['is_anomaly'].value_counts()
            self.logger.info(f"Розподіл is_anomaly:")
            self.logger.info(f"  False (нормальні): {anomaly_stats.get(False, 0)}")
            self.logger.info(f"  True (аномалії): {anomaly_stats.get(True, 0)}")
            
        except Exception as e:
            self.logger.error(f"Помилка завантаження файлу: {e}")
            raise
        
        # Фільтрація рядків, де is_anomaly = False для навчання
        df_normal = df[df['is_anomaly'] == False].copy()
        self.logger.info(f"Після фільтрації залишилось {len(df_normal)} нормальних записів для навчання")
        
        # Фільтрація категоріальних стовпців (нечислові стовпці), але зберігаємо `is_anomaly`
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        non_numeric_columns.remove('is_anomaly')  # Не видаляти 'is_anomaly'
        df_filtered = df.drop(columns=non_numeric_columns)
        self.logger.info(f"Видалено категоріальні стовпці: {non_numeric_columns}")
        
        # Вибір ознак
        available_features = [col for col in self.feature_columns if col in df_filtered.columns]
        
        # Якщо мало ознак знайдено, використовуємо всі числові
        if len(available_features) < 5:
            self.logger.warning(f"Знайдено лише {len(available_features)} ознак із заданих. Використовуються всі числові ознаки.")
            # Визначаємо всі числові колонки
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            # Виключаємо цільову змінну та ідентифікатори
            exclude_cols = ['is_anomaly', 'id', 'message_id']
            available_features = [col for col in numeric_cols if col not in exclude_cols]
            
            # Якщо є булеві колонки, конвертуємо їх в числові
            bool_cols = df_filtered.select_dtypes(include=['bool']).columns.tolist()
            for col in bool_cols:
                if col not in exclude_cols and col not in available_features:
                    df_filtered[col] = df_filtered[col].astype(int)
                    available_features.append(col)
        
        self.logger.info(f"Використовуються ознаки ({len(available_features)}): {available_features}")
        
        # Підготовка даних для навчання (тільки нормальні)
        X_train = df_normal[available_features].values
        y_train = df_normal['is_anomaly'].values.astype(int)
        
        # Підготовка всіх даних для валідації
        X_all = df_filtered[available_features].values
        y_all = df_filtered['is_anomaly'].values.astype(int)
        
        self.feature_names = available_features
        
        # Обробка пропущених значень
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        
        return df_normal, X_train, y_train, X_all, y_all

        
    def experimental_model_selection(self, X_train: np.ndarray) -> Dict[str, Any]:
        """Експериментальний вибір найкращих параметрів моделі"""
        self.logger.info("=== Експериментальний вибір параметрів моделі ===")
        
        # Різні стратегії масштабування
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        best_score = -np.inf
        best_config = None
        
        for scaler_name, scaler in scalers.items():
            self.logger.info(f"\nТестування з {scaler_name} scaler...")
            X_scaled = scaler.fit_transform(X_train)
            
            # Створюємо валідаційний набір з синтетичними аномаліями
            X_val, y_val = self._create_validation_set(X_scaled)
            
            # Сітковий пошук
            param_combinations = self._get_param_combinations()
            
            for i, params in enumerate(param_combinations):
                self.logger.info(f"Експеримент {i+1}/{len(param_combinations)}: {params}")
                
                # Створюємо модель
                model_params = {**self.base_model_params, **params}
                model = IsolationForest(**model_params)
                
                try:
                    # Навчаємо модель тільки на нормальних даних
                    model.fit(X_scaled)
                    
                    # Оцінюємо на валідаційному наборі
                    val_scores = model.score_samples(X_val)
                    val_pred = model.predict(X_val)
                    
                    # Конвертуємо передбачення (-1 для аномалій, 1 для нормальних)
                    val_pred_binary = (val_pred == -1).astype(int)
                    
                    # Рахуємо метрики
                    if len(np.unique(y_val)) > 1:
                        precision = precision_score(y_val, val_pred_binary, zero_division=0)
                        recall = recall_score(y_val, val_pred_binary, zero_division=0)
                        f1 = f1_score(y_val, val_pred_binary, zero_division=0)
                        auc = roc_auc_score(y_val, -val_scores)
                        
                        # Пріоритет precision > 0.9
                        if precision > 0.9:
                            score = auc
                        else:
                            score = auc * 0.5  # Штраф за низьку precision
                    else:
                        score = 0
                    
                    self.logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
                    
                    # Зберігаємо результати експерименту
                    experiment_result = {
                        'scaler': scaler_name,
                        'params': params,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc,
                        'score': score,
                        'timestamp': datetime.now()
                    }
                    self.experiment_results.append(experiment_result)
                    
                    if score > best_score:
                        best_score = score
                        best_config = {
                            'scaler': scaler_name,
                            'scaler_object': scaler,
                            'params': model_params,
                            'score': score,
                            'metrics': {
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'auc': auc
                            }
                        }
                        
                except Exception as e:
                    self.logger.error(f"  Помилка під час експерименту: {e}")
                    continue
                    
        self.logger.info(f"\nНайкращий результат: {best_config['params']}")
        self.logger.info(f"Метрики: Precision={best_config['metrics']['precision']:.4f}, AUC={best_config['metrics']['auc']:.4f}")
        return best_config
        
    def _create_validation_set(self, X_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Створення валідаційного набору з синтетичними аномаліями"""
        n_normal = len(X_normal)
        n_anomalies = int(n_normal * self.anomaly_ratio)
        
        # Створюємо синтетичні аномалії
        anomalies = []
        for _ in range(n_anomalies):
            # Вибираємо випадковий нормальний зразок
            idx = np.random.randint(0, n_normal)
            anomaly = X_normal[idx].copy()
            
            # Модифікуємо кілька ознак
            n_features_to_modify = np.random.randint(1, max(2, len(anomaly) // 3))
            features_to_modify = np.random.choice(len(anomaly), n_features_to_modify, replace=False)
            
            for feat_idx in features_to_modify:
                # Додаємо шум або екстремальні значення
                modification = np.random.choice(['extreme_high', 'extreme_low', 'noise'])
                if modification == 'extreme_high':
                    anomaly[feat_idx] = anomaly[feat_idx] + np.random.uniform(3, 5) * np.std(X_normal[:, feat_idx])
                elif modification == 'extreme_low':
                    anomaly[feat_idx] = anomaly[feat_idx] - np.random.uniform(3, 5) * np.std(X_normal[:, feat_idx])
                else:
                    anomaly[feat_idx] = np.random.uniform(
                        np.min(X_normal[:, feat_idx]),
                        np.max(X_normal[:, feat_idx])
                    )
                    
            anomalies.append(anomaly)
            
        # Об'єднуємо нормальні та аномальні дані
        X_anomalies = np.array(anomalies)
        X_val = np.vstack([X_normal, X_anomalies])
        y_val = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        
        # Перемішуємо
        indices = np.random.permutation(len(X_val))
        X_val = X_val[indices]
        y_val = y_val[indices]
        
        return X_val, y_val
        
    def _get_param_combinations(self) -> List[Dict]:
        """Генерація комбінацій параметрів для експериментів"""
        from itertools import product
        
        # Створюємо всі комбінації
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = []
        
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
            
        # Обмежуємо кількість експериментів
        if len(combinations) > 50:
            self.logger.info(f"Скорочення кількості експериментів з {len(combinations)} до 50")
            indices = np.linspace(0, len(combinations)-1, 50, dtype=int)
            combinations = [combinations[i] for i in indices]
            
        return combinations
        
    def train_ensemble(self, X_train: np.ndarray, X_all: np.ndarray, 
                      y_all: np.ndarray, best_config: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
        """Навчання ансамблю моделей"""
        self.logger.info("=== Навчання ансамблю моделей ===")
        
        # Масштабування
        self.scaler = best_config['scaler_object']
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_all_scaled = self.scaler.transform(X_all)
        
        # Створення ансамблю з різними random_state
        n_models = 5
        self.models = {}
        all_scores = []
        
        start_time = time.time()
        
        for i in range(n_models):
            self.logger.info(f"Навчання моделі {i+1}/{n_models}...")
            
            # Модифікуємо параметри для різноманітності
            model_params = best_config['params'].copy()
            model_params['random_state'] = 42 + i
            
            # Варіюємо параметри для різноманітності
            if i > 0:
                if 'max_samples' in model_params and model_params['max_samples'] != 'auto':
                    model_params['max_samples'] = min(1.0, model_params['max_samples'] + np.random.uniform(-0.1, 0.1))
                if 'contamination' in model_params:
                    model_params['contamination'] = max(0.001, min(0.5, 
                        model_params['contamination'] + np.random.uniform(-0.02, 0.02)))
                    
            model = IsolationForest(**model_params)
            model.fit(X_train_scaled)  # Навчаємо тільки на нормальних даних
            
            # Оцінка на всіх даних
            scores = model.score_samples(X_all_scaled)
            all_scores.append(scores)
            
            self.models[f'model_{i}'] = model
            
        training_time = time.time() - start_time
        self.logger.info(f"Час навчання ансамблю: {training_time/60:.2f} хвилин")
        
        # Ансамблеве передбачення
        ensemble_scores = np.mean(all_scores, axis=0)
        
        # Оптимізація порогу
        self._optimize_threshold(y_all, ensemble_scores)
        
        # Розрахунок важливості ознак
        self._calculate_feature_importance(X_train_scaled)
        
        return ensemble_scores, X_all_scaled, training_time/60
        
    def _optimize_threshold(self, y_true: np.ndarray, scores: np.ndarray):
        """Оптимізація порогу для максимізації F1-score з урахуванням цілі precision > 0.9"""
        self.logger.info("=== Оптимізація порогу рішення ===")
        
        # Інвертуємо scores (менші значення = аномалії)
        anomaly_scores = -scores
        
        # Розрахунок кривих
        precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Знаходимо пороги з precision > 0.9
        high_precision_idx = np.where(precision[:-1] > 0.9)[0]
        
        if len(high_precision_idx) > 0:
            # Серед них вибираємо з найкращим F1
            best_idx = high_precision_idx[np.argmax(f1_scores[high_precision_idx])]
            self.logger.info("Обрано поріг з precision > 0.9")
        else:
            # Якщо не можемо досягти precision > 0.9, беремо найкращий F1
            best_idx = np.argmax(f1_scores[:-1])
            self.logger.warning("Не вдалося досягти precision > 0.9, обрано поріг з найкращим F1")
            
        self.optimal_threshold = -thresholds[best_idx]
        
        # Логування метрик
        self.logger.info(f"Оптимальний поріг: {self.optimal_threshold:.6f}")
        self.logger.info(f"Precision: {precision[best_idx]:.4f}")
        self.logger.info(f"Recall: {recall[best_idx]:.4f}")
        self.logger.info(f"F1-score: {f1_scores[best_idx]:.4f}")
        
        # Візуалізація
        self._plot_threshold_optimization(precision, recall, thresholds, f1_scores, best_idx)
        
    def _calculate_feature_importance(self, X_train: np.ndarray):
        """Розрахунок важливості ознак через isolation paths"""
        self.logger.info("Розрахунок важливості ознак...")
        
        # Використовуємо першу модель
        model = list(self.models.values())[0]
        
        # Для кожної ознаки рахуємо середню глибину ізоляції
        importance_scores = []
        
        for i in range(X_train.shape[1]):
            # Створюємо дані з тільки однією ознакою
            X_single = np.zeros_like(X_train)
            X_single[:, i] = X_train[:, i]
            
            # Отримуємо scores
            scores = model.score_samples(X_single)
            
            # Важливість = обернена середня глибина
            importance = -np.mean(scores)
            importance_scores.append(importance)
            
        # Нормалізація
        importance_scores = np.array(importance_scores)
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
            
        self.feature_importance = dict(zip(self.feature_names, importance_scores))
        
        # Сортування та логування
        sorted_importance = sorted(self.feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        self.logger.info("Важливість ознак:")
        for feat, imp in sorted_importance:
            self.logger.info(f"  {feat}: {imp:.4f}")
            
        # Візуалізація
        self._plot_feature_importance()
        
    def predict_and_save(self, ensemble_scores: np.ndarray):
        """Створення передбачень для всього датасету і збереження результатів"""
        self.logger.info("=== Створення передбачень для всього датасету ===")
        
        # Створюємо передбачення
        predictions = (ensemble_scores < self.optimal_threshold).astype(int)
        
        # Додаємо передбачення до датасету
        self.full_dataset['predicted_anomaly_if'] = predictions
        self.full_dataset['anomaly_score_if'] = ensemble_scores
        
        # Статистика передбачень
        pred_stats = pd.Series(predictions).value_counts()
        self.logger.info(f"Статистика передбачень:")
        self.logger.info(f"  Нормальні (0): {pred_stats.get(0, 0)}")
        self.logger.info(f"  Аномалії (1): {pred_stats.get(1, 0)}")
        
        # Порівняння з реальними мітками
        if 'is_anomaly' in self.full_dataset.columns:
            comparison = pd.crosstab(
                self.full_dataset['is_anomaly'], 
                self.full_dataset['predicted_anomaly_if'],
                rownames=['Actual'],
                colnames=['Predicted']
            )
            self.logger.info(f"\nПорівняння з реальними мітками:\n{comparison}")
        
        # Збереження файлу
        output_file = os.path.join(
            self.output_dir, 
            f'predictions_isolation_forest_{self.timestamp}.csv'
        )
        
        try:
            self.full_dataset.to_csv(output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"Передбачення збережено в: {output_file}")
            
            # Також зберігаємо тільки аномалії
            anomalies_df = self.full_dataset[self.full_dataset['predicted_anomaly_if'] == 1]
            if len(anomalies_df) > 0:
                anomalies_file = os.path.join(
                    self.output_dir,
                    f'detected_anomalies_if_{self.timestamp}.csv'
                )
                anomalies_df.to_csv(anomalies_file, index=False, encoding='utf-8-sig')
                self.logger.info(f"Виявлені аномалії збережено в: {anomalies_file}")
                
        except Exception as e:
            self.logger.error(f"Помилка збереження передбачень: {e}")
            
    def evaluate_model(self, y_true: np.ndarray, ensemble_scores: np.ndarray) -> Dict[str, Any]:
        """Повна оцінка моделі"""
        self.logger.info("=== Оцінка моделі ===")
        
        # Передбачення
        predictions = (ensemble_scores < self.optimal_threshold).astype(int)
        
        # Метрики
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y_true, -ensemble_scores) if len(np.unique(y_true)) > 1 else 0
        }
        
        # Звіт
        self.logger.info("\nМетрики моделі:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
            
        # Детальний звіт
        if len(np.unique(y_true)) > 1:
            report = classification_report(y_true, predictions, 
                                         target_names=['Normal', 'Anomaly'],
                                         digits=4)
            self.logger.info(f"\nДетальний звіт:\n{report}")
            
            # Матриця плутанини
            cm = confusion_matrix(y_true, predictions)
            self.logger.info(f"\nМатриця плутанини:\n{cm}")
            
        # Візуалізації
        self._create_evaluation_plots(y_true, predictions, ensemble_scores)
        
        return metrics
        
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray):
        """Створення всіх візуалізацій для оцінки"""
        # 1. Матриця плутанини
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/confusion_matrix_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Розподіл scores
        plt.figure(figsize=(12, 6))
        
        if len(np.unique(y_true)) > 1:
            normal_scores = scores[y_true == 0]
            anomaly_scores = scores[y_true == 1]
            
            plt.hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', 
                    density=True, color='blue', edgecolor='black')
            plt.hist(anomaly_scores, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_scores)})', 
                    density=True, color='red', edgecolor='black')
        else:
            plt.hist(scores, bins=50, alpha=0.7, label='All samples', 
                    density=True, color='blue', edgecolor='black')
            
        plt.axvline(x=self.optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold={self.optimal_threshold:.4f}')
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Anomaly Scores', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/score_distribution_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC крива
        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_curve
            
            fpr, tpr, _ = roc_curve(y_true, -scores)
            auc = roc_auc_score(y_true, -scores)
            
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random classifier')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.plot_dir}/roc_curve_{self.timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def _plot_threshold_optimization(self, precision: np.ndarray, recall: np.ndarray, 
                                   thresholds: np.ndarray, f1_scores: np.ndarray, 
                                   optimal_idx: int):
        """Візуалізація процесу оптимізації порогу"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Precision-Recall крива
        ax = axes[0, 0]
        ax.plot(recall, precision, 'b-', linewidth=2, label='PR curve')
        ax.scatter(recall[optimal_idx], precision[optimal_idx], 
                  color='red', s=100, zorder=5, label='Optimal point')
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Target precision=0.9')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. F1-score vs Threshold
        ax = axes[0, 1]
        ax.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
        ax.axvline(x=thresholds[optimal_idx], color='red', linestyle='--', 
                  label=f'Optimal threshold')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('F1-score', fontsize=12)
        ax.set_title('F1-score vs Threshold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Precision vs Threshold
        ax = axes[1, 0]
        ax.plot(thresholds, precision[:-1], 'b-', linewidth=2, label='Precision')
        ax.plot(thresholds, recall[:-1], 'r-', linewidth=2, label='Recall')
        ax.axvline(x=thresholds[optimal_idx], color='green', linestyle='--')
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision & Recall vs Threshold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Таблиця з метриками для різних порогів
        ax = axes[1, 1]
        ax.axis('off')
        
        # Вибираємо кілька порогів для порівняння
        indices = np.linspace(0, len(thresholds)-2, 5, dtype=int)
        
        table_data = []
        for idx in indices:
            table_data.append([
                f"{thresholds[idx]:.4f}",
                f"{precision[idx]:.3f}",
                f"{recall[idx]:.3f}",
                f"{f1_scores[idx]:.3f}"
            ])
            
        table = ax.table(cellText=table_data,
                        colLabels=['Threshold', 'Precision', 'Recall', 'F1-score'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Виділяємо оптимальний рядок
        for i, idx in enumerate(indices):
            if idx == optimal_idx:
                for j in range(4):
                    table[(i+1, j)].set_facecolor('#ffcccc')
                    
        ax.set_title('Metrics for Different Thresholds', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/threshold_optimization_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_feature_importance(self):
        """Візуалізація важливості ознак"""
        if not self.feature_importance:
            return
            
        # Сортуємо за важливістю
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]
        
        plt.figure(figsize=(10, max(8, len(features) * 0.5)))
        y_pos = np.arange(len(features))
        
        bars = plt.barh(y_pos, importance, color='skyblue', edgecolor='navy')
        
        # Додаємо значення на барах
        for i, (feature, imp) in enumerate(zip(features, importance)):
            plt.text(imp + 0.01, i, f'{imp:.3f}', 
                    va='center', fontsize=10)
                    
        plt.yticks(y_pos, features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Feature Importance (Isolation-based)', fontsize=14)
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/feature_importance_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_model_and_metadata(self, metrics: Dict[str, Any], training_time: float):
        """Збереження моделі та метаданих"""
        self.logger.info("=== Збереження моделі ===")
        
        # Підготовка даних для збереження
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metrics': metrics,
            'experiment_results': self.experiment_results,
            'timestamp': self.timestamp,
            'config': {
                'input_file': self.input_file,
                'anomaly_ratio': self.anomaly_ratio,
                'base_model_params': self.base_model_params,
                'param_grid': self.param_grid
            }
        }
        
        # Збереження моделі
        model_filename = f'isolation_forest_ensemble_{self.timestamp}.pkl'
        model_path = os.path.join(self.models_dir, model_filename)
        
        joblib.dump(model_data, model_path)
        self.logger.info(f"Модель збережено: {model_path}")
        
        # Збереження метаданих у БД
        self._save_to_database(model_path, metrics, training_time)
        
        # Збереження результатів експериментів
        self._save_experiment_results()
        
    def _save_to_database(self, model_path: str, metrics: Dict[str, Any], training_time: float):
        """Збереження метаданих у базу даних"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Підготовка даних
            model_name = f"IsolationForest_Ensemble_{self.timestamp}"
            model_type = "IsolationForest"
            version = "2.0"
            
            # Конфігурація моделей як JSON
            config_data = {
                'n_models': len(self.models),
                'base_params': self.base_model_params,
                'optimal_threshold': float(self.optimal_threshold),
                'scaler_type': type(self.scaler).__name__,
                'anomaly_ratio': self.anomaly_ratio,
                'input_file': self.input_file
            }
            config_json = json.dumps(config_data)
            
            # Розмір датасету
            training_dataset_size = len(self.full_dataset) if self.full_dataset is not None else 0
            
            # Вставка даних
            cursor.execute("""
                INSERT INTO models (
                    model_name, model_type, version, file_path, config_json, 
                    feature_names, accuracy, precision_score, recall, f1_score, 
                    roc_auc, training_dataset_size, training_duration_minutes,
                    validation_score, is_active, deployment_status, trained_by, 
                    training_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                model_type,
                version,
                model_path,
                config_json,
                json.dumps(self.feature_names),
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('roc_auc', 0),
                training_dataset_size,
                int(training_time),
                metrics.get('roc_auc', 0),  # validation_score
                1 if metrics.get('precision', 0) > 0.9 else 0,  # is_active
                "Ready" if metrics.get('precision', 0) > 0.9 else "Testing",
                "AI System",
                f"Ensemble of {len(self.models)} IF models trained on {self.input_file}"
            ))
            
            conn.commit()
            self.logger.info("Метадані збережено в БД")
            
        except Exception as e:
            self.logger.error(f"Помилка збереження в БД: {e}")
        finally:
            conn.close()
            
    def _save_experiment_results(self):
        """Збереження результатів експериментів"""
        exp_file = os.path.join(self.log_dir, f'experiments_{self.timestamp}.json')
        
        # Конвертуємо datetime об'єкти
        experiments_to_save = []
        for exp in self.experiment_results:
            exp_copy = exp.copy()
            exp_copy['timestamp'] = exp_copy['timestamp'].isoformat()
            experiments_to_save.append(exp_copy)
            
        with open(exp_file, 'w', encoding='utf-8') as f:
            json.dump(experiments_to_save, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Результати експериментів збережено: {exp_file}")
        
    def generate_report(self, metrics: Dict[str, Any], training_time: float):
        """Генерація фінального звіту"""
        report_file = os.path.join(self.log_dir, f'final_report_{self.timestamp}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ISOLATION FOREST ANOMALY DETECTION - FINAL REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Output predictions: data/predictions/predictions_isolation_forest_{self.timestamp}.csv\n\n")
            
            f.write("DATASET STATISTICS:\n")
            f.write("-" * 30 + "\n")
            if self.full_dataset is not None:
                f.write(f"Total samples: {len(self.full_dataset)}\n")
                if 'is_anomaly' in self.full_dataset.columns:
                    anomaly_stats = self.full_dataset['is_anomaly'].value_counts()
                    f.write(f"Normal samples: {anomaly_stats.get(False, 0)}\n")
                    f.write(f"Anomaly samples: {anomaly_stats.get(True, 0)}\n")
                if 'predicted_anomaly_if' in self.full_dataset.columns:
                    pred_stats = self.full_dataset['predicted_anomaly_if'].value_counts()
                    f.write(f"\nPredicted normal: {pred_stats.get(0, 0)}\n")
                    f.write(f"Predicted anomalies: {pred_stats.get(1, 0)}\n")
            
            f.write("\n\nMODEL PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
                
            f.write(f"\nTarget Achieved: {'YES' if metrics.get('precision', 0) > 0.9 else 'NO'}\n")
            f.write(f"Training time: {training_time:.2f} minutes\n")
            
            f.write("\n\nFEATURE IMPORTANCE:\n")
            f.write("-" * 30 + "\n")
            if self.feature_importance:
                for feature, importance in sorted(self.feature_importance.items(), 
                                                key=lambda x: x[1], reverse=True):
                    f.write(f"{feature}: {importance:.4f}\n")
                    
            f.write("\n\nMODEL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of models in ensemble: {len(self.models)}\n")
            f.write(f"Optimal threshold: {self.optimal_threshold:.6f}\n")
            f.write(f"Scaler type: {type(self.scaler).__name__}\n")
            
            f.write("\n\nEXPERIMENTAL RESULTS SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total experiments conducted: {len(self.experiment_results)}\n")
            if self.experiment_results:
                # Найкращий за AUC
                best_auc = max(self.experiment_results, key=lambda x: x.get('auc', 0))
                f.write(f"\nBest AUC: {best_auc.get('auc', 0):.4f}\n")
                f.write(f"Parameters: {best_auc['params']}\n")
                
                # Найкращий за Precision
                best_precision = max(self.experiment_results, key=lambda x: x.get('precision', 0))
                f.write(f"\nBest Precision: {best_precision.get('precision', 0):.4f}\n")
                f.write(f"Parameters: {best_precision['params']}\n")
                
        self.logger.info(f"Фінальний звіт збережено: {report_file}")
        
    def run_training_pipeline(self):
        """Основний пайплайн навчання"""
        self.logger.info("=" * 60)
        self.logger.info("ПОЧАТОК НАВЧАННЯ ISOLATION FOREST")
        self.logger.info("=" * 60)
        
        try:
            # 1. Завантаження даних
            df_normal, X_train, y_train, X_all, y_all = self.load_and_prepare_data()
            
            # 2. Експериментальний вибір параметрів
            best_config = self.experimental_model_selection(X_train)
            
            # 3. Навчання ансамблю
            ensemble_scores, X_all_scaled, training_time = self.train_ensemble(
                X_train, X_all, y_all, best_config
            )
            
            # 4. Створення та збереження передбачень
            self.predict_and_save(ensemble_scores)
            
            # 5. Оцінка моделі
            metrics = self.evaluate_model(y_all, ensemble_scores)
            
            # 6. Збереження моделі
            self.save_model_and_metadata(metrics, training_time)
            
            # 7. Генерація звіту
            self.generate_report(metrics, training_time)
            
            self.logger.info("=" * 60)
            self.logger.info("НАВЧАННЯ ЗАВЕРШЕНО УСПІШНО")
            self.logger.info(f"Передбачення збережено в: data/predictions/predictions_isolation_forest_{self.timestamp}.csv")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Критична помилка: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    # Запуск навчання
    trainer = IsolationForestTrainer(
        input_file='data/datasets/smpp_weekly_dataset_features_optimized.csv',
        config_file='if_config.json',
        db_path='data/db/smpp.sqlite',
        anomaly_ratio=0.05
    )
    
    success = trainer.run_training_pipeline()