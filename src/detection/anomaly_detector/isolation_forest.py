import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score
import logging
import os
from datetime import datetime
import warnings
import joblib
import json

warnings.filterwarnings('ignore')

class IsolationForestTrainer:
    def __init__(self, input_file='datasets/smpp_dataset_features.csv', config_file='if_config.json'):
        # Створення папок
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = 'logs/isolation_forest'
        self.plot_dir = 'plots'
        
        # Завантаження конфігурації
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.feature_columns = config["feature_columns"]
        self.model_configs = config["model_configs"]
        self.log_dir = config["paths"]["log_dir"]
        self.plot_dir = config["paths"]["plot_dir"]
        self.models_dir = config["paths"]["models_dir"]

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Налаштування логування
        log_file = os.path.join(self.log_dir, f'training_{self.timestamp}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Ініціалізація IsolationForestTrainer")
        self.logger.info(f"Вхідний файл: {input_file}")
        
        self.input_file = input_file
        self.scaler = StandardScaler()
        self.models = {}
        self.optimal_threshold = None
        
    def load_and_prepare_data(self):
        """Завантаження та підготовка даних"""
        self.logger.info("Завантаження даних...")
        
        try:
            df = pd.read_csv(self.input_file, encoding='utf-8-sig')
            self.logger.info(f"Завантажено {len(df)} записів")
        except Exception as e:
            self.logger.error(f"Помилка завантаження файлу: {e}")
            raise
        
        # Використання ознак з if_config.json
        feature_columns = self.feature_columns
        
        # Фільтруємо тільки існуючі колонки
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = set(feature_columns) - set(available_features)
        
        if missing_features:
            self.logger.warning(f"Відсутні ознаки: {missing_features}")
        
        self.logger.info(f"Використовується {len(available_features)} ознак")
        
        X = df[available_features].values
        y = df['is_anomaly'].values
        
        # Збереження метаданих
        self.feature_names = available_features
        self.df = df
        
        return X, y
    def train_models(self, X, y):
        """Навчання ансамблю моделей Isolation Forest (рівне усереднення без фільтрації)"""
        self.logger.info("Початок навчання моделей...")

        X_normal = X[y == 0]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.logger.info(f"Розмір тренувального набору: {len(X_train)}")
        self.logger.info(f"Розмір валідаційного набору: {len(X_val)}")
        self.logger.info(f"Кількість нормальних зразків для навчання: {len(X_normal)}")

        # Нормалізація
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        X_val_scaled = self.scaler.transform(X_val)

        configs = self.model_configs
        self.logger.info(f"Конфігурації моделей: {configs}")

        val_scores_all = []

        for i, config in enumerate(configs):
            self.logger.info(f"Навчання моделі {i+1}: {config}")

            model = IsolationForest(random_state=42, **config)
            model.fit(X_normal_scaled)

            model_name = f'model_{i}'
            self.models[model_name] = model

            val_scores = model.score_samples(X_val_scaled)
            val_scores_all.append(val_scores)

        # Рівне усереднення
        val_scores_all = np.array(val_scores_all)
        ensemble_scores = np.mean(val_scores_all, axis=0)

        self._optimize_threshold(y_val, ensemble_scores)

        return X_val, y_val, ensemble_scores


    
    def _optimize_threshold(self, y_true, scores):
        """Оптимізація порогу для максимізації F1-score"""
        self.logger.info("Оптимізація порогу...")
        
        precision, recall, thresholds = precision_recall_curve(y_true, -scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        optimal_idx = np.argmax(f1_scores[:-1])
        self.optimal_threshold = -thresholds[optimal_idx]
        
        self.logger.info(f"Оптимальний поріг: {self.optimal_threshold:.4f}")
        self.logger.info(f"F1-score: {f1_scores[optimal_idx]:.3f}")
        self.logger.info(f"Precision: {precision[optimal_idx]:.3f}")
        self.logger.info(f"Recall: {recall[optimal_idx]:.3f}")
        # Візуалізація
        self._plot_threshold_optimization(precision, recall, thresholds, f1_scores, optimal_idx)
    
    def evaluate(self, X, y):
        """Оцінка моделі на всіх даних"""
        self.logger.info("Оцінка моделі...")
        
        X_scaled = self.scaler.transform(X)
        
        # Отримання ансамблевих scores
        all_scores = []
        for model_name, model in self.models.items():
            scores = model.score_samples(X_scaled)
            all_scores.append(scores)
        
        ensemble_scores = np.mean(all_scores, axis=0)
        
        # Передбачення
        predictions = (ensemble_scores < self.optimal_threshold).astype(int)
        
        # Метрики
        report = classification_report(y, predictions, target_names=['Normal', 'Anomaly'])
        self.logger.info(f"\nЗвіт класифікації:\n{report}")
        
        # Матриця плутанини
        cm = confusion_matrix(y, predictions)
        self.logger.info(f"\nМатриця плутанини:\n{cm}")
        
        # Візуалізації
        self._create_visualizations(y, predictions, ensemble_scores)
        
        return predictions, ensemble_scores
    
    def _create_visualizations(self, y_true, y_pred, scores):
        """Створення всіх візуалізацій"""
        # 1. Матриця плутанини
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/confusion_matrix_{self.timestamp}.png', dpi=300)
        plt.close()
        
        # 2. Розподіл scores
        plt.figure(figsize=(10, 6))
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', density=True)
        plt.axvline(x=self.optimal_threshold, color='red', linestyle='--', 
                   label=f'Threshold={self.optimal_threshold:.4f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/score_distribution_{self.timestamp}.png', dpi=300)
        plt.close()
        
        # 3. Аналіз помилок по категоріях
        if 'category' in self.df.columns:
            plt.figure(figsize=(12, 6))
            self.df['predicted'] = y_pred
            self.df['error'] = (y_true != y_pred).astype(int)
            
            error_by_category = self.df.groupby('category')['error'].mean()
            error_by_category.plot(kind='bar')
            plt.title('Error Rate by Category')
            plt.xlabel('Category')
            plt.ylabel('Error Rate')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{self.plot_dir}/errors_by_category_{self.timestamp}.png', dpi=300)
            plt.close()
        
        self.logger.info(f"Візуалізації збережено в папці '{self.plot_dir}'")
    
    def _plot_threshold_optimization(self, precision, recall, thresholds, f1_scores, optimal_idx):
        """Візуалізація оптимізації порогу"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Precision-Recall крива
        ax1.plot(recall, precision, 'b-', linewidth=2)
        ax1.scatter(recall[optimal_idx], precision[optimal_idx], 
                   color='red', s=100, zorder=5)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.grid(True, alpha=0.3)
        
        # F1-score vs threshold
        ax2.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
        ax2.axvline(x=thresholds[optimal_idx], color='red', linestyle='--')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('F1-score')
        ax2.set_title('F1-score vs Threshold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/threshold_optimization_{self.timestamp}.png', dpi=300)
        plt.close()
    
    def save_models(self):
        """Збереження навчених моделей"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': self.feature_names,
            'timestamp': self.timestamp
        }
        
        model_path = f'models/isolation_forest_ensemble_{self.timestamp}.pkl'
        joblib.dump(model_data, model_path)
        self.logger.info(f"Моделі збережено в '{model_path}'")
    
    def run_training_pipeline(self):
        """Повний пайплайн навчання"""
        self.logger.info("=== Початок навчання Isolation Forest ===")
        
        try:
            # Завантаження даних
            X, y = self.load_and_prepare_data()
            
            # Навчання
            X_val, y_val, val_scores = self.train_models(X, y)
            
            # Оцінка на всіх даних
            predictions, scores = self.evaluate(X, y)
            
            # Збереження моделей
            self.save_models()
            
            # Збереження результатів
            results_df = self.df.copy()
            results_df['anomaly_score'] = scores
            results_df['predicted_anomaly'] = predictions
            results_file = f'datasets/isolation_forest_results_{self.timestamp}.csv'
            results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"Результати збережено в '{results_file}'")
            
            self.logger.info("=== Навчання завершено успішно ===")
            
        except Exception as e:
            self.logger.error(f"Помилка під час навчання: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Запуск навчання
    trainer = IsolationForestTrainer(input_file='datasets/smpp_dataset_features.csv')
    trainer.run_training_pipeline()