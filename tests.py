import pandas as pd
import numpy as np
import joblib
import pickle
import re
from datetime import datetime
from sklearn.metrics import classification_report, f1_score

# 🧠 Імпорти кастомних трансформерів (щоб deserialization спрацював)
from src.detection.classification.classification import TextSelector, TextFeatureExtractor, SourceFeatureExtractor, TimeFeatureExtractor

# 1. Завантаження збереженого словника
with open('models/sms_classifier_20250607_0605.pkl', 'rb') as f:
    model_bundle = pickle.load(f)

pipeline = model_bundle['best_model']  # <-- головна модель
# Якщо потрібно, можеш звернутися до: model_bundle['label_encoder'], model_bundle['training_time'], тощо

# 2. Завантаження тестових даних
df = pd.read_csv('data/datasets/smpp_weekly_dataset_20250607_033017.csv', encoding='utf-16')
df = df[df['is_anomaly'] == False].copy()

# 3. Попередня обробка тексту (така ж, як при навчанні)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

df['processed_text'] = df['message_text'].apply(preprocess_text)

predicted = pipeline.predict(df)

# Подивися, що саме повертає модель
print(predicted[:10])  # Числа чи рядки?

# Якщо числа — декодуй
if isinstance(predicted[0], (int, float, np.integer)):
    label_encoder = model_bundle['label_encoder']
    predicted_labels = label_encoder.inverse_transform(predicted)
else:
    predicted_labels = predicted


# 5. Збереження результату
df['predicted_category'] = predicted_labels
df.to_csv('tests/predicted_sms.csv', index=False, encoding='utf-8-sig')
print("Готово! Результати збережено у 'tests/predicted_sms.csv'")
