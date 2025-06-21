# Dockerfile для SMPP Anomaly Detection System
FROM python:3.11-slim

# Встановлення системних залежностей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsqlite3-dev \
    libffi-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Створення робочої директорії
WORKDIR /app

# Копіювання requirements файлу з папки docker
COPY docker/requirements.txt .

# Встановлення Python залежностей
RUN pip install --no-cache-dir -r requirements.txt

# Копіювання проєкту
COPY . .

# Створення необхідних директорій
RUN mkdir -p data/db data/datasets data/plots logs models

# Встановлення змінних середовища
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Порт для Streamlit dashboard
EXPOSE 8501

# Команда за замовчуванням
CMD ["streamlit", "run", "src/dashboard/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]