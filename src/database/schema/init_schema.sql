-- Active: 1749215574633@@127.0.0.1@3306
-- -*- mode: sql; sql-product: sqlite; -*-
-- 1. ТАБЛИЦЯ ЗАХОПЛЕНИХ PDU (тільки сирі дані протоколу)
CREATE TABLE IF NOT EXISTS captured_pdus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    src_ip TEXT,
    dst_ip TEXT,
    src_port INTEGER,
    dst_port INTEGER,
    command_id INTEGER NOT NULL,
    command_name TEXT NOT NULL,
    command_status INTEGER,
    sequence_number INTEGER,
    command_length INTEGER,
    raw_data TEXT,
    raw_header BLOB,
    raw_body BLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. ТАБЛИЦЯ SMPP ПОВІДОМЛЕНЬ (оброблені дані + всі ознаки)
CREATE TABLE IF NOT EXISTS smpp_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pdu_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    source_addr TEXT NOT NULL,
    source_addr_ton INTEGER DEFAULT 0,
    source_addr_npi INTEGER DEFAULT 0,
    dest_addr TEXT NOT NULL,
    dest_addr_ton INTEGER DEFAULT 0,
    dest_addr_npi INTEGER DEFAULT 0,
    message_text TEXT,
    message_length INTEGER,
    data_coding INTEGER DEFAULT 0,
    esm_class INTEGER DEFAULT 0,
    priority_flag INTEGER DEFAULT 0,
    category TEXT,
    message_parts INTEGER,
    encoding_issues BOOLEAN DEFAULT 0,
    excessive_length BOOLEAN DEFAULT 0,
    dest_is_valid BOOLEAN DEFAULT 0,
    source_addr_length INTEGER,
    source_is_numeric BOOLEAN DEFAULT 0,
    empty_message BOOLEAN DEFAULT 0,
    protocol_anomaly_score REAL DEFAULT 0,
    sender_frequency INTEGER DEFAULT 0,
    recipient_frequency INTEGER DEFAULT 0,
    sender_burst BOOLEAN DEFAULT 0,
    recipient_burst BOOLEAN DEFAULT 0,
    high_sender_frequency BOOLEAN DEFAULT 0,
    high_recipient_frequency BOOLEAN DEFAULT 0,
    suspicious_word_count INTEGER DEFAULT 0,
    url_count INTEGER DEFAULT 0,
    suspicious_url BOOLEAN DEFAULT 0,
    urgency_score REAL DEFAULT 0,
    financial_patterns INTEGER DEFAULT 0,
    phone_numbers INTEGER DEFAULT 0,
    message_entropy REAL DEFAULT 0,
    obfuscation_score REAL DEFAULT 0,
    social_engineering BOOLEAN DEFAULT 0,
    typosquatting REAL DEFAULT 0,
    night_time BOOLEAN DEFAULT 0,
    weekend BOOLEAN DEFAULT 0,
    business_hours BOOLEAN DEFAULT 0,
    source_category_mismatch BOOLEAN DEFAULT 0,
    time_category_anomaly BOOLEAN DEFAULT 0,
    unusual_sender_pattern BOOLEAN DEFAULT 0,
    category_time_mismatch BOOLEAN DEFAULT 0,
    sender_legitimacy REAL DEFAULT 0,
    hour INTEGER,
    day_of_week INTEGER,
    features_extracted BOOLEAN DEFAULT 0,
    feature_extraction_version TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pdu_id) REFERENCES captured_pdus(id)
);

-- 3. ТАБЛИЦЯ РЕЗУЛЬТАТІВ АНАЛІЗУ ML (тільки результати моделей)
CREATE TABLE IF NOT EXISTS anomaly_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    final_anomaly_score REAL,
    is_anomaly BOOLEAN DEFAULT 0,
    risk_level TEXT,
    confidence_level REAL,
    isolation_forest_score REAL,
    isolation_forest_anomaly BOOLEAN DEFAULT 0,
    isolation_forest_version TEXT,
    autoencoder_score REAL,
    autoencoder_anomaly BOOLEAN DEFAULT 0,
    autoencoder_reconstruction_error REAL,
    autoencoder_version TEXT,
    ensemble_method TEXT,
    ensemble_weights TEXT,
    model_version TEXT,
    processing_time_ms INTEGER,
    feature_vector_used TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES smpp_messages(id)
);

-- 4. ТАБЛИЦЯ АЛЕРТІВ (тільки критичні випадки для операторів)
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    source_addr TEXT NOT NULL,
    dest_addr TEXT NOT NULL,
    message_preview TEXT,
    anomaly_score REAL NOT NULL,
    risk_level TEXT NOT NULL,
    category TEXT,
    status TEXT DEFAULT 'new',
    priority INTEGER DEFAULT 3,
    assigned_to TEXT,
    handled_by TEXT,
    handled_at DATETIME,
    resolution_time_minutes INTEGER,
    notes TEXT,
    escalated BOOLEAN DEFAULT 0,
    escalated_at DATETIME,
    related_alerts TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id) REFERENCES anomaly_analysis(id)
);

-- 5. ТАБЛИЦЯ СИСТЕМНОЇ СТАТИСТИКИ (для моніторингу продуктивності)
CREATE TABLE IF NOT EXISTS system_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    component TEXT NOT NULL,
    messages_processed INTEGER DEFAULT 0,
    processing_time_ms INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    memory_usage_mb REAL,
    cpu_usage_percent REAL,
    metrics_data TEXT,
    version TEXT,
    node_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 6. ТАБЛИЦЯ МОДЕЛЕЙ (версіонування та метаданні ML)
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    file_path TEXT,
    config_json TEXT,
    feature_names TEXT,
    accuracy REAL,
    precision_score REAL,
    recall REAL,
    f1_score REAL,
    roc_auc REAL,
    training_dataset_size INTEGER,
    training_duration_minutes INTEGER,
    validation_score REAL,
    is_active BOOLEAN DEFAULT 0,
    deployment_status TEXT,
    trained_by TEXT,
    training_notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ===== ІНДЕКСИ =====
CREATE INDEX IF NOT EXISTS idx_pdus_timestamp ON captured_pdus(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON smpp_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON anomaly_analysis(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);

CREATE INDEX IF NOT EXISTS idx_messages_pdu_id ON smpp_messages(pdu_id);
CREATE INDEX IF NOT EXISTS idx_analysis_message_id ON anomaly_analysis(message_id);
CREATE INDEX IF NOT EXISTS idx_alerts_analysis_id ON alerts(analysis_id);

CREATE INDEX IF NOT EXISTS idx_messages_source ON smpp_messages(source_addr);
CREATE INDEX IF NOT EXISTS idx_messages_category ON smpp_messages(category);
CREATE INDEX IF NOT EXISTS idx_messages_features_extracted ON smpp_messages(features_extracted);

CREATE INDEX IF NOT EXISTS idx_analysis_is_anomaly ON anomaly_analysis(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_analysis_risk_level ON anomaly_analysis(risk_level);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority);

CREATE INDEX IF NOT EXISTS idx_messages_time_category ON smpp_messages(timestamp, category);
CREATE INDEX IF NOT EXISTS idx_messages_source_time ON smpp_messages(source_addr, timestamp);
CREATE INDEX IF NOT EXISTS idx_analysis_anomaly_score ON anomaly_analysis(is_anomaly, final_anomaly_score);
CREATE INDEX IF NOT EXISTS idx_alerts_status_priority ON alerts(status, priority);

CREATE INDEX IF NOT EXISTS idx_stats_component_time ON system_stats(component, timestamp);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active, model_type);
