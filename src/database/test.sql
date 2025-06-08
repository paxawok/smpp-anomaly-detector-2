SELECT
    aa.is_anomaly,aa.final_anomaly_score,
    sm.message_text
FROM
    anomaly_analysis AS aa
JOIN
    smpp_messages AS sm
    ON aa.message_id = sm.id
LIMIT 1000;

DELETE FROM captured_pdus;
DELETE FROM smpp_messages;
DELETE FROM sqlite_sequence WHERE name='captured_pdus';
DELETE FROM sqlite_sequence WHERE name='smpp_messages';
DROP TABLE IF EXISTS anomaly_analysis;
DELETE FROM anomaly_analysis;
DELETE FROM sqlite_sequence WHERE name='anomaly_analysis';
CREATE TABLE IF NOT EXISTS anomaly_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL UNIQUE,
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