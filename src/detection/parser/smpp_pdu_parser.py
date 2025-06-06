import sqlite3
import struct
from datetime import datetime
from typing import List, Dict, Tuple, Optional

def read_cstring(data: bytes, offset: int) -> Tuple[str, int]:
    """Читає C-style string (null-terminated) з байтового масиву"""
    end = data.find(b'\x00', offset)
    if end == -1:
        raise ValueError("CString not null-terminated")
    return data[offset:end].decode('ascii', errors='replace'), end + 1

def parse_submit_sm(raw_body: bytes) -> Dict[str, any]:
    """Парсить тіло submit_sm команди"""
    offset = 0
    result = {}
    
    try:
        if isinstance(raw_body, str):
            raw_body = bytes.fromhex(raw_body)
        # service_type
        _, offset = read_cstring(raw_body, offset)
        
        # source address
        source_addr_ton = raw_body[offset]
        offset += 1
        source_addr_npi = raw_body[offset]
        offset += 1
        source_addr, offset = read_cstring(raw_body, offset)
        
        # destination address
        dest_addr_ton = raw_body[offset]
        offset += 1
        dest_addr_npi = raw_body[offset]
        offset += 1
        dest_addr, offset = read_cstring(raw_body, offset)
        
        # message flags
        esm_class = raw_body[offset]
        offset += 1
        protocol_id = raw_body[offset]
        offset += 1
        priority_flag = raw_body[offset]
        offset += 1
        
        # schedule_delivery_time
        _, offset = read_cstring(raw_body, offset)
        # validity_period
        _, offset = read_cstring(raw_body, offset)
        
        registered_delivery = raw_body[offset]
        offset += 1
        replace_if_present_flag = raw_body[offset]
        offset += 1
        data_coding = raw_body[offset]
        offset += 1
        sm_default_msg_id = raw_body[offset]
        offset += 1
        # sm_length — кількість байтів, не символів
        sm_length = raw_body[offset]
        offset += 1
        short_message = raw_body[offset:offset + sm_length]
        offset += sm_length

        # Декодуємо повідомлення
        try:
            if data_coding == 8:
                message_text = short_message.decode('utf-16-be')
            else:
                message_text = short_message.decode('latin-1')
        except Exception as e:
            message_text = f"[decode error]: {e}"

        result.update({
            'source_addr': source_addr,
            'source_addr_ton': int(source_addr_ton),
            'source_addr_npi': int(source_addr_npi),
            'dest_addr': dest_addr,
            'dest_addr_ton': int(dest_addr_ton),
            'dest_addr_npi': int(dest_addr_npi),
            'esm_class': int(esm_class),
            'priority_flag': int(priority_flag),
            'data_coding': int(data_coding),
            'message_text': message_text,
            'message_length': len(message_text),
        })
    
        
    except Exception as e:
        raise ValueError(f"Error parsing submit_sm: {e}")
    
    return result

def parse_deliver_sm(raw_body: bytes) -> Dict[str, any]:
    """Парсить тіло deliver_sm команди (той самий формат що submit_sm)"""
    return parse_submit_sm(raw_body)

def calculate_basic_features(parsed_data: Dict[str, any]) -> Dict[str, any]:
    """Розраховує базові характеристики повідомлення"""
    features = {}
    
    # Перевірка чи джерело є числовим
    source_addr = parsed_data['source_addr']
    features['source_is_numeric'] = source_addr.isdigit() if source_addr else False
    features['source_addr_length'] = len(source_addr) if source_addr else 0
    
    # Перевірка на пусте повідомлення
    message_text = parsed_data.get('message_text', '')
    features['empty_message'] = len(message_text.strip()) == 0
    
    # Перевірка на проблеми з кодуванням (наявність replacement characters)
    features['encoding_issues'] = '�' in message_text or '\ufffd' in message_text
    
    # Перевірка на надмірну довжину
    features['excessive_length'] = parsed_data['message_length'] > 160
    
    return features

def create_tables(conn: sqlite3.Connection):
    """Створює таблицю smpp_messages якщо вона не існує"""
    cursor = conn.cursor()
    
    cursor.execute('''
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
    )
    ''')
    
    conn.commit()

def process_and_store_pdus(db_path: str, limit: int = 1000) -> int:
    """Обробляє PDU з таблиці captured_pdus та зберігає в smpp_messages"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Створюємо таблицю якщо не існує
    create_tables(conn)
    
    # Вибираємо submit_sm та deliver_sm PDU
    query = """
        SELECT id, raw_body, timestamp, command_name
        FROM captured_pdus
        WHERE command_name IN ('submit_sm', 'deliver_sm')
        ORDER BY timestamp DESC
        LIMIT ?
    """
    
    rows = cur.execute(query, (limit,)).fetchall()
    inserted = 0
    errors = 0
    
    print(f"Знайдено {len(rows)} PDU для обробки")
    
    for row in rows:
        pdu_id = row['id']
        raw_body = row['raw_body']
        timestamp = row['timestamp']
        command_name = row['command_name']
        
        # Конвертуємо raw_body якщо це hex string
        if isinstance(raw_body, str):
            try:
                raw_body = bytes.fromhex(raw_body)
            except ValueError:
                # Можливо це вже bytes в БД
                pass
        
        try:
            # Перевіряємо чи вже існує запис для цього pdu_id
            check = cur.execute("SELECT 1 FROM smpp_messages WHERE pdu_id = ?", (pdu_id,)).fetchone()
            if check:
                continue
            
            # Парсимо PDU
            if command_name == 'submit_sm':
                parsed = parse_submit_sm(raw_body)
            elif command_name == 'deliver_sm':
                parsed = parse_deliver_sm(raw_body)
            else:
                continue
            
            # Розраховуємо базові характеристики
            features = calculate_basic_features(parsed)
            
            # Отримуємо час з timestamp
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            hour = dt.hour
            day_of_week = dt.weekday()
            
            # Визначаємо часові категорії
            night_time = hour >= 22 or hour < 6
            weekend = day_of_week >= 5  # Субота (5) або Неділя (6)
            business_hours = 9 <= hour <= 18 and day_of_week < 5
            
            # Вставляємо дані
            cur.execute("""
                INSERT INTO smpp_messages (
                    pdu_id, timestamp,
                    source_addr, source_addr_ton, source_addr_npi,
                    dest_addr, dest_addr_ton, dest_addr_npi,
                    message_text, message_length,
                    data_coding, esm_class, priority_flag,
                    source_addr_length, source_is_numeric,
                    empty_message, encoding_issues, excessive_length,
                    hour, day_of_week,
                    night_time, weekend, business_hours
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pdu_id, timestamp,
                parsed['source_addr'], parsed['source_addr_ton'], parsed['source_addr_npi'],
                parsed['dest_addr'], parsed['dest_addr_ton'], parsed['dest_addr_npi'],
                parsed['message_text'], parsed['message_length'],
                parsed['data_coding'], parsed['esm_class'], parsed['priority_flag'],
                features['source_addr_length'], features['source_is_numeric'],
                features['empty_message'], features['encoding_issues'], features['excessive_length'],
                hour, day_of_week,
                night_time, weekend, business_hours
            ))
            
            inserted += 1
            
            if inserted % 100 == 0:
                print(f"  Оброблено {inserted} повідомлень...")
                conn.commit()
            
        except Exception as e:
            errors += 1
            print(f"[!] Помилка при обробці PDU id={pdu_id}: {e}")
            if errors < 10:  # Показуємо тільки перші 10 помилок
                print(f"    Raw body (перші 50 байт): {raw_body[:50].hex() if raw_body else 'None'}")
    
    conn.commit()
    conn.close()
    
    print(f"\nРезультат:")
    print(f"  Успішно оброблено: {inserted} повідомлень")
    print(f"  Помилок: {errors}")
    
    return inserted

def main():
    """Основна функція"""
    import sys
    
    if len(sys.argv) < 2:
        print("Використання: python smpp_pdu_parser.py <шлях_до_бази_даних> [ліміт]")
        sys.exit(1)
    
    db_path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    print(f"Обробка PDU з бази даних: {db_path}")
    print(f"Ліміт: {limit} записів")
    
    processed = process_and_store_pdus(db_path, limit)
    
    print(f"\nГотово! Оброблено {processed} PDU.")

if __name__ == "__main__":
    main()