import binascii
import sqlite3
import struct
from datetime import datetime
from typing import List, Dict, Tuple, Optional

def read_cstring(data: bytes, offset: int) -> Tuple[str, int]:
    """Читає C-style string (null-terminated) з байтового масиву"""
    end = data.find(b'\x00', offset)
    if end == -1:
        # Якщо не знайдено null-terminator, беремо до кінця
        return data[offset:].decode('ascii', errors='replace'), len(data)
    return data[offset:end].decode('ascii', errors='replace'), end + 1

def parse_submit_sm(raw_body: bytes) -> Dict[str, any]:
    """Парсить тіло submit_sm команди"""
    offset = 0
    result = {}
    
    try:
        # Конвертуємо raw_body в bytes якщо потрібно
        if isinstance(raw_body, str):
            # Спробуємо як hex string
            try:
                raw_body = bytes.fromhex(raw_body)
            except ValueError:
                # Можливо це base64 або інше кодування
                raw_body = raw_body.encode('latin-1')
        elif raw_body is None:
            raise ValueError("raw_body is None")
            
        if len(raw_body) < 20:  # Мінімальний розмір для submit_sm
            raise ValueError(f"raw_body занадто короткий: {len(raw_body)} байт")
        
        # service_type
        service_type, offset = read_cstring(raw_body, offset)
        
        # source address
        if offset >= len(raw_body):
            raise ValueError("Неочікуваний кінець даних при читанні source_addr_ton")
        source_addr_ton = raw_body[offset]
        offset += 1
        
        if offset >= len(raw_body):
            raise ValueError("Неочікуваний кінець даних при читанні source_addr_npi")
        source_addr_npi = raw_body[offset]
        offset += 1
        
        source_addr, offset = read_cstring(raw_body, offset)
        
        # destination address
        if offset >= len(raw_body):
            raise ValueError("Неочікуваний кінець даних при читанні dest_addr_ton")
        dest_addr_ton = raw_body[offset]
        offset += 1
        
        if offset >= len(raw_body):
            raise ValueError("Неочікуваний кінець даних при читанні dest_addr_npi")
        dest_addr_npi = raw_body[offset]
        offset += 1
        
        dest_addr, offset = read_cstring(raw_body, offset)
        
        # message flags
        if offset + 3 > len(raw_body):
            raise ValueError("Неочікуваний кінець даних при читанні message flags")
        esm_class = raw_body[offset]
        offset += 1
        protocol_id = raw_body[offset]
        offset += 1
        priority_flag = raw_body[offset]
        offset += 1
        
        # schedule_delivery_time
        schedule_delivery_time, offset = read_cstring(raw_body, offset)
        # validity_period
        validity_period, offset = read_cstring(raw_body, offset)
        
        if offset + 5 > len(raw_body):
            raise ValueError("Неочікуваний кінець даних при читанні delivery flags")
        
        registered_delivery = raw_body[offset]
        offset += 1
        replace_if_present_flag = raw_body[offset]
        offset += 1
        data_coding = raw_body[offset]
        offset += 1
        sm_default_msg_id = raw_body[offset]
        offset += 1
        
        # sm_length
        sm_length = raw_body[offset]
        offset += 1
        
        if offset + sm_length > len(raw_body):
            raise ValueError(f"sm_length ({sm_length}) виходить за межі raw_body")
        
        short_message = raw_body[offset:offset + sm_length]
        offset += sm_length

        # Обробка UDH та декодування повідомлення
        message_parts = 1
        udh_info = {}
        
        if esm_class & 0x40:  # UDHI bit set
            if len(short_message) > 0:
                udh_len = short_message[0]
                if udh_len + 1 <= len(short_message):
                    udh_data = short_message[1:udh_len + 1]
                    user_data = short_message[udh_len + 1:]
                    
                    # Парсимо UDH для отримання інформації про multipart
                    if len(udh_data) >= 5:
                        iei = udh_data[0]
                        iedl = udh_data[1]
                        if iei == 0x00 and iedl == 0x03:  # Concatenated SMS, 8-bit ref
                            udh_info['ref_num'] = udh_data[2]
                            udh_info['total_parts'] = udh_data[3]
                            udh_info['part_num'] = udh_data[4]
                            message_parts = udh_data[3]
                else:
                    user_data = short_message
            else:
                user_data = short_message
        else:
            user_data = short_message

        # Декодуємо повідомлення
        try:
            if data_coding == 0x08:  # UCS2
                message_text = user_data.decode('utf-16-be', errors='replace')
            elif data_coding == 0x00:  # GSM 7-bit default
                message_text = user_data.decode('latin-1', errors='replace')
            else:
                # Спробуємо різні кодування
                try:
                    message_text = user_data.decode('utf-8', errors='strict')
                except:
                    try:
                        message_text = user_data.decode('latin-1', errors='replace')
                    except:
                        message_text = f"[binary data: {user_data.hex()[:50]}...]"
        except Exception as e:
            message_text = f"[decode error: {e}]"

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
            'message_parts': message_parts,
            'udh_info': udh_info
        })
        
    except Exception as e:
        raise ValueError(f"Error parsing submit_sm at offset {offset}: {e}")
    
    return result

def parse_deliver_sm(raw_body: bytes) -> Dict[str, any]:
    """Парсить тіло deliver_sm команди (той самий формат що submit_sm)"""
    return parse_submit_sm(raw_body)

def calculate_basic_features(parsed_data: Dict[str, any]) -> Dict[str, any]:
    """Розраховує базові характеристики повідомлення"""
    features = {}
    
    # Перевірка чи джерело є числовим
    source_addr = parsed_data.get('source_addr', '')
    features['source_is_numeric'] = bool(source_addr and source_addr.strip().replace('+', '').isdigit())
    features['source_addr_length'] = len(source_addr) if source_addr else 0
    
    # Перевірка на пусте повідомлення
    message_text = parsed_data.get('message_text', '')
    features['empty_message'] = len(message_text.strip()) == 0
    
    # Перевірка на проблеми з кодуванням
    features['encoding_issues'] = any(c in message_text for c in ['�', '\ufffd', '\x00'])
    
    # Перевірка на надмірну довжину
    features['excessive_length'] = parsed_data.get('message_length', 0) > 160
    
    # Перевірка валідності номера призначення
    dest_addr = parsed_data.get('dest_addr', '')
    features['dest_is_valid'] = bool(dest_addr and dest_addr.strip().replace('+', '').isdigit())
    
    return features

def process_and_store_pdus(db_path: str, limit: int = 1000) -> int:
    """Обробляє PDU з таблиці captured_pdus та зберігає в smpp_messages"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Створюємо таблицю якщо не існує
    create_tables(conn)
    
    # Вибираємо submit_sm та deliver_sm PDU які ще не оброблені
    query = """
        SELECT p.id, p.raw_body, p.timestamp, p.command_name
        FROM captured_pdus p
        LEFT JOIN smpp_messages m ON p.id = m.pdu_id
        WHERE p.command_name IN ('submit_sm', 'deliver_sm')
        AND m.id IS NULL
        ORDER BY p.timestamp DESC
        LIMIT ?
    """
    
    rows = cur.execute(query, (limit,)).fetchall()
    inserted = 0
    errors = 0
    
    print(f"Знайдено {len(rows)} необроблених PDU")
    
    for idx, row in enumerate(rows):
        pdu_id = row['id']
        raw_body = row['raw_body']
        timestamp = row['timestamp']
        command_name = row['command_name']
        
        try:
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
            if isinstance(timestamp, str):
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            else:
                dt = timestamp
                
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
                    message_parts,
                    source_addr_length, source_is_numeric,
                    empty_message, encoding_issues, excessive_length,
                    dest_is_valid,
                    hour, day_of_week,
                    night_time, weekend, business_hours,
                    features_extracted,
                    feature_extraction_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pdu_id, timestamp,
                parsed['source_addr'], parsed['source_addr_ton'], parsed['source_addr_npi'],
                parsed['dest_addr'], parsed['dest_addr_ton'], parsed['dest_addr_npi'],
                parsed['message_text'], parsed['message_length'],
                parsed['data_coding'], parsed['esm_class'], parsed['priority_flag'],
                parsed.get('message_parts', 1),
                features['source_addr_length'], features['source_is_numeric'],
                features['empty_message'], features['encoding_issues'], features['excessive_length'],
                features['dest_is_valid'],
                hour, day_of_week,
                night_time, weekend, business_hours,
                1,  # features_extracted
                '1.0'  # feature_extraction_version
            ))
            
            inserted += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Оброблено {idx + 1}/{len(rows)} PDU...")
                conn.commit()
            
        except Exception as e:
            errors += 1
            if errors <= 10:  # Показуємо тільки перші 10 помилок
                print(f"[!] Помилка при обробці PDU id={pdu_id}: {e}")
                if raw_body:
                    print(f"    Command: {command_name}")
                    print(f"    Raw body length: {len(raw_body) if isinstance(raw_body, bytes) else 'not bytes'}")
                    if isinstance(raw_body, bytes):
                        print(f"    First 50 bytes: {raw_body[:50].hex()}")
    
    conn.commit()
    conn.close()
    
    print(f"\nРезультат:")
    print(f"  Успішно оброблено: {inserted} повідомлень")
    print(f"  Помилок: {errors}")
    
    return inserted

def create_tables(conn: sqlite3.Connection):
    """Створює таблицю smpp_messages якщо вона не існує"""
    # Код таблиці залишається без змін
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