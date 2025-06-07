import sqlite3
import random
import struct
import binascii
import json
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Tuple, Dict, Optional
import chardet

class SMPPPDUGenerator:
    """Генератор даних SMPP 3.4 PDU для заповнення БД"""
    
    def __init__(self, config_path: str = "config/pdu_config.json"):
        self.sequence_number = 1
        self.base_time = datetime.now() - timedelta(hours=24)
        
        # Завантажуємо конфігурацію
        self.config = self._load_config(config_path)
        
        # Конвертуємо команди з hex string в int
        self.commands = {}
        for hex_id, cmd_info in self.config['commands'].items():
            self.commands[int(hex_id, 16)] = cmd_info['name']
        
        # Конвертуємо статус коди
        self.status_codes = {}
        for hex_code, status_info in self.config['status_codes'].items():
            self.status_codes[int(hex_code, 16)] = status_info['name']
    
    def _load_config(self, config_path: str) -> Dict:
        """Завантажує конфігурацію з JSON файлу"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Конфігураційний файл {config_path} не знайдено")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def generate_ip_addresses(self) -> Tuple[str, str]:
        """Генерує пару IP адрес (джерело та призначення)"""
        network_config = self.config['network']
        src_ip = random.choice(network_config['client_ips'])
        dst_ip = random.choice(network_config['server_ips'])
        
        return src_ip, dst_ip
    
    def generate_ports(self) -> Tuple[int, int]:
        """Генерує порти для SMPP з'єднання"""
        network_config = self.config['network']
        dst_port = random.choice(network_config['smpp_ports'])
        
        port_range = network_config['client_port_range']
        src_port = random.randint(port_range['min'], port_range['max'])
        
        return src_port, dst_port
    
    def generate_pdu_header(self, command_id: int, command_status: int, body_length: int) -> bytes:
        """Генерує SMPP PDU header"""
        command_length = 16 + body_length  # Header (16 bytes) + Body
        
        # SMPP Header Format:
        # command_length (4 bytes)
        # command_id (4 bytes)  
        # command_status (4 bytes)
        # sequence_number (4 bytes)
        header = struct.pack('>IIII', 
                           command_length,
                           command_id,
                           command_status,
                           self.sequence_number)
        
        return header
    
    def generate_submit_sm_pdus_from_text(self, source_addr: str, dest_addr: str, 
                                         message_text: str, submit_time: datetime) -> List[Dict]:
        """
        Генерує один або кілька submit_sm PDU для збереження в таблиці captured_pdus
        """
        
        try:
            # Очищення та нормалізація тексту
            message_text = message_text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2026', '...')
            message_text = ' '.join(message_text.split())
            
            # Кодуємо в UCS2 (UTF-16 BE)
            short_message = message_text.encode('utf-16-be')
            data_coding_value = 0x08  # UCS2
            
        except Exception as e:
            # Fallback на GSM 7-bit
            try:
                short_message = message_text.encode('latin-1')
                data_coding_value = 0x00  # Default alphabet
            except:
                raise ValueError(f"Неможливо закодувати повідомлення: {message_text[:50]}...")

        segments = []
        max_single_part_len = 140  # 70 символів UCS2
        max_segment_len = 134      # 140 - 6 байтів UDH

        if len(short_message) <= max_single_part_len:
            segments = [short_message]
        else:
            ref_number = random.randint(0, 255)
            parts = []
            for i in range(0, len(short_message), max_segment_len):
                parts.append(short_message[i:i + max_segment_len])
            
            total_parts = len(parts)
            
            for i, part in enumerate(parts):
                udh = bytes([
                    0x05,  # UDH length
                    0x00,  # IEI: Concatenated short messages, 8-bit reference
                    0x03,  # IEDL: length of data
                    ref_number & 0xFF,
                    total_parts & 0xFF,
                    (i + 1) & 0xFF
                ])
                segments.append(udh + part)

        pdu_list = []
        
        # Генеруємо мережеві параметри один раз для всієї сесії
        src_ip, dst_ip = self.generate_ip_addresses()
        src_port, dst_port = self.generate_ports()
        
        for i, segment in enumerate(segments):
            # ESM class
            esm_class = 0x00 if len(segments) == 1 else 0x40  # UDHI indicator для multipart
            
            # Параметри адресації
            source_addr_ton = 0x01  # International
            source_addr_npi = 0x01  # ISDN
            dest_addr_ton = 0x01    # International
            dest_addr_npi = 0x01    # ISDN
            
            # Інші параметри
            service_type = b'\x00'
            protocol_id = 0x00
            priority_flag = 0x00
            schedule_delivery_time = b'\x00'
            validity_period = b'\x00'
            registered_delivery = 0x01
            replace_if_present_flag = 0x00
            sm_default_msg_id = 0x00
            sm_length = len(segment)
            
            # Збираємо тіло PDU
            body = b''
            body += service_type
            body += source_addr_ton.to_bytes(1, 'big')
            body += source_addr_npi.to_bytes(1, 'big')
            body += source_addr.encode('ascii', errors='replace') + b'\x00'
            body += dest_addr_ton.to_bytes(1, 'big')
            body += dest_addr_npi.to_bytes(1, 'big')
            body += dest_addr.encode('ascii', errors='replace') + b'\x00'
            body += esm_class.to_bytes(1, 'big')
            body += protocol_id.to_bytes(1, 'big')
            body += priority_flag.to_bytes(1, 'big')
            body += schedule_delivery_time
            body += validity_period
            body += registered_delivery.to_bytes(1, 'big')
            body += replace_if_present_flag.to_bytes(1, 'big')
            body += data_coding_value.to_bytes(1, 'big')
            body += sm_default_msg_id.to_bytes(1, 'big')
            body += sm_length.to_bytes(1, 'big')
            body += segment
            
            # Заголовок PDU
            command_id = 0x00000004  # submit_sm
            command_status = 0x00000000
            command_length = 16 + len(body)
            
            header = struct.pack('>IIII', 
                               command_length,
                               command_id,
                               command_status,
                               self.sequence_number)
            
            full_pdu = header + body
            
            # Формуємо словник для таблиці captured_pdus
            pdu_data = {
                'timestamp': submit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'src_ip': src_ip,
                'dst_ip': dst_ip, 
                'src_port': src_port,
                'dst_port': dst_port,
                'command_id': command_id,
                'command_name': 'submit_sm',
                'command_status': command_status,
                'sequence_number': self.sequence_number,
                'command_length': command_length,
                'raw_data': binascii.hexlify(full_pdu).decode('ascii'),
                'raw_header': header,  # BLOB
                'raw_body': body,      # BLOB
            }
            
            self.sequence_number += 1
            pdu_list.append(pdu_data)
        
        return pdu_list

    def generate_bind_body(self) -> bytes:
        """Генерує тіло для bind команд"""
        test_data = self.config['test_data']
        
        system_id = random.choice(test_data['system_ids']).encode('latin-1')
        password = random.choice(test_data['passwords']).encode('latin-1')
        system_type = random.choice(test_data['system_types']).encode('latin-1')
        interface_version = b'\x34'  # 3.4
        addr_ton = b'\x00'
        addr_npi = b'\x00'
        address_range = b''
        
        body = system_id + b'\x00' + \
               password + b'\x00' + \
               system_type + b'\x00' + \
               interface_version + \
               addr_ton + addr_npi + \
               address_range + b'\x00'
        
        return body
    
    def generate_submit_sm_body(self) -> bytes:
        """Генерує тіло для submit_sm команди"""
        test_data = self.config['test_data']
        
        service_type = random.choice(test_data['service_types']).encode('latin-1')
        source_addr_ton = b'\x01'
        source_addr_npi = b'\x01'
        
        prefix = random.choice(test_data['phone_prefixes'])
        source_addr = f"{prefix}{random.randint(1000000, 9999999)}".encode('latin-1')
        
        dest_addr_ton = b'\x01'
        dest_addr_npi = b'\x01'
        prefix = random.choice(test_data['phone_prefixes'])
        destination_addr = f"{prefix}{random.randint(1000000, 9999999)}".encode('latin-1')
        
        esm_class = b'\x00'
        protocol_id = b'\x00'
        priority_flag = b'\x00'
        schedule_delivery_time = b'\x00'
        validity_period = b'\x00'
        registered_delivery = b'\x01'
        replace_if_present_flag = b'\x00'
        
        message_text = random.choice(test_data['messages'])

        try:
            short_message = message_text.encode('latin-1')
            data_coding_value = 0
        except UnicodeEncodeError:
            short_message = message_text.encode('utf-16-be')
            data_coding_value = 8

        data_coding = data_coding_value.to_bytes(1, 'big')
        sm_default_msg_id = b'\x00'
        sm_length = len(short_message).to_bytes(1, 'big')

        body = (
            service_type + b'\x00' +
            source_addr_ton + source_addr_npi + source_addr + b'\x00' +
            dest_addr_ton + dest_addr_npi + destination_addr + b'\x00' +
            esm_class + protocol_id + priority_flag +
            schedule_delivery_time + 
            validity_period + 
            registered_delivery + replace_if_present_flag +
            data_coding + 
            sm_default_msg_id + 
            sm_length + 
            short_message
        )
        
        return body
    
    def generate_pdu(self) -> Dict:
        """Генерує повний PDU"""
        # Вибираємо команду
        command_id = random.choice(list(self.commands.keys()))
        command_name = self.commands[command_id]
        
        # Для відповідей (resp) статус може бути різний, для запитів - 0
        if command_id & 0x80000000:  # Це відповідь
            command_status = random.choice([0] * 8 + list(self.status_codes.keys())[1:3])
        else:
            command_status = 0
        
        # Генеруємо тіло PDU залежно від команди
        if "bind" in command_name and "resp" not in command_name:
            body = self.generate_bind_body()
        elif command_name == "submit_sm":
            body = self.generate_submit_sm_body()
        elif command_name == "enquire_link":
            body = b''  # Enquire link не має тіла
        elif "resp" in command_name:
            # Для відповідей можемо додати system_id або message_id
            if "bind" in command_name and command_status == 0:
                server_id = random.choice(self.config['test_data']['server_ids'])
                body = server_id.encode('latin-1') + b'\x00'
            elif "submit_sm_resp" in command_name and command_status == 0:
                prefix = random.choice(self.config['test_data']['message_ids_prefix'])
                body = f"{prefix}{random.randint(1000000, 9999999)}".encode('latin-1') + b'\x00'
            else:
                body = b''
        else:
            body = b''  # Спрощено для інших команд
        
        # Генеруємо header
        header = self.generate_pdu_header(command_id, command_status, len(body))
        
        # Повний PDU
        full_pdu = header + body
        
        # Генеруємо мета-дані
        src_ip, dst_ip = self.generate_ip_addresses()
        src_port, dst_port = self.generate_ports()
        
        # Якщо це відповідь, міняємо джерело і призначення
        if command_id & 0x80000000:
            src_ip, dst_ip = dst_ip, src_ip
            src_port, dst_port = dst_port, src_port
        
        # Час з випадковим зсувом
        timestamp = self.base_time + timedelta(seconds=random.randint(0, 86400))
        
        pdu_data = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'command_id': command_id,
            'command_name': command_name,
            'command_status': command_status,
            'sequence_number': self.sequence_number,
            'command_length': len(full_pdu),
            'raw_data': binascii.hexlify(full_pdu).decode('ascii'),
            'raw_header': header,
            'raw_body': body
        }
        
        self.sequence_number += 1
        return pdu_data
    
    def generate_conversation(self, scenario_name: str = "basic_session") -> List[Dict]:
        """Генерує логічну послідовність SMPP повідомлень за сценарієм"""
        scenarios = self.config.get('scenarios', {})
        
        if scenario_name not in scenarios:
            # Якщо сценарій не знайдено, використовуємо базовий
            scenario_name = "basic_session"
            
        scenario = scenarios.get(scenario_name, {})
        sequence = scenario.get('sequence', [])
        
        pdus = []
        
        for step in sequence:
            command_name = step['command']
            expect_status = step.get('expect_status', 0)
            repeat = step.get('repeat', 1)
            delay = step.get('delay', 0)
            
            # Знаходимо command_id за назвою
            command_id = None
            for cmd_id, cmd_name in self.commands.items():
                if cmd_name == command_name:
                    command_id = cmd_id
                    break
            
            if command_id is None:
                continue
                
            # Генеруємо PDU потрібну кількість разів
            for _ in range(repeat):
                # Запит
                if not (command_id & 0x80000000):
                    pdu = self.generate_specific_pdu(command_id)
                    pdus.append(pdu)
                    
                    # Відповідь (якщо не alert_notification або outbind)
                    if command_id not in [0x00000102, 0x0000000B]:
                        resp_id = command_id | 0x80000000
                        
                        # Конвертуємо статус, якщо він у форматі hex string
                        if isinstance(expect_status, str):
                            status = int(expect_status, 16)
                        else:
                            status = expect_status
                            
                        resp_pdu = self.generate_specific_pdu(resp_id, status=status)
                        
                        # Додаємо затримку якщо вказано
                        if delay > 0:
                            resp_pdu['timestamp'] = (
                                datetime.strptime(pdu['timestamp'], '%Y-%m-%d %H:%M:%S') + 
                                timedelta(seconds=delay)
                            ).strftime('%Y-%m-%d %H:%M:%S')
                            
                        pdus.append(resp_pdu)
                else:
                    # Це вже відповідь
                    pdus.append(self.generate_specific_pdu(command_id, status=expect_status))
        
        return pdus
    
    def generate_specific_pdu(self, command_id: int, status: int = 0) -> Dict:
        """Генерує PDU для конкретної команди"""
        old_seq = self.sequence_number
        pdu = self.generate_pdu()
        
        # Переписуємо з потрібними параметрами
        pdu['command_id'] = command_id
        pdu['command_name'] = self.commands[command_id]
        pdu['command_status'] = status
        
        # Перегенеруємо header та body для правильної команди
        command_name = self.commands[command_id]
        
        # Генеруємо тіло PDU залежно від команди
        if "bind" in command_name and "resp" not in command_name:
            body = self.generate_bind_body()
        elif command_name == "submit_sm":
            body = self.generate_submit_sm_body()
        elif command_name == "deliver_sm":
            body = self.generate_submit_sm_body()  # Використовуємо той самий формат
        elif command_name == "enquire_link":
            body = b''
        elif "resp" in command_name:
            if "bind" in command_name and status == 0:
                server_id = random.choice(self.config['test_data']['server_ids'])
                body = server_id.encode('latin-1') + b'\x00'
            elif "submit_sm_resp" in command_name and status == 0:
                prefix = random.choice(self.config['test_data']['message_ids_prefix'])
                body = f"{prefix}{random.randint(1000000, 9999999)}".encode('latin-1') + b'\x00'
            else:
                body = b''
        else:
            body = b''
        
        # Генеруємо новий header
        header = self.generate_pdu_header(command_id, status, len(body))
        full_pdu = header + body
        
        # Оновлюємо дані PDU
        pdu['raw_header'] = header
        pdu['raw_body'] = body
        pdu['raw_data'] = binascii.hexlify(full_pdu).decode('ascii')
        pdu['command_length'] = len(full_pdu)
        
        return pdu

# Глобальні функції (не методи класу)
def detect_encoding(file_path: str) -> str:
    """Автоматично визначає кодування файлу"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Перші 10KB
        
        # Перевірка на UTF-16 BOM
        if raw_data.startswith(b'\xff\xfe'):
            return 'utf-16-le'
        elif raw_data.startswith(b'\xfe\xff'):
            return 'utf-16-be'
        
        # Використовуємо chardet для визначення
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

def read_csv_safe(csv_path: str) -> pd.DataFrame:
    """Безпечне читання CSV з автовизначенням кодування"""
    encoding = detect_encoding(csv_path)
    print(f"  Визначено кодування: {encoding}")
    
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"  Успішно прочитано {len(df)} записів")
        return df
    except Exception as e:
        print(f"  Спроба з альтернативними кодуваннями...")
        # Спробуємо інші популярні кодування
        for enc in ['utf-16', 'utf-8', 'latin-1', 'cp1251']:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                print(f"  Успішно прочитано з кодуванням {enc}")
                return df
            except:
                continue
        raise ValueError(f"Не вдалося прочитати CSV файл: {e}")

def insert_pdus_batch(conn: sqlite3.Connection, pdus: List[Dict]) -> int:
    """Вставляє PDU в БД пакетами для кращої продуктивності"""
    cursor = conn.cursor()
    inserted = 0
    
    # Вставляємо пакетами по 100 записів
    batch_size = 100
    for i in range(0, len(pdus), batch_size):
        batch = pdus[i:i + batch_size]
        
        # Підготовка даних для bulk insert
        pdu_values = []
        for pdu in batch:
            pdu_values.append((
                pdu['timestamp'],
                pdu.get('src_ip', '127.0.0.1'),
                pdu.get('dst_ip', '127.0.0.1'),
                pdu.get('src_port', 12345),
                pdu.get('dst_port', 2775),
                pdu['command_id'],
                pdu['command_name'],
                pdu.get('command_status', 0),
                pdu['sequence_number'],
                pdu['command_length'],
                pdu['raw_data'],
                pdu['raw_header'],
                pdu['raw_body']
            ))
        
        try:
            cursor.executemany("""
                INSERT INTO captured_pdus (
                    timestamp, src_ip, dst_ip, src_port, dst_port,
                    command_id, command_name, command_status,
                    sequence_number, command_length, raw_data,
                    raw_header, raw_body
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, pdu_values)
            inserted += len(pdu_values)
        except Exception as e:
            print(f"[!] Помилка вставки пакету: {e}")
            
    conn.commit()
    return inserted

def insert_pdus(conn: sqlite3.Connection, pdus: List[Dict]):
    """Стара функція для сумісності"""
    cursor = conn.cursor()
    
    for pdu in pdus:
        cursor.execute('''
        INSERT INTO captured_pdus (
            timestamp, src_ip, dst_ip, src_port, dst_port,
            command_id, command_name, command_status, sequence_number,
            command_length, raw_data, raw_header, raw_body
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pdu['timestamp'],
            pdu['src_ip'],
            pdu['dst_ip'],
            pdu['src_port'],
            pdu['dst_port'],
            pdu['command_id'],
            pdu['command_name'],
            pdu['command_status'],
            pdu['sequence_number'],
            pdu['command_length'],
            pdu['raw_data'],
            pdu['raw_header'],
            pdu['raw_body']
        ))
    
    conn.commit()

def main():
    """Головна функція для генерації PDU з CSV та інших джерел"""
    
    # Конфігурація
    config_path = "config/pdu_config.json"
    if not os.path.exists(config_path):
        print(f"Помилка: Файл конфігурації {config_path} не знайдено!")
        return

    print("=" * 60)
    print("SMPP PDU Generator - Початок роботи")
    print("=" * 60)

    # Зчитування конфігурації
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    db_path = config.get("database_path")
    csv_path = config.get("csv_dataset_path")

    # Перевірка БД
    if not db_path or not os.path.exists(db_path):
        print(f"Помилка: База даних {db_path} не існує!")
        return

    # Ініціалізація генератора
    generator = SMPPPDUGenerator(config_path)
    print("\n📊 Генерація SMPP PDU даних...")

    all_pdus = []
    csv_pdu_count = 0

    # 1. Генерація PDU з CSV-файлу
    if csv_path and os.path.exists(csv_path):
        print(f"\n📁 Обробка CSV файлу: {csv_path}")
        try:
            # Читання CSV з автовизначенням кодування
            df = read_csv_safe(csv_path)
            
            # Статистика по CSV
            print(f"  Колонки: {', '.join(df.columns)}")
            print(f"  Перші 3 записи для перевірки:")
            
            # Обробка записів
            processed = 0
            skipped = 0
            errors = 0
            
            for idx, row in df.iterrows():
                try:
                    # Отримання даних
                    source = str(row['source_addr']).strip()
                    dest = str(row['dest_addr']).strip()
                    text = str(row['message_text'])
                    
                    # Перевірки
                    if pd.isna(text) or text == 'nan' or not text.strip():
                        skipped += 1
                        continue
                    
                    # Парсинг timestamp
                    try:
                        timestamp = pd.to_datetime(row['submit_time'])
                    except:
                        timestamp = datetime.now() - timedelta(hours=idx % 24)
                    
                    # Показуємо прогрес для перших записів
                    if idx < 3:
                        print(f"    [{idx}] {source} -> {dest}: {text[:50]}...")
                    
                    # Генерація PDU
                    pdus = generator.generate_submit_sm_pdus_from_text(
                        source, dest, text, timestamp
                    )
                    all_pdus.extend(pdus)
                    csv_pdu_count += len(pdus)
                    processed += 1
                    
                    # Прогрес кожні 100 записів
                    if (idx + 1) % 100 == 0:
                        print(f"  ✓ Оброблено {idx + 1} записів...")
                        
                except Exception as e:
                    errors += 1
                    if errors <= 5:  # Показуємо лише перші 5 помилок
                        print(f"  [!] Помилка в рядку {idx}: {e}")
            
            print(f"\n  📊 Результати обробки CSV:")
            print(f"     • Оброблено: {processed}")
            print(f"     • Пропущено: {skipped}")
            print(f"     • Помилок: {errors}")
            print(f"     • Згенеровано PDU: {csv_pdu_count}")
            
        except Exception as e:
            print(f"  [!] Критична помилка при обробці CSV: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  CSV-файл не знайдено: {csv_path}")

    # 2. Генерація сценаріїв
    print(f"\n🎭 Генерація тестових сценаріїв...")
    scenarios = ["basic_session", "high_throughput", "error_handling", "delivery_receipts"]
    
    scenario_start = len(all_pdus)
    for i, scenario in enumerate(scenarios):
        print(f"  • {scenario}...", end='')
        generator.base_time = datetime.now() - timedelta(hours=20 - i * 5)
        scenario_pdus = generator.generate_conversation(scenario)
        all_pdus.extend(scenario_pdus)
        print(f" ✓ ({len(scenario_pdus)} PDU)")

    # 3. Випадкові PDU
    print(f"\n🎲 Генерація випадкових PDU...")
    random_start = len(all_pdus)
    for _ in range(30):
       all_pdus.append(generator.generate_pdu())
    print(f"  ✓ Згенеровано {len(all_pdus) - random_start} випадкових PDU")

    # Сортування за часом
    all_pdus.sort(key=lambda x: x['timestamp'])
    
    # 4. Збереження в БД
    print(f"\n💾 Збереження в базу даних...")
    conn = sqlite3.connect(db_path)
    
    try:
        # Статистика до вставки
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM captured_pdus")
        count_before = cursor.fetchone()[0]
        
        # Вставка даних
        inserted = insert_pdus_batch(conn, all_pdus)
        
        # Статистика після вставки
        cursor.execute("SELECT COUNT(*) FROM captured_pdus")
        count_after = cursor.fetchone()[0]
        
        print(f"  ✓ Успішно вставлено {inserted} PDU")
        print(f"  📈 Всього записів: {count_before} -> {count_after}")
        
        # 5. Детальна статистика
        print("\n📊 Статистика по типах команд:")
        cursor.execute("""
            SELECT command_name, COUNT(*) as count 
            FROM captured_pdus 
            GROUP BY command_name 
            ORDER BY count DESC
        """)
        for row in cursor.fetchall():
            print(f"  • {row[0]:<20} {row[1]:>6}")
        
        # Статистика помилок
        cursor.execute("""
            SELECT command_status, COUNT(*) 
            FROM captured_pdus 
            WHERE command_status != 0 
            GROUP BY command_status 
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        errors = cursor.fetchall()
        if errors:
            print("\n⚠️  Статистика помилок (топ-10):")
            for code, count in errors:
                status_name = generator.status_codes.get(code, f"Unknown (0x{code:08X})")
                print(f"  • {status_name:<30} {count:>6}")
        
        # Статистика по джерелах
        print(f"\n📌 Статистика по джерелах PDU:")
        print(f"  • З CSV файлу:      {csv_pdu_count:>6}")
        print(f"  • Зі сценаріїв:     {random_start - scenario_start:>6}")
        print(f"  • Випадкові:        {len(all_pdus) - random_start:>6}")
        print(f"  • ВСЬОГО:           {len(all_pdus):>6}")
        
    except Exception as e:
        print(f"\n❌ Помилка при роботі з БД: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Генерація завершена успішно!")
    print("=" * 60)

if __name__ == "__main__":
    main()