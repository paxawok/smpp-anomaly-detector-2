import sqlite3
from pathlib import Path

def initialize_database(db_path: Path, schema_file: Path):
    # Створити директорію для БД, якщо не існує
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Відкрити з'єднання з БД (створює файл, якщо його ще немає)
    conn = sqlite3.connect(db_path)
    print(f"[✓] Підключено до бази даних: {db_path}")

    # Зчитати SQL-скрипт зі схемою
    with open(schema_file, 'r', encoding='utf-8') as f:
        sql_script = f.read()

    # Виконати SQL-інструкції
    try:
        with conn:
            conn.executescript(sql_script)
        print("[✓] Схему бази даних успішно створено.")
    except sqlite3.Error as e:
        print(f"[✗] Помилка під час створення схеми: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    DB_PATH = Path("data/db/smpp.sqlite")
    SCHEMA_FILE = Path("src/database/schema/init_schema.sql")

    initialize_database(DB_PATH, SCHEMA_FILE)
