import pandas as pd
import re
from typing import Dict, Set, Tuple
import unicodedata
import json


class ObfuscationDetector:
    """Детектор обфускації"""
    
    def __init__(self):
        config_name = 'src/detection/obfuscation/obfuscation_config.json'
        with open(config_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.invisible_chars = set([bytes(c, "utf-8").decode("unicode_escape") for c in data["invisible_chars"]])
            self.confusables = data["confusables"] 
            self.brands_with_numbers = set(data["brands_with_numbers"])
            self.known_brands = set(data["known_brands"])   
            self.allowed_english_words = set(data["allowed_english_words"])    
    
    def is_url(self, text: str) -> bool:
        """Перевірка чи є текст URL"""
        url_pattern = r'https?://|www\.|\.com|\.ua|\.org|\.net'
        return bool(re.search(url_pattern, text, re.IGNORECASE))
    
    def is_emoji(self, char: str) -> bool:
        """Перевірка чи є символ емодзі"""
        # Емодзі знаходяться в певних діапазонах Unicode
        return any([
            '\U0001F300' <= char <= '\U0001F9FF',  # Основні емодзі
            '\U00002600' <= char <= '\U000027BF',  # Різні символи
            '\U0001F600' <= char <= '\U0001F64F',  # Емотикони
            '\U0001F680' <= char <= '\U0001F6FF',  # Транспорт
            '\U00002700' <= char <= '\U000027BF',  # Dingbats
        ])
    
    def is_valid_brand_with_numbers(self, word: str) -> bool:
        """Перевірка чи є слово брендом з цифрами"""
        word_lower = word.lower()
        
        # Прямий збіг з відомими брендами з цифрами
        if word_lower in self.brands_with_numbers:
            return True
        
        # Перевірка патерну: бренд + цифри в кінці
        match = re.match(r'^([a-zA-Z]+)(\d+)$', word_lower)
        if match:
            brand_part = match.group(1)
            numbers = match.group(2)
            # Якщо основа - відомий бренд і цифри в кінці (2+ цифри)
            if brand_part in self.known_brands and len(numbers) >= 2:
                return True
        
        return False
    
    def is_known_term(self, word: str) -> bool:
        """Перевірка чи є слово відомим брендом або терміном"""
        word_lower = word.lower()
        
        # Перевірка брендів з цифрами
        if self.is_valid_brand_with_numbers(word_lower):
            return True
        
        # Прямий збіг
        if word_lower in self.known_brands or word_lower in self.allowed_english_words:
            return True
        
        # Перевірка чи слово містить відомий бренд
        for brand in self.known_brands:
            if len(brand) >= 4 and brand in word_lower:
                return True
        
        return False
    
    def detect_alphabet(self, char: str) -> str:
        """Визначення алфавіту символу"""
        if '\u0400' <= char <= '\u04FF':
            return 'cyrillic'
        elif 'a' <= char.lower() <= 'z':
            return 'latin'
        elif char.isdigit():
            return 'digit'
        elif self.is_emoji(char):
            return 'emoji'
        else:
            category = unicodedata.category(char)
            if category.startswith('L'):
                name = unicodedata.name(char, '').lower()
                if 'cyrillic' in name:
                    return 'cyrillic'
                elif 'latin' in name:
                    return 'latin'
        return 'other'
    
    def analyze_digit_placement(self, word: str) -> float:
        """Аналіз розміщення цифр у слові"""
        if not any(c.isdigit() for c in word):
            return 0.0
        
        # Якщо слово - відомий бренд з цифрами
        if self.is_valid_brand_with_numbers(word):
            return 0.0
        
        # Патерни з низькою підозрою
        # Цифри тільки в кінці (2+ цифри)
        if re.match(r'^[^\d]+\d{2,}$', word):
            return 0.0
        
        # Цифри тільки на початку
        if re.match(r'^\d+[^\d]+$', word):
            return 0.1
        
        # Підозрілі патерни
        suspicious_score = 0.0
        
        # Цифри всередині слова між літерами
        for i in range(1, len(word) - 1):
            if word[i].isdigit():
                if word[i-1].isalpha() and word[i+1].isalpha():
                    # Перевірка чи це заміна схожої літери
                    if word[i] in self.confusables:
                        suspicious_score += 0.3
                    else:
                        suspicious_score += 0.1
        
        # Поодинокі цифри в кінці (підозріло)
        if re.match(r'^[^\d]+\d$', word):
            suspicious_score += 0.2
        
        return min(suspicious_score, 1.0)
    
    def has_confusable_substitution(self, word: str) -> Tuple[bool, float]:
        """Перевірка на заміну схожими символами"""
        if len(word) < 2:
            return False, 0.0
        
        # Визначаємо склад слова
        alphabets = {'cyrillic': 0, 'latin': 0, 'digit': 0, 'emoji': 0, 'other': 0}
        char_positions = []
        
        for i, char in enumerate(word):
            alphabet = self.detect_alphabet(char)
            alphabets[alphabet] += 1
            char_positions.append((char, alphabet, i))
        
        # Ігноруємо емодзі при підрахунку
        total_letters = alphabets['cyrillic'] + alphabets['latin']
        
        if total_letters == 0:
            return False, 0.0
        
        # Аналіз цифр
        digit_score = self.analyze_digit_placement(word)
        
        # Якщо слово повністю одним алфавітом (без урахування цифр/емодзі)
        if alphabets['cyrillic'] == 0 or alphabets['latin'] == 0:
            # Але є підозрілі цифри
            if digit_score > 0.3:
                return True, digit_score
            # Якщо немає змішування алфавітів, але є цифри всередині слова
            if digit_score > 0:
                return True, digit_score
            return False, digit_score
        
        # Визначаємо домінуючий алфавіт
        main_alphabet = 'cyrillic' if alphabets['cyrillic'] > alphabets['latin'] else 'latin'
        foreign_ratio = min(alphabets['cyrillic'], alphabets['latin']) / total_letters
        
        # Якщо є змішування алфавітів - це вже підозріло
        if foreign_ratio > 0 and total_letters > 2:
            # Перевіряємо чи це схожі символи
            for char in word:
                if char in self.confusables:
                    # Якщо символ має схожі в іншому алфавіті
                    char_alphabet = self.detect_alphabet(char)
                    for confusable in self.confusables[char]:
                        confusable_alphabet = self.detect_alphabet(confusable)
                        # Якщо символ одного алфавіту, а його схожий - іншого
                        if char_alphabet != confusable_alphabet:
                            # І в слові є змішування - це обфускація
                            if foreign_ratio > 0:
                                return True, max(0.6, foreign_ratio * 2)
        
        # Детальний аналіз замін
        suspicious_substitutions = 0
        
        for i, (char, char_alphabet, pos) in enumerate(char_positions):
            if char_alphabet in ['cyrillic', 'latin', 'digit']:
                # Контекст символу
                prev_char = char_positions[i-1] if i > 0 else None
                next_char = char_positions[i+1] if i < len(char_positions)-1 else None
                
                # Перевірка на заміну схожим символом
                if char in self.confusables:
                    confusable_chars = self.confusables[char]
                    
                    # Якщо символ не відповідає контексту
                    if prev_char and prev_char[1] != char_alphabet and prev_char[1] != 'digit':
                        # Перевірка чи це може бути заміна
                        if any(c in word for c in confusable_chars):
                            suspicious_substitutions += 1
                    
                    if next_char and next_char[1] != char_alphabet and next_char[1] != 'digit':
                        if any(c in word for c in confusable_chars):
                            suspicious_substitutions += 0.5
        
        # Розрахунок фінального балу
        substitution_score = suspicious_substitutions / len(word)
        
        # Комбінований бал
        final_score = max(
            substitution_score * 0.6 + foreign_ratio * 0.4,
            digit_score,
            foreign_ratio * 2  # Якщо є змішування - це вже підозріло
        )
        
        # Поріг для визначення обфускації
        if final_score > 0.1 or (foreign_ratio > 0.1 and total_letters > 2):
            return True, min(final_score * 1.5, 1.0)
        
        return False, 0.0
    
    def calculate_obfuscation_score(self, text: str) -> float:
        """Розрахунок рівня обфускації для тексту"""
        if not text or not isinstance(text, str):
            return 0.0

        text = text.strip()
        if len(text) < 2:
            return 0.0

        # Видаляємо номери телефону формату +380xxxxxxxxx
        phone_pattern = r'\+380\d{9}'
        text = re.sub(phone_pattern, '', text)

        # Видаляємо URL
        url_pattern = r'https?://\S+|www\.\S+|\S+\.(com|ua|org|net)\b'
        text = re.sub(url_pattern, '', text, flags=re.IGNORECASE)

        # Розбиваємо на слова
        words = re.findall(r'\b[\w]+\b', text, re.UNICODE)
        if not words:
            return 0.0

        obfuscation_scores = []
        total_words = 0

        for word in words:
            # Пропускаємо короткі слова та чисті числа
            if len(word) < 3 or word.isdigit():
                continue

            # Пропускаємо відомі бренди та терміни
            if self.is_known_term(word):
                continue

            total_words += 1

            # Перевіряємо на обфускацію
            is_obfuscated, score = self.has_confusable_substitution(word)
            if is_obfuscated:
                obfuscation_scores.append(score)

        # Невидимі символи
        invisible_count = sum(char in self.invisible_chars for char in text)
        invisible_score = max(0.5, min(invisible_count * 0.5, 1.0)) if invisible_count > 0 else 0.0

        # Дивні символи Unicode (не емодзі)
        strange_chars = 0
        for char in text:
            if ord(char) > 127 and not self.is_emoji(char):
                if char not in 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ':
                    category = unicodedata.category(char)
                    if category in ['Cf', 'Co', 'Cn']:
                        strange_chars += 1

        strange_score = min(strange_chars * 0.2, 1.0)

        # Фінальний розрахунок
        if not obfuscation_scores and invisible_score == 0 and strange_score == 0:
            return 0.0

        # Середній бал обфускованих слів
        word_score = sum(obfuscation_scores) / max(len(obfuscation_scores), 1) if obfuscation_scores else 0

        # Комбінований бал
        final_score = (
            word_score * 0.6 +
            invisible_score * 0.3 +
            strange_score * 0.1
        )

        return min(final_score, 1.0)
    
    def analyze_text_and_sender(self, message_text: str, sender_addr: str) -> Dict[str, float]:
        """Аналіз повідомлення та відправника"""
        message_score = self.calculate_obfuscation_score(message_text)
        sender_score = self.calculate_obfuscation_score(sender_addr)
        
        # Якщо обидва мають обфускацію - максимальний бал
        if message_score > 0.3 and sender_score > 0.3:
            combined_score = min(message_score + sender_score, 1.0)
        # Якщо тільки один має обфускацію - високий бал
        elif message_score > 0.3 or sender_score > 0.3:
            combined_score = max(message_score, sender_score) * 0.8
        # Інакше - середнє зважене
        else:
            combined_score = message_score * 0.7 + sender_score * 0.3
        
        return {
            'message_obfuscation': message_score,
            'sender_obfuscation': sender_score,
            'combined_obfuscation': combined_score
        }
    
    def process_csv(self, csv_file: str, output_file: str = None) -> pd.DataFrame:
        """Обробка CSV файлу"""
        print(f"Читання CSV: {csv_file}")
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # Очищення даних
        df['message_text'] = df['message_text'].fillna('').astype(str)
        df['source_addr'] = df['source_addr'].fillna('').astype(str)
        
        # Розрахунок обфускації
        print("Розрахунок обфускації...")
        results = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Оброблено: {idx}/{len(df)}")
            
            result = self.analyze_text_and_sender(
                row['message_text'], row['source_addr']
            )
            results.append(result)
        
        # Додавання результатів
        df['message_obfuscation'] = [r['message_obfuscation'] for r in results]
        df['sender_obfuscation'] = [r['sender_obfuscation'] for r in results]
        df['obfuscation_score'] = [r['combined_obfuscation'] for r in results]
        
        # Статистика
        print(f"\nСтатистика обфускації:")
        print(f"Оброблено записів: {len(df)}")
        print(f"Середній рівень: {df['obfuscation_score'].mean():.3f}")
        print(f"Максимальний: {df['obfuscation_score'].max():.3f}")
        print(f"Повідомлень з обфускацією >0.3: {(df['obfuscation_score'] > 0.3).sum()}")
        print(f"Повідомлень з обфускацією >0.5: {(df['obfuscation_score'] > 0.5).sum()}")
        print(f"Повідомлень з обфускацією >0.7: {(df['obfuscation_score'] > 0.7).sum()}")
        
        # Топ обфускованих повідомлень
        top_obfuscated = df.nlargest(10, 'obfuscation_score')
        print(f"\nТоп-10 обфускованих повідомлень:")
        for idx, row in top_obfuscated.iterrows():
            print(f"  Бал: {row['obfuscation_score']:.3f}")
            print(f"    Текст: '{row['message_text'][:80]}...'")
            print(f"    Відправник: '{row['source_addr']}'")
            print()
        
        # Збереження
        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"Результат збережено: {output_file}")
        
        return df

if __name__ == "__main__":
    # Тестування
    detector = ObfuscationDetector()
    
    # Обробка CSV:
    csv_file = 'data/datasets/smpp_weekly_dataset_20250604_223745.csv'
    output_file = 'data/datasets/smpp_with_obfuscation_v3.csv'
        
    df_result = detector.process_csv(csv_file, output_file)