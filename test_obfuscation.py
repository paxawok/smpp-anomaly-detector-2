from src.detection.obfuscation.obfuscation_detector import ObfuscationDetector


def print_score_details(text: str, score: float, label: str):
    print(f"\n--- {label} ---")
    print(f"Текст: {text}")
    print(f"Скор: {score:.3f}")


def test_mixed_script_obfuscation_in_message():
    detector = ObfuscationDetector()
    text = "Зaйдiть на 0nlіnebank. Ваш k0д: 1234"
    score = detector.calculate_obfuscation_score(text)

    print_score_details(text, score, "Обфускація змішаних скриптів у повідомленні")
    assert score > 0.2, "Має бути виявлено змішання латиниці й кирилиці з цифрами"


def test_clean_sender_should_be_ignored():
    detector = ObfuscationDetector()
    sender = "NovaPoshta"
    score = detector.calculate_obfuscation_score(sender)

    print_score_details(sender, score, "Чистий відправник")
    assert score == 0.0, "Чистий бренд не має викликати обфускацію"


def test_digit_obfuscation_in_sender():
    detector = ObfuscationDetector()
    sender = "pr1vаtbаnk24"  # підміна: 1 → l, а + цифри
    score = detector.calculate_obfuscation_score(sender)

    print_score_details(sender, score, "Цифрова обфускація у sender")
    assert score > 0.2, "Цифри всередині імітації бренду мають викликати обфускацію"


def test_invisible_chars_in_sender():
    detector = ObfuscationDetector()
    sender = "gоv\u200Bukr"  # невидимий символ + кирилична 'о'
    score = detector.calculate_obfuscation_score(sender)

    print_score_details(sender, score, "Invisible char у sender")
    assert score > 0.3, "Невидимий символ має підняти скор при короткому sender'і"


def test_both_sender_and_message_obfuscated():
    detector = ObfuscationDetector()
    message = "Vаsh cod: 1111. Nе передавайте!"
    sender = "0nlinePay"

    result = detector.analyze_text_and_sender(message, sender)

    print(f"\n--- Сумарна перевірка ---")
    print(f"Текст: {message}")
    print(f"Sender: {sender}")
    print(f"Message score: {result['message_obfuscation']:.3f}")
    print(f"Sender score: {result['sender_obfuscation']:.3f}")
    print(f"Combined score: {result['combined_obfuscation']:.3f}")

    assert result['message_obfuscation'] > 0.02, "Обфускація у тексті має бути виявлена"
    assert result['sender_obfuscation'] > 0.02, "Цифровий sender має бути обфускований"
    assert result['combined_obfuscation'] > 0.3, "Загальний скор має бути високим при обох обфускаціях"


def test_false_positive_simple_code():
    detector = ObfuscationDetector()
    text = "Ваш код: 123456. Не передавайте стороннім"
    score = detector.calculate_obfuscation_score(text)

    print_score_details(text, score, "Тест на false positive для простого коду")
    assert score < 0.1, "Код без обфускації не має викликати хибне спрацьовування"

if __name__ == "__main__":
    test_mixed_script_obfuscation_in_message()
    test_clean_sender_should_be_ignored()
    test_digit_obfuscation_in_sender()
    test_invisible_chars_in_sender()
    test_both_sender_and_message_obfuscated()
    test_false_positive_simple_code()
    print("Всі тести пройдено успішно!")