#!/bin/bash
# docker-scripts.sh - Скрипти для управління Docker контейнерами

# Функція для виведення допомоги
show_help() {
    echo "Використання: ./docker-scripts.sh [КОМАНДА]"
    echo ""
    echo "КОМАНДИ:"
    echo "  build       - Побудувати Docker образ"
    echo "  start       - Запустити dashboard"
    echo "  stop        - Зупинити всі сервіси"
    echo "  restart     - Перезапустити dashboard"
    echo "  logs        - Показати логи dashboard"
    echo ""
    echo "PIPELINE (послідовна обробка даних):"
    echo "  pipeline    - Запустити повний pipeline (1→2→3→4)"
    echo "  generate    - 1. Генерація SMPP PDU"
    echo "  parse       - 2. Парсинг PDU в повідомлення"
    echo "  extract     - 3. Екстракція ознак"
    echo "  detect      - 4. Виявлення аномалій"
    echo ""
    echo "УТИЛІТИ:"
    echo "  shell       - Відкрити shell в dashboard контейнері"
    echo "  clean       - Очистити всі контейнери та образи"
    echo "  status      - Показати статус контейнерів"
    echo "  quick       - Швидкий старт (build + start)"
    echo "  full        - Повний цикл (pipeline + dashboard)"
    echo "  help        - Показати цю допомогу"
}

# Функція для перевірки Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker не встановлено!"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose не встановлено!"
        exit 1
    fi
}

# Побудувати образ
build() {
    echo "🔨 Побудова Docker образу..."
    docker-compose build --no-cache
    echo "✅ Образ побудовано!"
}

# Запустити основний сервіс (dashboard)
start() {
    echo "🚀 Запуск SMPP Dashboard..."
    docker-compose up -d dashboard
    echo "✅ Dashboard запущено!"
    echo "📊 Dashboard доступний на: http://localhost:8501"
}

# Зупинити сервіси
stop() {
    echo "🛑 Зупинка всіх сервісів..."
    docker-compose down
    echo "✅ Сервіси зупинено!"
}

# Перезапуск
restart() {
    echo "🔄 Перезапуск dashboard..."
    stop
    sleep 2
    start
}

# Показати логи
logs() {
    echo "📝 Логи dashboard:"
    docker-compose logs -f dashboard
}

# Запуск повного pipeline обробки даних
pipeline() {
    echo "🚀 Запуск повного SMPP pipeline..."
    
    echo "1️⃣ Генерація SMPP PDU..."
    docker-compose --profile pipeline up data-generator
    
    echo "2️⃣ Парсинг PDU в повідомлення..."
    docker-compose --profile pipeline up pdu-parser
    
    echo "3️⃣ Екстракція ознак..."
    docker-compose --profile pipeline up feature-extractor
    
    echo "4️⃣ Виявлення аномалій..."
    docker-compose --profile pipeline up anomaly-detector
    
    echo "✅ Pipeline завершено! Дані оновлено."
}

# Окремі етапи pipeline
generate() {
    echo "📊 Генерація SMPP PDU..."
    docker-compose --profile pipeline up data-generator
    echo "✅ Генерація завершена!"
}

parse() {
    echo "🔍 Парсинг PDU..."
    docker-compose --profile pipeline up pdu-parser
    echo "✅ Парсинг завершений!"
}

extract() {
    echo "🔧 Екстракція ознак..."
    docker-compose --profile pipeline up feature-extractor
    echo "✅ Екстракція завершена!"
}

detect() {
    echo "🎯 Виявлення аномалій..."
    docker-compose --profile pipeline up anomaly-detector
    echo "✅ Виявлення завершено!"
}

# Shell в dashboard контейнері
shell() {
    echo "🐚 Відкриття shell в dashboard контейнері..."
    docker-compose exec dashboard /bin/bash
}

# Очистити все
clean() {
    echo "🧹 Очищення Docker ресурсів..."
    docker-compose down --rmi all --volumes --remove-orphans
    docker system prune -f
    echo "✅ Очищення завершено!"
}

# Показати статус
status() {
    echo "📊 Статус контейнерів:"
    docker-compose ps
    echo ""
    echo "📊 Використання ресурсів:"
    docker stats --no-stream
}

# Швидкий старт (тільки dashboard)
quick_start() {
    echo "⚡ Швидкий старт dashboard..."
    build
    start
    echo "📊 Dashboard: http://localhost:8501"
}

# Повний цикл (pipeline + dashboard)
full() {
    echo "🔄 Повний цикл: pipeline + dashboard..."
    pipeline
    echo "5️⃣ Запуск dashboard..."
    start
    echo "✅ Система готова!"
    echo "📊 Dashboard: http://localhost:8501"
}

# Основна логіка
case "$1" in
    "build")
        check_docker
        build
        ;;
    "start")
        check_docker
        start
        ;;
    "stop")
        check_docker
        stop
        ;;
    "restart")
        check_docker
        restart
        ;;
    "logs")
        check_docker
        logs
        ;;
    "generate")
        check_docker
        generate
        ;;
    "parse")
        check_docker
        parse
        ;;
    "extract")
        check_docker
        extract
        ;;
    "detect")
        check_docker
        detect
        ;;
    "pipeline")
        check_docker
        pipeline
        ;;
    "shell")
        check_docker
        shell
        ;;
    "clean")
        check_docker
        clean
        ;;
    "status")
        check_docker
        status
        ;;
    "full")
        check_docker
        full
        ;;
    "quick")
        check_docker
        quick_start
        ;;
    "help"|*)
        show_help
        ;;
esac