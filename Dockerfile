# 1. Базовый образ: лёгкий Linux с Python 3.13 (как в нашем .venv)
FROM python:3.13-slim

# 2. Рабочая папка внутри контейнера
WORKDIR /app

# 3. Сначала только зависимости — Docker закеширует этот слой,
#    и при правке кода пакеты не будут переустанавливаться
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Потом весь код проекта (лишнее отсекает .dockerignore)
COPY . .

# 5. Порт, на котором слушает сервис
EXPOSE 8000

# 6. Команда запуска. 0.0.0.0 — принимать запросы снаружи контейнера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
