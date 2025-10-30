FROM python:3.9
WORKDIR /app
# Устанавливаем системные зависимости для matplotlib, plotly и scipy
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
# Копируем и устанавливаем зависимости из requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt