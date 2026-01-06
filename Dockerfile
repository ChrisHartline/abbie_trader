# Q-Prime Trading Bot - EKF + FFNN + HyperDUM
# Mean-reversion strategy for BTC/stocks

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables (override at runtime)
ENV KRAKEN_KEY=""
ENV KRAKEN_SECRET=""
ENV LIVE="false"
ENV PYTHONUNBUFFERED=1

# Default command - run the main trading bot
CMD ["python", "main.py"]
