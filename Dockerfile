# Q-Prime Trading System
# Multi-stage build for both mean-reversion (crypto) and trend-following (stocks)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY *.npy ./  2>/dev/null || true
COPY *.pth ./  2>/dev/null || true
COPY *.csv ./  2>/dev/null || true

# Create directory for logs and data
RUN mkdir -p /app/logs /app/data

# Environment variables (override in docker-compose or runtime)
ENV PYTHONUNBUFFERED=1
ENV LIVE=false

# Default command (can be overridden)
CMD ["python", "main_mean.py"]
