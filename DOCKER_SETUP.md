# Docker Setup Guide

## Quick Start

### 1. Prerequisites

- Docker Desktop installed ([download](https://www.docker.com/products/docker-desktop/))
- Git (to clone the repo)

### 2. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url> abbie_trader
cd abbie_trader

# Create environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use any text editor
```

### 3. Create .env File

```bash
# .env file contents

# Kraken (for BTC)
KRAKEN_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_api_secret

# Webull (for stocks) - optional
WEBULL_EMAIL=your_email
WEBULL_PASSWORD=your_password
WEBULL_TRADE_PIN=your_trade_pin

# Mode (false = paper trade, true = live)
LIVE=false
```

### 4. Build and Run

```bash
# Build the Docker image
docker-compose build

# Run BTC mean-reversion (paper trading)
docker-compose up btc

# Run TSLA trend-following (paper trading)
docker-compose up tsla

# Run 2x ETF with tighter crisis thresholds
docker-compose up tsll
```

### 5. Training and Backtesting

```bash
# Train models from cached data
docker-compose --profile training up train

# Run comprehensive backtest
docker-compose --profile testing up backtest
```

---

## Detailed Instructions

### Running Without Docker (Local Python)

If you prefer running locally without Docker:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
echo "KRAKEN_KEY=your_key" > .env
echo "KRAKEN_SECRET=your_secret" >> .env

# Train models (first time only)
python train_from_cache.py

# Run backtest to verify
python backtest_comprehensive.py

# Run paper trading
python main_mean.py        # BTC
python main_trend.py       # TSLA
```

### Running in Production

For production deployment:

```bash
# 1. Set LIVE=true in .env
echo "LIVE=true" >> .env

# 2. Start with very small capital first
# Edit config.py: INITIAL_USD = 100.0

# 3. Run in detached mode
docker-compose up -d btc

# 4. View logs
docker-compose logs -f btc

# 5. Stop
docker-compose down
```

### Monitoring

```bash
# View real-time logs
docker-compose logs -f btc

# Check container status
docker-compose ps

# Enter container shell (for debugging)
docker-compose exec btc bash
```

---

## File Structure

```
abbie_trader/
├── Dockerfile              # Docker build instructions
├── docker-compose.yml      # Service definitions
├── requirements.txt        # Python dependencies
├── .env                    # API keys (create this!)
│
├── main_mean.py           # BTC mean-reversion (Kraken)
├── main_trend.py          # TSLA trend-following (Webull)
│
├── config.py              # BTC configuration
├── config_tsla.py         # TSLA configuration
├── config_tsll.py         # 2x ETF configuration
│
├── train_from_cache.py    # Model training
├── backtest_comprehensive.py  # Backtesting
│
├── btc_model.pth          # Trained FFNN model
├── btc_scaler.pth         # Feature scaler
├── projector.npy          # HyperDUM projector
├── memory.npy             # HyperDUM memory vector
│
└── regime_analysis.csv    # Cached training data
```

---

## Troubleshooting

### "Model files not found"
```bash
# Train the models first
python train_from_cache.py
```

### "API connection failed"
- Check your API keys in `.env`
- For Kraken: Ensure testnet keys for paper trading
- For Webull: Ensure 2FA is set up correctly

### "Insufficient data"
- The system needs 60+ days of historical data
- Wait for data to accumulate or use cached data

### Container keeps restarting
```bash
# Check logs for errors
docker-compose logs btc

# Common issues:
# - Missing .env file
# - Invalid API keys
# - Network connectivity
```

---

## Security Notes

1. **Never commit `.env` file** to git
2. **Use testnet first** before going live
3. **Start with minimal capital** ($100) when going live
4. **Monitor regularly** during first few weeks
5. **Set up alerts** for crisis triggers and large trades

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker-compose up btc` | Run BTC mean-reversion |
| `docker-compose up tsla` | Run TSLA trend-following |
| `docker-compose up -d btc` | Run in background |
| `docker-compose logs -f btc` | View live logs |
| `docker-compose down` | Stop all services |
| `docker-compose build` | Rebuild after code changes |
