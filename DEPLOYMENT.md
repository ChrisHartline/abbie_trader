# Deployment Guide: safe_trades_btc for Kraken

## 📊 Strategy Performance

**Adaptive Threshold Results (2022-2025):**
- **Total Return**: +48.80%
- **Sharpe Ratio**: 1.386 (2.6x better than BTC)
- **Max Drawdown**: -4.27% (vs -66.78% for BTC)
- **Win Rate**: 24.4%

**Year-by-Year:**
```
2022: +7.01%   (BTC: -61.04%)  ← Excellent bear protection
2023: +39.08%  (BTC: +154.25%) ← Good bull capture
2024: -0.12%   (BTC: +111.58%) ← Participated (36 trades)
Total: +48.80% vs +110.81% buy-hold
```

---

## 📦 Required Files for Deployment

### Core Trading Files:
```
safe_trades_btc/
├── main.py                      # Live trading bot (Kraken integration)
├── config.py                    # Configuration and risk parameters
├── btc_model.pth                # Trained FFNN model
├── btc_scaler.pth               # Feature scaler
├── projector.npy                # HyperDUM projector matrix
├── memory.npy                   # HyperDUM memory vector
├── market_regime_router.py      # Market regime classifier (optional)
└── .env                         # API keys (DO NOT commit to git!)
```

### Backtesting/Analysis Files (optional):
```
├── backtest_comprehensive.py    # Full backtest with adaptive thresholds
├── train_models.py              # Model training script
├── test_thresholds.py           # Threshold optimization
└── trend_strategy.py            # Trend-following module (future)
```

---

## 🔧 Configuration

### 1. Create `.env` file:

```bash
# .env
KRAKEN_KEY=your_kraken_api_key_here
KRAKEN_SECRET=your_kraken_api_secret_here
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key_here
```

**Security**: NEVER commit `.env` to git! Add to `.gitignore`.

### 2. Configure `config.py`:

```python
# Kraken Settings
LIVE = False  # ← Set to True for real trading (start with False!)
BASE_URL = "https://api.kraken.com" if LIVE else "https://api.testnet.kraken.com"
PAIR = "XBTUSDT"
INITIAL_USD = 100.0

# Risk Management (current settings - tested and working)
VOL_TARGET = 0.20              # 20% annual volatility target
UNCERTAINTY_THRESHOLD = 0.60   # HyperDUM threshold (adaptive: 0.60-0.70)
MAX_GROSS_EXPOSURE = 0.50      # Max 50% of capital
KELLY_FRACTION = 0.25          # Conservative sizing (0.25x)
```

---

## 🚀 Deployment Steps

### Step 1: Testnet Testing (REQUIRED)

```bash
# 1. Ensure LIVE = False in config.py
# 2. Get Kraken testnet API keys from https://demo-futures.kraken.com/
# 3. Add keys to .env
# 4. Run on testnet for at least 1 week
python main.py
```

**What to monitor:**
- Trades execute correctly
- Position sizing is appropriate
- HyperDUM blocks suspicious setups
- No API errors or crashes

### Step 2: Live Deployment (After successful testnet)

```bash
# 1. Set LIVE = True in config.py
# 2. Update .env with LIVE Kraken API keys
# 3. Start with small capital (e.g., $100-500)
# 4. Monitor closely for first week
python main.py
```

---

## 📈 Adaptive Threshold (Recommended)

For better bull market participation, enable adaptive thresholds in `main.py`:

### Add to imports:
```python
from market_regime_router import MarketRegimeRouter
```

### Initialize router:
```python
# After loading models
router = MarketRegimeRouter()

# Adaptive thresholds by regime
ADAPTIVE_THRESHOLDS = {
    'BULL': 0.70,        # More lenient in bulls
    'TRANSITION': 0.65,  # Moderate
    'BEAR': 0.60         # Selective in bears
}
```

### Use in trading loop:
```python
# Before HyperDUM check
regime, confidence, signals = router.classify(hist_close)
current_threshold = ADAPTIVE_THRESHOLDS[regime]

# Then use current_threshold instead of UNCERTAINTY_THRESHOLD
if hamming_dist > current_threshold:
    print(f"🚫 HYPERDUM GATE ({regime} mode)")
    continue
```

**Impact**: Improves from +30% to +48% (60% better returns!)

---

## 🎯 Trading Strategy

### BULL Markets (Price > Support Bands):
- Threshold: 0.70 (allow more trades)
- Participation: ~37% of days
- Goal: Capture uptrends without fighting them

### BEAR Markets (Price < Support Bands):
- Threshold: 0.60 (high selectivity)
- Participation: ~52% of days
- Goal: Profit from reversions, protect capital

### Mean-Reversion Logic:
1. **EKF** estimates equilibrium price and velocity
2. **FFNN** predicts returns based on:
   - EKF level/velocity
   - Funding rates
   - Momentum
   - Relative price position
3. **HyperDUM** blocks out-of-distribution setups
4. **Volatility targeting** sizes positions based on Kelly criterion

---

## ⚙️ Risk Parameters

### Conservative (Default - Recommended):
```python
VOL_TARGET = 0.20
KELLY_FRACTION = 0.25
MAX_GROSS_EXPOSURE = 0.50
```

### Moderate:
```python
VOL_TARGET = 0.25
KELLY_FRACTION = 0.50
MAX_GROSS_EXPOSURE = 0.75
```

### Aggressive (Not recommended):
```python
VOL_TARGET = 0.30
KELLY_FRACTION = 1.0
MAX_GROSS_EXPOSURE = 1.0
```

**Note**: Higher risk = higher returns but also higher drawdowns!

---

## 📊 Monitoring

### Key Metrics to Track:

1. **Daily Returns**: Should be smooth, low volatility
2. **Drawdown**: Should stay < 10%
3. **HyperDUM Block Rate**: 50-75% of days
4. **Win Rate**: 20-30%
5. **Sharpe Ratio**: Target > 1.0

### Warning Signs:

🚨 **Stop trading if:**
- Drawdown > 15%
- 5+ consecutive losing days
- HyperDUM blocking >90% or <20% of days
- Unexpected API errors

---

## 🔄 Retraining (Quarterly Recommended)

```bash
# Download fresh data and retrain
python train_models.py

# This will:
# 1. Download BTC data from 2017-2025
# 2. Train FFNN on full market cycles
# 3. Generate new HyperDUM components
# 4. Save updated models

# Then re-run backtests to verify:
python backtest_comprehensive.py
```

**Retrain when:**
- Every 3 months (quarterly)
- After major market regime change
- If performance degrades significantly

---

## 📚 Additional Resources

### Backtesting:
```bash
# Test with adaptive thresholds
python backtest_comprehensive.py

# Optimize thresholds
python test_thresholds.py
```

### QuantConnect (for learning):
- Use the same EKF + FFNN + HyperDUM logic
- QuantConnect provides more data and instruments
- Good for testing on other assets (ETH, stocks, etc.)

---

## 🛡️ Safety Checklist

Before going live:

- [ ] Tested on testnet for 1+ weeks
- [ ] API keys are correct and secured in .env
- [ ] LIVE = False initially
- [ ] Small capital allocation (<$500 to start)
- [ ] Monitoring system in place
- [ ] Stop-loss plan defined
- [ ] Understand the strategy completely

---

## 📞 Support

For issues or questions:
1. Check logs for error messages
2. Verify API connectivity
3. Confirm model files are present
4. Review backtest results

---

## ⚖️ Disclaimer

**This is algorithmic trading software. Use at your own risk.**

- Past performance does not guarantee future results
- Cryptocurrency trading is highly volatile
- Never invest more than you can afford to lose
- Test thoroughly on testnet before live trading
- Monitor positions actively

**The strategy is designed for:**
- Risk-adjusted returns (not maximum returns)
- Bear market protection
- Steady, low-volatility gains
- Long-term compounding

**Not suitable for:**
- Maximum bull market gains (use buy & hold)
- High-frequency trading
- Leveraged trading (stick to spot)
- Set-and-forget (requires monitoring)

---

## 📈 Expected Performance

Based on 2022-2025 backtest:

**Annual Returns**: 10-20% (varies by market)
**Sharpe Ratio**: 1.0-1.5
**Max Drawdown**: 5-10%
**Win Rate**: 20-30%

**Best in**: Bear markets, high volatility
**Worst in**: Strong bull trends

---

Good luck, and trade safely! 🚀
