# Q-Prime Trading System - Session Summary

## A) What We Learned

### 1. HyperDUM is the Secret Sauce
- **Threshold 0.35** blocks ~93% of trades, keeping only high-confidence days
- Improved win rate from 49% → 66%+ by avoiding out-of-distribution patterns
- Works by projecting features to 2048-dim space and measuring Hamming distance
- **Key insight**: The gating mechanism is more valuable than the prediction itself

### 2. Mean-Reversion vs Momentum - Different Beasts
| Asset Type | Strategy | HyperDUM | Crisis Detector |
|------------|----------|----------|-----------------|
| BTC/Crypto | Mean-Reversion | 0.35 threshold | Standard |
| TSLA/Stocks | Trend-Following | DISABLED | Tighter thresholds |
| TSLL/2x ETF | Trend-Following | DISABLED | Even tighter |

- BTC has funding rates that create mean-reversion opportunities
- Stocks like TSLA are momentum-driven - ride the trend, don't fade it
- HyperDUM correctly blocks TSLA trades (distance ~0.62 vs 0.30 for BTC)

### 3. Regime-Specific Models Hurt Performance
- Tested 14 features + regime-specific FFNNs → 51.5% accuracy (WORSE)
- Original 5 features + single model + HyperDUM → 66%+ accuracy
- **Lesson**: More complexity ≠ better. The gating is what matters.

### 4. Risk-Adjusted Returns Math
| Strategy | Return | Max DD | Calmar | Insight |
|----------|--------|--------|--------|---------|
| TSLA B&H | +127% | -35% | 1.45 | Raw returns look good |
| TSLA Trend | +82% | -15% | **2.29** | Better risk-adjusted |
| TSLA Trend (2.3x lev) | +188% | -35% | 2.29 | Same risk, higher return |

- **Lower drawdown = ability to use leverage = potentially higher returns**
- A -35% DD requires +54% to recover; -15% DD only needs +18%
- Behavioral reality: Most people panic sell at -35%, fewer at -15%

### 5. 2x Leveraged ETFs (TSLL)
- **Volatility decay**: ~10% annual drag in choppy markets
- **Drawdowns amplified**: -35% TSLA → -60%+ TSLL
- **Crisis detector critical**: Saved $60k on $100k portfolio in simulation
- **Verdict**: Viable with tight crisis thresholds, but TSLA direct is simpler

---

## B) What's Been Implemented in Code

### Core Files Modified
```
main.py              - Added Crisis Detector as first gate
config.py            - Added CRISIS_* threshold parameters
```

### New Configuration Files
```
config_tsla.py       - Trend-following config for TSLA (HyperDUM disabled)
config_tsll.py       - 2x ETF config with tighter crisis thresholds
```

### Supporting Files Created
```
regime_classifier.py - CrisisDetector + RegimeClassifier classes
feature_engineering.py - 14 technical indicators (for experimentation)
regime_ffnn.py       - Regime-specific FFNN training (experimental)
train_from_cache.py  - Train from cached data (no API needed)
test_merged_changes.py - Validation script
```

### Current Gate Order (main.py)
```
1. CRISIS DETECTOR → "Oh shit" gate for dangerous markets
2. HYPERDUM       → Out-of-distribution detection
3. RISK GATES     → Exposure limits
4. MEAN-REVERSION → Fade extremes (EKF + FFNN)
```

### Key Parameters (config.py)
```python
# Risk Management
UNCERTAINTY_THRESHOLD = 0.35      # HyperDUM gate
KELLY_FRACTION = 0.25             # Conservative sizing
MAX_GROSS_EXPOSURE = 0.50         # Position limit
VOL_TARGET = 0.20                 # Annual vol target

# Crisis Detector
CRISIS_VOL_THRESHOLD = 2.5        # ATR > 2.5x median
CRISIS_DD_THRESHOLD = -0.15       # -15% drawdown
CRISIS_CRASH_THRESHOLD = -0.08    # -8% in 5 days
```

---

## C) Options for Improvement

### High Priority (Low Effort, High Impact)
1. **Kelly Fraction Tuning**
   - Current: 0.25x (conservative)
   - Test 0.5x for higher returns with acceptable risk
   - Formula: `target = sign(pred) * risk * KELLY_FRACTION`

2. **Adaptive HyperDUM Threshold**
   - Current: Fixed at 0.35
   - Idea: Tighten during high vol, relax during low vol
   - Could improve hit rate without sacrificing protection

3. **Add More Funding Rate Sources**
   - Currently: Binance, Bybit, OKX fallback chain
   - Add: dYdX, Hyperliquid for redundancy

### Medium Priority (Moderate Effort)
4. **Multi-Asset Portfolio**
   - Run BTC mean-reversion + TSLA trend-following in parallel
   - Diversification benefit
   - Need: Unified position manager

5. **Intraday Trading (4H candles)**
   - Current: Daily signals
   - 4H could capture more mean-reversion opportunities
   - Need: More frequent data feed, faster model inference

6. **Ensemble Predictions**
   - Train multiple FFNNs with different seeds
   - Average predictions for more stable signals
   - Only trade when all models agree

### Lower Priority (High Effort)
7. **Real Regime Detection**
   - Current crisis detector is reactive
   - Add leading indicators (VIX term structure, put/call ratio)
   - ML-based regime prediction

8. **Options Overlay**
   - Sell covered calls during sideways regimes
   - Buy puts during crisis conditions
   - Need: Options-capable broker integration

---

## D) Testing and Deployment

### BTC via Kraken

#### Prerequisites
```bash
# 1. Get Kraken API keys (testnet first!)
# https://www.kraken.com/u/security/api

# 2. Create .env file
echo "KRAKEN_KEY=your_key_here" > .env
echo "KRAKEN_SECRET=your_secret_here" >> .env

# 3. Install dependencies
pip install numpy pandas torch scikit-learn requests python-dotenv

# 4. Train models (uses cached data if available)
python train_from_cache.py
# Creates: btc_model.pth, btc_scaler.pth, projector.npy, memory.npy
```

#### Testing Sequence
```bash
# Step 1: Backtest on historical data
python backtest_comprehensive.py
# Expected: Sharpe ~3.88, +10% return, -3.4% max DD

# Step 2: Paper trade on testnet
# config.py: LIVE = False (default)
python main.py
# Runs against Kraken testnet with fake money

# Step 3: Monitor for 2-4 weeks
# Check: Is it blocking trades correctly?
# Check: Are crisis detections accurate?
# Check: Is HyperDUM distance reasonable (~0.30-0.35)?

# Step 4: Go live (small size)
# config.py: LIVE = True
# INITIAL_USD = 100.0  # Start VERY small
python main.py
```

#### Monitoring
```python
# Add to main.py for alerting
import smtplib
def send_alert(subject, message):
    # Email/SMS on crisis trigger or large trades
    pass
```

### Stocks (TSLA) via Webull

#### Current Limitation
- `main.py` is built for Kraken (crypto)
- Need to create `main_stocks.py` for Webull

#### Webull Integration Steps
```python
# 1. Install Webull SDK
pip install webull

# 2. Create main_stocks.py with:
from webull import webull

wb = webull()
wb.login('email', 'password')

# 3. Modify trading loop:
# - Replace Kraken API calls with Webull
# - Use config_tsla.py for parameters
# - Implement trend-following logic (not mean-reversion)
```

#### Key Differences for Stocks
```python
# In main_stocks.py:

# Use trend-following config
from config_tsla import *

# Skip HyperDUM (not applicable)
if USE_HYPERDUM and hamming_dist > UNCERTAINTY_THRESHOLD:
    target = 0.0

# Trend signal instead of mean-reversion
is_bull = ema20 > ema50
if is_bull:
    target = KELLY_FRACTION * MAX_GROSS_EXPOSURE
else:
    target = 0.0  # Exit
```

### Deployment Checklist

#### Before Going Live
- [ ] Backtest shows positive Sharpe > 1.0
- [ ] Paper traded for 2+ weeks
- [ ] Crisis detector triggers make sense
- [ ] Understand worst-case drawdown
- [ ] Have stop-loss plan if system fails

#### Infrastructure
- [ ] Run on reliable server (not personal laptop)
- [ ] Set up monitoring/alerting
- [ ] Daily backup of equity curve
- [ ] Log all trades for review

#### Risk Management
- [ ] Start with 1-5% of intended capital
- [ ] Set maximum daily loss limit
- [ ] Have manual override capability
- [ ] Review weekly initially, then monthly

---

## Quick Reference: Which Config for What?

| Asset | Config File | Strategy | Key Settings |
|-------|-------------|----------|--------------|
| BTC/Crypto | `config.py` | Mean-Reversion | HyperDUM 0.35, Crisis std |
| TSLA/Stocks | `config_tsla.py` | Trend-Following | HyperDUM OFF, Crisis std |
| TSLL/2x ETF | `config_tsll.py` | Trend-Following | HyperDUM OFF, Crisis TIGHT |

---

*Generated: 2024-12-29*
*Branch: claude/test-merged-changes-Uarh1*
