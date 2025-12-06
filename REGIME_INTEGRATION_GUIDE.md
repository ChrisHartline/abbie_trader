# Regime Detection Integration Guide

## What Changed

Your original `main.py` has been enhanced with regime detection and saved as `main_with_regime.py`.

## Key Differences

### Original `main.py`:
- ✓ EKF for equilibrium tracking
- ✓ FFNN for return prediction
- ✓ HyperDUM for out-of-distribution detection
- ✓ Kelly position sizing

### New `main_with_regime.py`:
- ✓ **All of the above PLUS:**
- ✓ **Regime detection** (trending vs mean-reverting)
- ✓ **Regime-based position sizing**
- ✓ **Additional safety layer** on top of HyperDUM

---

## How Regime Detection Works

The bot now checks market conditions before each trade:

### 1. Trend Regime Detection
- **Mean-Reverting (0)**: Price bouncing in a range ✅
- **Trending (1)**: Price moving directionally ⚠️

### 2. Volatility Detection
- **Low Vol (0)**: Normal market conditions
- **High Vol (1)**: Elevated risk

### 3. Stability Check
- **>70%**: Regime is stable, safe to trade
- **<50%**: Regime is transitioning, dangerous

---

## Position Sizing Rules

| Regime Status | Condition | Position Size | Rationale |
|--------------|-----------|---------------|-----------|
| **FAVORABLE** | Mean-reverting + Stable (>70%) | **100%** | Perfect conditions for mean-reversion |
| **NEUTRAL** | Mixed conditions | **75%** | Reduced risk |
| **CAUTION** | Trending + Stable | **50%** | Half size - trend reduces mean-reversion |
| **WARNING** | Unstable (<50% stability) | **0%** | No trading - regime transitioning |

---

## Decision Flow

```
1. Check Regime → Get multiplier (0%, 50%, 75%, or 100%)
   ↓
2. HyperDUM Gate → Check if OOD (out-of-distribution)
   ↓
3. Risk Gate → Check position limits
   ↓
4. Calculate Position → Base target × Regime multiplier
   ↓
5. Execute Trade
```

---

## Example Output

When you run `main_with_regime.py`, you'll see:

```
================================================================================
📊 REGIME ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Regime Status:    FAVORABLE
Trend:            MEAN-REVERTING
Volatility:       LOW-VOL
Stability:        85.00%
Trend Strength:   0.2341
Volatility Level: 45.23%
Position Mult:    100%
✅ FAVORABLE: Mean-reverting regime with high stability
   → Strategy: Full position sizing enabled
────────────────────────────────────────────────────────────────────────────────
🎯 TRADING DECISION
────────────────────────────────────────────────────────────────────────────────
✓ All gates passed
  Base target exposure: 15.00%
  Regime multiplier: 100%
  Final target exposure: 15.00%
```

---

## How to Use

### Option 1: Replace Your Current Bot (Recommended)
```bash
# Backup original
cp main.py main_original.py

# Use regime-enhanced version
cp main_with_regime.py main.py

# Run as normal
python main.py
```

### Option 2: Run Side-by-Side
```bash
# Keep both versions
# Run original: python main.py
# Run enhanced: python main_with_regime.py
```

---

## What This Fixes

### Before (main.py only):
- HyperDUM catches micro-level OOD conditions (specific feature combinations)
- BUT: Doesn't catch macro-level regime shifts
- Example: 2024 ETF launch = trending regime but features look normal
- Result: Model tries to fade trend → loses money

### After (main_with_regime.py):
- HyperDUM catches micro-level OOD ✓
- Regime filter catches macro-level shifts ✓
- Example: 2024 ETF launch = detected as TRENDING regime → position size reduced to 50%
- Result: Reduced exposure during unfavorable periods ✓

---

## Performance Impact

### Expected Improvements:
1. **Lower drawdowns** during trending periods (2x reduction expected)
2. **Higher Sharpe ratio** (smoother equity curve)
3. **Fewer whipsaw losses** during regime transitions
4. **Same or better returns** (only trading in favorable conditions)

### Trade-off:
- **Fewer trades** (by design - only trades when conditions are good)
- **Lower turnover** (good for reducing fees)

---

## Testing Recommendation

1. **Backtest first** on historical data (2022-2025)
2. **Paper trade** for 1-2 weeks to verify behavior
3. **Compare results** with original main.py
4. **Go live** once comfortable with regime-aware behavior

---

## Monitoring

Watch for these regime transitions:
- **FAVORABLE → CAUTION**: Market shifting to trend (reduce size)
- **FAVORABLE → WARNING**: Instability detected (stop trading)
- **WARNING → FAVORABLE**: Regime stabilized (resume trading)

---

## Questions?

- Original bot: `main.py` (no regime detection)
- Enhanced bot: `main_with_regime.py` (with regime detection)
- Regime checker: `regime_detector.py` (standalone analysis tool)

Run `regime_detector.py` anytime to get detailed regime analysis without trading.
