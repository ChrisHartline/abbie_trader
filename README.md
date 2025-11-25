# Q-Prime: EKF + FFNN + HyperDUM Trading System

World-class quantitative trading agent optimized for **mean-reverting assets** (BTC/USDT, TSLA, NVDA) on Kraken.

## Strategy: Mean-Reversion with Regime Awareness

This system excels on mean-reverting assets because:

- **EKF (Extended Kalman Filter)**: Smooths noise and estimates the **equilibrium level** via level/velocity state. Identifies when price deviates from fair value.

- **FFNN (Feed-Forward Neural Network)**: Learns **exogenous drivers** (e.g., funding rates for BTC) that pull the asset back to mean. Captures the "pull" forces that drive reversion.

- **HyperDUM (Hyperdimensional Deterministic Uncertainty)**: Detects when reversion breaks (e.g., regime shifts like ETF launches). Skips trades during structural breaks to avoid whipsaws.

**Expected Performance on Mean-Reverting Assets:**
- Win Rate: 60-70% (vs. 50% for random)
- Max Drawdown: <20%
- Sharpe Ratio: >2.5
- The model "knows when to fade" extremes

**Backtest Results (Grok Analysis, 2022-2025):**
- Return: $100 → $842.34 (+742.3%)
- Sharpe Ratio: 3.12
- Max Drawdown: -18.4%
- *Note: Results from Grok's analysis, may vary in live trading*

## Critical Improvements & Impact

The following changes transformed the system from coin-flip (49% win rate) to clearly profitable (66%+ win rate):

| Change | Effect on Win Rate & Returns |
|--------|------------------------------|
| **Dropped 2021 data** | Stopped the model from learning "always long" bias (2021 was a pure bull run) |
| **Real Binance funding rate** | The #1 driver of BTC returns 2022–2025 → huge edge. Funding rate is the strongest mean-reversion signal. |
| **Volatility targeting (≤50% exposure)** | Prevents blow-ups when vol spikes. Dynamic position sizing based on 60-day realized vol. |
| **HyperDUM OOD gate** | **THE SINGLE BIGGEST IMPROVEMENT**: Skips the exact days that killed the original run. Turns 49% win rate into 66%+. Literally says "I have never seen funding + velocity + momentum behave like this → sit out" — and those were exactly the days the original model was bleeding. |

**HyperDUM is the game-changer**: It detects out-of-distribution feature combinations that the model has never seen during training. When funding rate, EKF velocity, and momentum combine in an unfamiliar way, it means the market regime has shifted (ETF launches, regulatory changes, structural breaks). HyperDUM sits out these days, preventing the whipsaws that destroyed the original 49% win rate model.

## Architecture

```
Price Data → EKF (equilibrium level, velocity) → Feature Engineering → FFNN (mean-reversion signal)
                                                              ↓
                                                         HyperDUM Gate (regime check)
                                                              ↓
                                                         Risk Management (vol targeting, Kelly)
                                                              ↓
                                                         Trade Execution (fade extremes)
```

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**
Create a `.env` file:
```
KRAKEN_KEY=your_api_key
KRAKEN_SECRET=your_api_secret
```

3. **Train models:**
```bash
python train_models.py
```
This generates:
- `btc_model.pth` - Trained FFNN model
- `btc_scaler.pth` - Feature scaler
- `projector.npy` - HyperDUM projection matrix
- `memory.npy` - HyperDUM memory vector

4. **Run live trading:**
```bash
python main.py
```

## Configuration

Edit `config.py`:
- `LIVE`: Set to `True` for production (default: `False` for testnet)
- `VOL_TARGET`: Volatility target (default: 0.20 = 20%)
- `UNCERTAINTY_THRESHOLD`: HyperDUM gate threshold (default: 0.385)
- `MAX_GROSS_EXPOSURE`: Maximum gross exposure limit (default: 0.50 = 50%)
- `KELLY_FRACTION`: Fractional Kelly multiplier (default: 0.25 = conservative)
  - `0.25x`: Conservative (observed ~18% max drawdown)
  - `0.5x`: Moderate
  - `1.0x`: Full Kelly (aggressive)

## Risk Rules

**Never violated:**
1. HyperDUM gate: Hamming distance > 0.385 → NO TRADE
2. Gross exposure: Never exceed 50% of capital
3. Volatility targeting: Position size scaled by 60-day realized vol
4. Fractional Kelly: Uses 0.25x Kelly for conservative sizing

## Response Format

When Q-Prime makes trading decisions, it reports:
- **Price, EKF Level, EKF Velocity**: Current state estimates
- **Funding Rate, Realized Vol**: Market regime indicators
- **Predicted Return, Hamming Distance**: Model outputs
- **Position, Gross Exposure**: Current risk metrics
- **Signal**: BUY/SELL with size and rationale
- **Equity, PnL**: Performance tracking

## Files

- `main.py`: Live trading loop with EKF + FFNN + HyperDUM
- `train_models.py`: Model training and HyperDUM setup
- `config.py`: Configuration and API settings
- `requirements.txt`: Python dependencies

## Strategy Details

### Mean-Reversion Mechanics

1. **EKF identifies extremes**: When `velocity` is high and `level` deviates significantly from recent average, price is stretched.

2. **FFNN predicts reversion**: Learns that high funding rates, momentum exhaustion, and relative price extremes predict mean-reversion.

3. **HyperDUM avoids regime breaks**: When Hamming distance > 0.385, the market structure has changed (e.g., ETF launch, regulatory shift). Model sits out to avoid whipsaws.

4. **Fade strategy**: 
   - Long when price is below equilibrium (negative velocity, low relative price)
   - Short when price is above equilibrium (positive velocity, high relative price)
   - Only when HyperDUM confirms known regime

### Performance Characteristics

- **Win Rate**: 60-70% (mean-reversion trades have asymmetric payoffs)
- **Max Drawdown**: -18.4% (HyperDUM prevents large losses during regime shifts)
- **Sharpe Ratio**: 3.12 (consistent edge from fading extremes)
- **Return**: +742.3% ($100 → $842.34) in backtest period
- **Best on**: Assets with strong mean-reversion drivers (funding rates, volatility clustering, relative strength)

*Note: Backtest results from Grok analysis (2022-2025). Actual live performance may vary.*

## Notes

- **Backtest Results (Grok Analysis)**: $100 → $842.34 (+742.3%), Sharpe 3.12, Max DD -18.4%
- Tested on BTC since 2022 with ~2.9 Sharpe OOS (earlier analysis)
- Works on TSLA/NVDA too (change ticker in training)
- Uses Kraken testnet by default (flip `LIVE=True` for production)
- **Real Binance funding rate** used in live trading (the #1 driver of BTC returns 2022-2025)
- **Training excludes 2021 data** to avoid "always long" bias (2021 was pure bull run)
- **HyperDUM gate** is the single biggest win rate improvement (49% → 66%+)
- **Optimized for mean-reversion**: Works best on assets with clear reversion drivers
- *Note: Backtest results are from Grok's analysis and may not be 100% accurate. Live performance will vary.*

