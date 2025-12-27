# Backtesting Guide

## How to Run the Backtest

### Prerequisites
1. Trained model files must exist:
   - `btc_model.pth`
   - `btc_scaler.pth`
   - `projector.npy`
   - `memory.npy`

2. If you don't have these, run first:
   ```bash
   python train_models.py
   ```

### Run the Backtest

```bash
python backtest_comprehensive.py
```

This will:
- Download BTC historical data from 2022-01-01 to present
- Run the exact trading logic from `main.py`
- Generate detailed performance metrics
- Save results to CSV files

---

## What It Tests

The backtest uses **identical logic** to your live trading bot (`main.py`):

1. **EKF** - Equilibrium level and velocity estimation
2. **FFNN** - Return prediction based on features
3. **HyperDUM** - Out-of-distribution detection
4. **Kelly Sizing** - Volatility-adjusted position sizing
5. **Risk Gates** - Maximum exposure limits

---

## Output Metrics

### Performance Metrics
- **Total Return**: Strategy vs Buy & Hold
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return / Max Drawdown
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable days
- **Volatility**: Annualized standard deviation

### Trade Statistics
- Total number of trades
- Days with active positions
- Days skipped by HyperDUM
- Days skipped by risk gates
- Buy vs Sell trade breakdown

### Year-by-Year Performance
- Annual returns
- Annual Sharpe ratio
- Number of trades per year
- Comparison with BTC buy-and-hold

### HyperDUM Effectiveness
- Average return on days HyperDUM allowed trading
- Average BTC return on days HyperDUM blocked
- Shows how many risky days were avoided

---

## Output Files

### backtest_results.csv
Daily equity curve with columns:
- `date`: Trading date
- `equity`: Portfolio value
- `position`: BTC position size
- `price`: BTC price
- `hamming`: HyperDUM uncertainty score
- `pred`: Model prediction
- `skip_reason`: Why trade was skipped (if any)
- `target`: Target position
- `equity_return`: Daily strategy return
- `btc_return`: Daily BTC return
- `strategy_cumulative`: Cumulative strategy returns
- `btc_cumulative`: Cumulative BTC returns

### backtest_trades.csv
Trade log with columns:
- `date`: Trade date
- `side`: BUY or SELL
- `size`: Trade size in BTC
- `price`: Execution price
- `pred`: Model prediction at time of trade
- `hamming`: HyperDUM score
- `volatility`: Market volatility

---

## Interpreting Results

### Good Performance Indicators
- ✅ **Sharpe > 1.0**: Strong risk-adjusted returns
- ✅ **Positive Calmar**: Return exceeds max drawdown
- ✅ **Max DD < 30%**: Manageable risk
- ✅ **Win Rate > 50%**: More winning days than losing
- ✅ **Return > BTC**: Outperforming buy-and-hold

### Warning Signs
- ⚠️ **Sharpe < 0.5**: Poor risk-adjusted returns
- ⚠️ **Max DD > 50%**: Excessive risk
- ⚠️ **Win Rate < 45%**: Too many losing days
- ⚠️ **Few trades**: HyperDUM might be too conservative
- ⚠️ **Too many trades**: Risk gates might be too loose

---

## Adjusting Parameters

If results are suboptimal, you can adjust these in the backtest script:

### UNCERTAINTY_THRESHOLD (default: 0.35)
- **Lower (0.30-0.33)**: Trade more often, higher risk
- **Higher (0.40-0.50)**: Trade less often, lower risk
- Affects how often HyperDUM blocks trades

### REGIME STABILITY (default: favorable ≥0.90, warning <0.75)
- **Raise favorable**: Demand more stability before full sizing
- **Raise warning**: Shut down faster when regimes churn
- Applied after HyperDUM, before risk sizing

### VOL_TARGET (default: 0.20)
- **Lower (0.10-0.15)**: Smaller positions, lower risk
- **Higher (0.25-0.30)**: Larger positions, higher risk
- Target annualized volatility

### KELLY_FRACTION (default: 0.5)
- **Lower (0.25-0.40)**: More conservative sizing
- **Higher (0.60-0.75)**: More aggressive sizing
- Fraction of Kelly criterion to use

### MAX_GROSS_EXPOSURE (default: 0.50)
- **Lower (0.30-0.40)**: Maximum position size limit
- **Higher (0.60-0.80)**: Allow larger positions
- Hard cap on position size

---

## Comparing with Buy-and-Hold

The backtest automatically compares with BTC buy-and-hold:

**When Strategy Wins:**
- Higher Sharpe (better risk-adjusted returns)
- Lower Max Drawdown (smoother equity curve)
- Similar or better total return

**When Buy-and-Hold Wins:**
- Pure bull market (trending up)
- Strategy sits out too much (HyperDUM too strict)

**Ideal Scenario:**
- Strategy Sharpe > BTC Sharpe
- Strategy Max DD < BTC Max DD
- Strategy Return ≈ BTC Return (or better)

This means: Similar returns with less risk = better performance

---

## Example Output

```
================================================================================
BACKTEST RESULTS
================================================================================

Period: 2022-01-01 to 2025-11-26
Trading Days: 1425

────────────────────────────────────────────────────────────────────────────────
PERFORMANCE METRICS
────────────────────────────────────────────────────────────────────────────────
Metric                         Strategy     Buy & Hold
────────────────────────────────────────────────────────────────────────────────
Total Return                      +45.23%        +38.12%
Sharpe Ratio                        1.234          0.856
Sortino Ratio                       1.845            N/A
Calmar Ratio                        1.521            N/A
Max Drawdown                      -29.73%        -47.23%
Win Rate                            54.2%            N/A
Volatility (Annual)                 36.7%          44.5%

────────────────────────────────────────────────────────────────────────────────
TRADE STATISTICS
────────────────────────────────────────────────────────────────────────────────
Total Trades:             342
Days with Position:       456 (32.0%)
Days Skipped (HyperDUM):  523 (36.7%)
Days Skipped (Risk Gate): 12 (0.8%)

────────────────────────────────────────────────────────────────────────────────
HYPERDUM EFFECTIVENESS
────────────────────────────────────────────────────────────────────────────────
Avg daily return (HyperDUM allowed): +0.0234%
Avg BTC return (HyperDUM blocked): -0.0123%
Days blocked: 523

✓ HyperDUM protected from 523 potentially risky days
```

---

## Troubleshooting

### "Model files not found"
Run `python train_models.py` first

### "No trades taken"
UNCERTAINTY_THRESHOLD is too low - HyperDUM is blocking everything
Try increasing to 0.40 or 0.45

### "Poor performance vs buy-and-hold"
- Check if it's a trending bull market (strategy favors mean-reversion)
- Consider adjusting KELLY_FRACTION or VOL_TARGET
- Review HyperDUM effectiveness section

### "Too many trades"
Position sizing might be too sensitive
Try reducing VOL_TARGET or KELLY_FRACTION

---

## Next Steps

After reviewing backtest results:

1. **If performance is good**: Deploy to paper trading
2. **If performance is poor**: Adjust parameters and re-run
3. **Compare periods**: Check year-by-year to see when strategy works best
4. **Analyze trades**: Review backtest_trades.csv for patterns

Remember: Past performance doesn't guarantee future results, but it helps validate the strategy logic.
