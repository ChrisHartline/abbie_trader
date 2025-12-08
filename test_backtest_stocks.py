"""
Test Backtest Script for Stock Models with Synthetic Data

Uses synthetic data to verify the backtesting pipeline works correctly.
This can be used when network access is restricted.

Usage:
    python test_backtest_stocks.py NVDA      # Test NVIDIA
    python test_backtest_stocks.py TSLA      # Test Tesla
    python test_backtest_stocks.py NVDA TSLA # Test multiple
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# FFNN Model Definition (needed for loading saved models)
# -----------------------------
class FFNN(nn.Module):
    def __init__(self, input_size=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        return self.net(x)

# Default tickers if none specified
DEFAULT_TICKERS = ['NVDA', 'TSLA']

if len(sys.argv) > 1:
    TICKERS = [t.upper() for t in sys.argv[1:]]
else:
    TICKERS = DEFAULT_TICKERS

# Risk parameters
VOL_TARGET = 0.20
UNCERTAINTY_THRESHOLD = 0.385
MAX_GROSS_EXPOSURE = 0.50
KELLY_FRACTION = 0.25

# -----------------------------
# 1. Extended Kalman Filter (EKF)
# -----------------------------
def run_ekf(price_series, dt=1.0):
    """Extended Kalman Filter for state estimation"""
    if isinstance(price_series, pd.Series):
        values = price_series.values
        index = price_series.index
    else:
        values = np.array(price_series)
        index = pd.RangeIndex(len(values))

    if len(values) == 0:
        raise ValueError("Cannot run EKF on empty price series.")

    n = len(values)
    x = np.zeros((n, 3))
    P = np.zeros((n, 3, 3))

    x[0] = [values[0], 0.0, -5.0]
    P[0] = np.eye(3) * 1.0

    Q = np.diag([0.01, 1e-4, 1e-4])
    R = 0.5

    smoothed = np.zeros(n)

    for t in range(1, n):
        F = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1]])
        x_pred = F @ x[t-1]
        P_pred = F @ P[t-1] @ F.T + Q

        y = values[t] - x_pred[0]
        S = P_pred[0,0] + R
        K = P_pred[:,0] / S

        x[t] = x_pred + K * y
        P[t] = (np.eye(3) - np.outer(K, np.array([1,0,0]))) @ P_pred
        smoothed[t] = x[t][0]

    level = pd.Series(smoothed, index=index)
    velocity = pd.Series(x[:,1], index=index)
    return level, velocity

# -----------------------------
# 2. Generate Synthetic Stock Data
# -----------------------------
def generate_synthetic_data(ticker, n_days=1000):
    """Generate realistic synthetic stock data for testing"""
    np.random.seed(hash(ticker) % 2**32)

    stock_params = {
        'NVDA': {'start_price': 200, 'drift': 0.0008, 'volatility': 0.03, 'mean_rev': 0.02},
        'TSLA': {'start_price': 250, 'drift': 0.0005, 'volatility': 0.04, 'mean_rev': 0.015},
        'AAPL': {'start_price': 150, 'drift': 0.0003, 'volatility': 0.02, 'mean_rev': 0.025},
    }

    params = stock_params.get(ticker, {'start_price': 100, 'drift': 0.0004, 'volatility': 0.025, 'mean_rev': 0.02})

    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)

    prices = np.zeros(n)
    prices[0] = params['start_price']
    equilibrium = params['start_price']

    for i in range(1, n):
        mr_force = -params['mean_rev'] * (prices[i-1] - equilibrium) / equilibrium
        drift = params['drift']
        shock = np.random.normal(0, params['volatility'])
        ret = drift + mr_force + shock
        prices[i] = prices[i-1] * (1 + ret)
        equilibrium *= (1 + params['drift'] * 0.5)

    vix = 20 + np.cumsum(np.random.normal(0, 1, n)) * 0.5
    vix = np.clip(vix, 10, 50)

    df = pd.DataFrame({'Close': prices}, index=dates)
    vix_series = pd.Series(vix, index=dates)

    return df, vix_series

# -----------------------------
# 3. Run Backtest
# -----------------------------
def run_backtest(ticker):
    """Run backtest for a single ticker using synthetic data"""
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {ticker} (SYNTHETIC DATA)")
    print(f"{'='*70}")

    ticker_clean = ticker.replace('-', '_').lower()

    # Check if model files exist
    model_files = [f'{ticker_clean}_model.pth', f'{ticker_clean}_scaler.pth',
                   f'{ticker_clean}_projector.npy', f'{ticker_clean}_memory.npy']

    missing = [f for f in model_files if not os.path.exists(f)]
    if missing:
        print(f"  Missing model files: {missing}")
        print(f"  Run 'python test_stock_models.py {ticker}' first.")
        return None

    # Load models
    print("  Loading models...")
    model = torch.load(f'{ticker_clean}_model.pth', weights_only=False)
    scaler = torch.load(f'{ticker_clean}_scaler.pth', weights_only=False)
    projector = np.load(f'{ticker_clean}_projector.npy')
    memory_vector = np.load(f'{ticker_clean}_memory.npy')

    # Generate synthetic data
    print("  Generating synthetic data...")
    df, vix_series = generate_synthetic_data(ticker, n_days=1500)

    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['sentiment'] = ((vix_series - 20) / 20).clip(-1, 1)
    df = df.dropna()

    # Run EKF
    df['level'], df['velocity'] = run_ekf(df['Close'])

    # Prepare features
    window = 30
    momentum_window = 5

    features = []
    for i in range(window, len(df)):
        row = [
            df['level'].iloc[i],
            df['velocity'].iloc[i],
            df['sentiment'].iloc[i],
            df['return'].iloc[i-momentum_window+1:i+1].mean(),
            df['Close'].iloc[i] / df['Close'].iloc[i-window:i].mean() - 1
        ]
        features.append(row)

    X = np.array(features)
    X_scaled = scaler.transform(X)

    # Run backtest
    print("  Running backtest simulation...")

    initial_equity = 10000.0
    equity = initial_equity
    position_exposure = 0.0
    trades_taken = 0
    trades_skipped = 0

    equity_curve = [initial_equity]
    returns_list = []
    dates_list = []

    model.eval()

    for i in range(len(X_scaled)):
        idx = window + i
        if idx >= len(df) - 1:
            break

        # Get prediction
        feat_t = torch.FloatTensor(X_scaled[i:i+1])
        with torch.no_grad():
            pred = model(feat_t).item()

        actual_return = df['return'].iloc[idx + 1]

        # HyperDUM uncertainty check
        projected = np.sign(X_scaled[i:i+1] @ projector)
        hamming_dist = np.mean(projected != memory_vector)

        # Calculate volatility
        lookback = max(0, idx - 60)
        recent_vol = df['return'].iloc[lookback:idx].std() * np.sqrt(252)
        if recent_vol < 0.01:
            recent_vol = 0.20

        # Position sizing
        target_exposure = 0.0
        if hamming_dist > UNCERTAINTY_THRESHOLD:
            trades_skipped += 1
        else:
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            target_exposure = np.sign(pred) * risk * KELLY_FRACTION
            if abs(target_exposure - position_exposure) > 0.001:
                trades_taken += 1

        # Calculate period return
        period_return = position_exposure * actual_return
        equity *= (1 + period_return)
        equity_curve.append(equity)
        returns_list.append(period_return)
        dates_list.append(df.index[idx + 1])
        position_exposure = target_exposure

    # Calculate metrics
    returns_array = np.array(returns_list)
    total_return = (equity / initial_equity - 1) * 100
    sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min() * 100

    # Win rate
    win_rate = np.mean(returns_array > 0) if len(returns_array) > 0 else 0

    # Sortino ratio
    neg_returns = returns_array[returns_array < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 0.01
    sortino = np.mean(returns_array) / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # Buy and hold comparison
    bh_return = (df['Close'].iloc[-1] / df['Close'].iloc[window] - 1) * 100

    # Print results
    print(f"\n  {'='*60}")
    print(f"  {ticker} BACKTEST RESULTS (SYNTHETIC DATA)")
    print(f"  {'='*60}")
    print(f"\n  Period: {df.index[window]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")
    print(f"  Total Days: {len(returns_list)}")
    print(f"\n  Performance Metrics:")
    print(f"    Total Return:      {total_return:+.2f}%")
    print(f"    Buy & Hold Return: {bh_return:+.2f}%")
    print(f"    Sharpe Ratio:      {sharpe:.3f}")
    print(f"    Sortino Ratio:     {sortino:.3f}")
    print(f"    Max Drawdown:      {max_dd:.2f}%")
    print(f"    Win Rate:          {win_rate:.1%}")
    print(f"\n  Trading Activity:")
    print(f"    Trades Taken:      {trades_taken}")
    print(f"    Days Skipped:      {trades_skipped} ({trades_skipped/len(returns_list)*100:.1f}%)")
    print(f"    Final Equity:      ${equity:,.2f}")

    return {
        'ticker': ticker,
        'total_return': total_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades_taken': trades_taken,
        'trades_skipped': trades_skipped,
        'bh_return': bh_return
    }

# -----------------------------
# 4. Main
# -----------------------------
if __name__ == "__main__":
    print("="*70)
    print("MULTI-ASSET BACKTEST TESTING (SYNTHETIC DATA)")
    print("="*70)
    print("\nNote: Using synthetic data because network access is restricted.")
    print("When running locally with network access, use backtest_stocks.py")

    all_results = []

    for ticker in TICKERS:
        try:
            result = run_backtest(ticker)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  Error backtesting {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Ticker':<10} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Win%':>10}")
        print("-"*60)
        for r in all_results:
            print(f"{r['ticker']:<10} {r['total_return']:>+9.2f}% {r['sharpe']:>10.3f} {r['max_dd']:>9.2f}% {r['win_rate']:>9.1%}")

    print("\n" + "="*70)
    print("Backtest complete!")
    print("="*70)
