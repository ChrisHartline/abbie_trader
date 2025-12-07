"""
Threshold Testing Script - Find optimal HyperDUM threshold

Tests multiple threshold values and compares:
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate
- % Days Traded
- Trade Count

Usage: python test_thresholds.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import VOL_TARGET, MAX_GROSS_EXPOSURE, KELLY_FRACTION
import os
from dotenv import load_dotenv
import requests
from datetime import datetime

load_dotenv()

# Thresholds to test
THRESHOLDS = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35]

# FFNN model definition (required for loading the model)
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

# Download BTC data from Alpha Vantage
def download_btc_alphavantage(api_key, start_date="2022-01-01"):
    """Download BTC daily data from Alpha Vantage"""
    print(f"Downloading BTC data from {start_date}...")

    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'BTC',
        'market': 'USD',
        'apikey': api_key,
        'outputsize': 'full'
    }

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    if 'Time Series (Digital Currency Daily)' not in data:
        raise ValueError(f"API error: {data.get('Error Message', data.get('Note', 'Unknown error'))}")

    time_series = data['Time Series (Digital Currency Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Find close column
    possible_cols = ['4a. close (USD)', '4. close', 'close', '4b. close (USD)']
    close_col = None
    for col in possible_cols:
        if col in df.columns:
            close_col = col
            break
    if close_col is None:
        close_col = next((col for col in df.columns if 'close' in col.lower()), None)
    if close_col is None:
        raise ValueError(f"Could not find close column")

    close = df[close_col].astype(float)
    close = close[close.index >= start_date]

    print(f"✓ Downloaded {len(close)} days")
    return close

# EKF function
def run_ekf(price_series, dt=1.0):
    """Extended Kalman Filter"""
    values = price_series.values if isinstance(price_series, pd.Series) else np.array(price_series)
    index = price_series.index if isinstance(price_series, pd.Series) else pd.RangeIndex(len(values))

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

def run_backtest_with_threshold(prices, model, scaler, projector, memory, threshold, window=30):
    """Run backtest with specific threshold"""

    # Prepare data
    df = pd.DataFrame({'Close': prices})
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Synthetic funding rate
    np.random.seed(42)
    funding = np.tanh(np.cumsum(np.random.normal(0, 0.0005, len(df))) +
                     0.0002 * np.sin(np.arange(len(df)) * 2 * np.pi / 180))
    df['funding_rate'] = funding + 0.0001 * np.random.randn(len(df))
    df = df.dropna()

    # Run EKF
    df['level'], df['velocity'] = run_ekf(df['Close'])

    # Prepare features
    features = []
    targets = []
    indices = []

    for i in range(window, len(df)-1):
        row = [
            df['level'].iloc[i],
            df['velocity'].iloc[i],
            df['funding_rate'].iloc[i],
            df['return'].iloc[i-4:i+1].mean() if i >= 4 else df['return'].iloc[:i+1].mean(),
            df['Close'].iloc[i] / df['Close'].iloc[i-window:i].mean() - 1
        ]
        features.append(row)
        targets.append(df['return'].iloc[i+1])
        indices.append(df.index[i+1])

    X = np.array(features)
    y = np.array(targets)

    # Scale features
    X_scaled = scaler.transform(X)

    # Get predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        predictions = model(X_tensor).numpy().flatten()

    # Run backtest with threshold
    equity = 100.0
    position = 0.0
    equity_curve = [equity]
    trades_taken = 0
    trades_blocked = 0
    returns = []

    for i in range(len(predictions)):
        pred = predictions[i]
        actual_return = y[i]

        # HyperDUM check
        feat_s = X_scaled[i:i+1]
        projected = np.sign(feat_s @ projector)
        hamming_dist = np.mean(projected != memory)

        # Calculate volatility
        lookback_start = max(0, window - 60 + i)
        lookback_end = window + i
        if lookback_end > lookback_start + 30:
            vol_window = df['return'].iloc[lookback_start:lookback_end]
            recent_vol = vol_window.std() * np.sqrt(252) if len(vol_window) > 0 else 0.20
        else:
            recent_vol = 0.20

        # Risk gates
        target_position = 0.0
        if hamming_dist > threshold:
            trades_blocked += 1
        elif abs(position) > MAX_GROSS_EXPOSURE:
            pass  # Skip
        else:
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            target_position = np.sign(pred) * risk * KELLY_FRACTION
            if abs(target_position - position) > 0.001:
                trades_taken += 1

        # Calculate return
        period_return = position * actual_return
        equity *= (1 + period_return)
        equity_curve.append(equity)
        returns.append(period_return)

        # Update position
        position = target_position

    # Calculate metrics
    equity_array = np.array(equity_curve)
    returns_array = np.array(returns)

    total_return = (equity_array[-1] / equity_array[0] - 1) * 100
    sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0

    cum_returns = np.cumprod(1 + returns_array)
    max_dd = np.min(cum_returns / np.maximum.accumulate(cum_returns) - 1) * 100

    win_rate = np.mean(returns_array > 0) * 100 if len(returns_array) > 0 else 0
    pct_days_traded = (1 - trades_blocked / len(predictions)) * 100

    return {
        'threshold': threshold,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'pct_traded': pct_days_traded,
        'trades': trades_taken,
        'blocked': trades_blocked
    }

def main():
    print("="*80)
    print("HYPERDUM THRESHOLD OPTIMIZATION")
    print("="*80)

    # Load models
    print("\nLoading models...")
    model = torch.load('btc_model.pth', map_location='cpu', weights_only=False)
    scaler = torch.load('btc_scaler.pth', map_location='cpu', weights_only=False)
    projector = np.load('projector.npy')
    memory = np.load('memory.npy')
    print("✓ Models loaded")

    # Download data
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY not found in .env")

    prices = download_btc_alphavantage(api_key, start_date="2022-01-01")
    print(f"Data range: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}\n")

    # Test each threshold
    results = []
    for threshold in THRESHOLDS:
        print(f"Testing threshold {threshold:.2f}...", end=" ")
        result = run_backtest_with_threshold(prices, model, scaler, projector, memory, threshold)
        results.append(result)
        print(f"Return: {result['total_return']:+.2f}%, Sharpe: {result['sharpe']:.3f}, Traded: {result['pct_traded']:.1f}%")

    # Create results table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    df = pd.DataFrame(results)

    print("\n" + df.to_string(index=False, formatters={
        'threshold': '{:.2f}'.format,
        'total_return': '{:+.2f}%'.format,
        'sharpe': '{:.3f}'.format,
        'max_dd': '{:.2f}%'.format,
        'win_rate': '{:.1f}%'.format,
        'pct_traded': '{:.1f}%'.format,
        'trades': '{:.0f}'.format,
        'blocked': '{:.0f}'.format
    }))

    # Find best by Sharpe
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    best_return = df.loc[df['total_return'].idxmax()]

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print(f"\nBest Sharpe Ratio: {best_sharpe['threshold']:.2f} "
          f"(Sharpe: {best_sharpe['sharpe']:.3f}, Return: {best_sharpe['total_return']:+.2f}%, "
          f"Traded: {best_sharpe['pct_traded']:.1f}%)")

    print(f"\nBest Total Return: {best_return['threshold']:.2f} "
          f"(Return: {best_return['total_return']:+.2f}%, Sharpe: {best_return['sharpe']:.3f}, "
          f"Traded: {best_return['pct_traded']:.1f}%)")

    # Compare to buy & hold
    btc_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    print(f"\nBuy & Hold BTC: {btc_return:+.2f}%")

    print("\n" + "="*80)
    print(f"\nTo use optimal threshold, update config.py:")
    print(f"  UNCERTAINTY_THRESHOLD = {best_sharpe['threshold']:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
