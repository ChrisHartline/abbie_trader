"""
Adaptive Backtest - Adjusts HyperDUM threshold based on market regime

Uses MarketRegimeRouter to detect BULL/BEAR/TRANSITION and adjusts strategy:
- BULL: Higher threshold (0.70) → Allow more trades
- TRANSITION: Medium threshold (0.65) → Moderate selectivity
- BEAR: Lower threshold (0.60) → High selectivity

This should improve participation in bull markets while maintaining protection in bears.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import VOL_TARGET, MAX_GROSS_EXPOSURE, KELLY_FRACTION
import os
from dotenv import load_dotenv
import requests
from market_regime_router import MarketRegimeRouter

load_dotenv()

# FFNN model definition (required for loading)
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


# Adaptive threshold configuration
THRESHOLDS = {
    'BULL': 0.70,        # More lenient in bulls
    'TRANSITION': 0.65,  # Moderate
    'BEAR': 0.60         # More selective in bears
}


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


def run_adaptive_backtest(prices, model, scaler, projector, memory, window=30):
    """Run backtest with adaptive threshold based on market regime"""

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

    # Initialize router
    router = MarketRegimeRouter()

    # Prepare features
    features = []
    targets = []
    indices = []
    regimes = []
    thresholds_used = []

    for i in range(window, len(df)-1):
        # Classify regime using prices up to current point
        regime, confidence, signals = router.classify(df['Close'].iloc[:i+1])
        regimes.append(regime)
        thresholds_used.append(THRESHOLDS[regime])

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

    # Run backtest with adaptive threshold
    equity = 100.0
    position = 0.0
    equity_curve = [equity]
    trades_taken = 0
    returns = []
    trades_by_regime = {'BULL': 0, 'BEAR': 0, 'TRANSITION': 0}
    blocked_by_regime = {'BULL': 0, 'BEAR': 0, 'TRANSITION': 0}
    regime_days = {'BULL': 0, 'BEAR': 0, 'TRANSITION': 0}

    for i in range(len(predictions)):
        pred = predictions[i]
        actual_return = y[i]
        regime = regimes[i]
        threshold = thresholds_used[i]

        regime_days[regime] += 1

        # HyperDUM check with adaptive threshold
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

        # Risk gates with adaptive threshold
        target_position = 0.0
        if hamming_dist > threshold:
            blocked_by_regime[regime] += 1
        elif abs(position) > MAX_GROSS_EXPOSURE:
            pass
        else:
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            target_position = np.sign(pred) * risk * KELLY_FRACTION
            if abs(target_position - position) > 0.001:
                trades_taken += 1
                trades_by_regime[regime] += 1

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

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades': trades_taken,
        'trades_by_regime': trades_by_regime,
        'blocked_by_regime': blocked_by_regime,
        'regime_days': regime_days,
        'equity_curve': equity_array
    }


def main():
    print("="*80)
    print("ADAPTIVE BACKTEST - Regime-Based Threshold Adjustment")
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

    # Show threshold configuration
    print("="*80)
    print("ADAPTIVE THRESHOLD CONFIGURATION")
    print("="*80)
    for regime, threshold in THRESHOLDS.items():
        print(f"  {regime:12s}: {threshold:.2f}")

    print("\nRunning adaptive backtest...\n")

    # Run adaptive backtest
    adaptive_results = run_adaptive_backtest(prices, model, scaler, projector, memory)

    # Also run fixed threshold backtest for comparison
    print("Running fixed threshold backtest (0.60) for comparison...\n")

    # Temporarily set all thresholds to 0.60
    original_thresholds = THRESHOLDS.copy()
    for regime in THRESHOLDS:
        THRESHOLDS[regime] = 0.60

    fixed_results = run_adaptive_backtest(prices, model, scaler, projector, memory)

    # Restore adaptive thresholds
    THRESHOLDS.update(original_thresholds)

    # Calculate buy & hold
    btc_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

    # Display results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    print("\n" + "-"*80)
    print("ADAPTIVE THRESHOLD (Bull: 0.70, Transition: 0.65, Bear: 0.60)")
    print("-"*80)
    print(f"Total Return:     {adaptive_results['total_return']:+.2f}%")
    print(f"Sharpe Ratio:     {adaptive_results['sharpe']:.3f}")
    print(f"Max Drawdown:     {adaptive_results['max_dd']:.2f}%")
    print(f"Win Rate:         {adaptive_results['win_rate']:.1f}%")
    print(f"Total Trades:     {adaptive_results['trades']}")

    print("\n  Trades by Regime:")
    for regime in ['BULL', 'TRANSITION', 'BEAR']:
        total_days = adaptive_results['regime_days'][regime]
        trades = adaptive_results['trades_by_regime'][regime]
        blocked = adaptive_results['blocked_by_regime'][regime]
        participation = (trades / total_days * 100) if total_days > 0 else 0
        print(f"    {regime:12s}: {trades:3d} trades, {blocked:4d} blocked, "
              f"{total_days:4d} days ({participation:.1f}% participation)")

    print("\n" + "-"*80)
    print("FIXED THRESHOLD (0.60 for all regimes)")
    print("-"*80)
    print(f"Total Return:     {fixed_results['total_return']:+.2f}%")
    print(f"Sharpe Ratio:     {fixed_results['sharpe']:.3f}")
    print(f"Max Drawdown:     {fixed_results['max_dd']:.2f}%")
    print(f"Win Rate:         {fixed_results['win_rate']:.1f}%")
    print(f"Total Trades:     {fixed_results['trades']}")

    print("\n" + "-"*80)
    print("BUY & HOLD")
    print("-"*80)
    print(f"Total Return:     {btc_return:+.2f}%")

    print("\n" + "="*80)
    print("IMPROVEMENT FROM ADAPTIVE THRESHOLD")
    print("="*80)
    return_diff = adaptive_results['total_return'] - fixed_results['total_return']
    sharpe_diff = adaptive_results['sharpe'] - fixed_results['sharpe']
    trades_diff = adaptive_results['trades'] - fixed_results['trades']

    print(f"Return difference:  {return_diff:+.2f}% ({'BETTER' if return_diff > 0 else 'WORSE'})")
    print(f"Sharpe difference:  {sharpe_diff:+.3f} ({'BETTER' if sharpe_diff > 0 else 'WORSE'})")
    print(f"Additional trades:  {trades_diff:+d}")

    if return_diff > 0:
        print(f"\n✓ Adaptive threshold improves returns by {return_diff:.2f}%!")
    else:
        print(f"\n✗ Adaptive threshold underperforms by {abs(return_diff):.2f}%")

    print("\n" + "="*80)
    print("To use adaptive threshold, update main.py to use market_regime_router")
    print("="*80)


if __name__ == "__main__":
    main()
