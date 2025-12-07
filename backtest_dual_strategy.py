"""
Dual-Strategy Backtest - Switches between Trend-Following and Mean-Reversion

BULL MARKETS: Trend-following strategy
- Buy and hold with trailing stops at support bands
- Capture uptrends without fighting them

BEAR MARKETS: Mean-reversion strategy
- EKF + FFNN + HyperDUM
- Active trading with high selectivity

This should combine the best of both worlds:
- Bear market protection (+7% in 2022 when BTC dropped -61%)
- Bull market participation (capture trends instead of fighting them)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import VOL_TARGET, MAX_GROSS_EXPOSURE, KELLY_FRACTION
import os
from dotenv import load_dotenv
import requests
from trend_strategy import TrendFollowingStrategy

load_dotenv()

# FFNN model definition
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


def download_btc_alphavantage(api_key, start_date="2022-01-01"):
    """Download BTC data from Alpha Vantage"""
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
        raise ValueError(f"API error")

    time_series = data['Time Series (Digital Currency Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Find close column
    possible_cols = ['4a. close (USD)', '4. close', 'close']
    close_col = None
    for col in possible_cols:
        if col in df.columns:
            close_col = col
            break
    if close_col is None:
        close_col = next((col for col in df.columns if 'close' in col.lower()), None)

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


def run_dual_strategy_backtest(prices, model, scaler, projector, memory, window=30):
    """
    Run backtest with dual strategy:
    - BULL: Trend-following
    - BEAR: Mean-reversion (EKF + FFNN + HyperDUM)
    """

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

    # Initialize trend strategy
    trend_strat = TrendFollowingStrategy(
        vol_target=VOL_TARGET,
        max_exposure=MAX_GROSS_EXPOSURE
    )

    # Calculate support bands for regime detection
    sma_20w = df['Close'].rolling(window=140).mean()
    ema_21w = df['Close'].ewm(span=147, adjust=False).mean()

    # Prepare features for mean-reversion strategy
    features = []
    targets = []
    indices = []
    modes = []  # Track which strategy mode we're in

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

        # Determine mode: BULL if price > both support bands
        price = df['Close'].iloc[i]
        sma = sma_20w.iloc[i]
        ema = ema_21w.iloc[i]

        if pd.notna(sma) and pd.notna(ema) and price > sma and price > ema:
            modes.append('BULL')
        else:
            modes.append('BEAR')

    X = np.array(features)
    y = np.array(targets)

    # Scale features
    X_scaled = scaler.transform(X)

    # Get mean-reversion predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        predictions = model(X_tensor).numpy().flatten()

    # Run backtest
    equity = 100.0
    position = 0.0
    equity_curve = [equity]
    trades = []
    mode_stats = {'BULL': {'days': 0, 'trades': 0, 'return': 0},
                  'BEAR': {'days': 0, 'trades': 0, 'return': 0}}
    mode_switches = []

    current_mode = None

    for i in range(len(predictions)):
        actual_return = y[i]
        mode = modes[i]
        price = df['Close'].iloc[window + i]

        mode_stats[mode]['days'] += 1

        # Track mode switches
        if current_mode != mode:
            mode_switches.append((indices[i], mode))
            current_mode = mode

        # Calculate volatility
        lookback_start = max(0, window - 60 + i)
        lookback_end = window + i
        if lookback_end > lookback_start + 30:
            vol_window = df['return'].iloc[lookback_start:lookback_end]
            recent_vol = vol_window.std() * np.sqrt(252) if len(vol_window) > 0 else 0.20
        else:
            recent_vol = 0.20

        # Strategy selection based on mode
        if mode == 'BULL':
            # TREND-FOLLOWING MODE
            hist_prices = df['Close'].iloc[:window+i+1]
            signal = trend_strat.generate_signal(
                hist_prices,
                price,
                abs(position),
                equity,
                recent_vol
            )

            target_position = signal['target_position']

            if signal['action'] in ['BUY', 'SELL'] and abs(target_position - abs(position)) > 0.001:
                mode_stats['BULL']['trades'] += 1
                trades.append({
                    'date': indices[i],
                    'mode': 'BULL',
                    'action': signal['action'],
                    'reason': signal['reason']
                })

        else:  # BEAR mode
            # MEAN-REVERSION MODE (EKF + FFNN + HyperDUM)
            pred = predictions[i]

            # HyperDUM check
            feat_s = X_scaled[i:i+1]
            projected = np.sign(feat_s @ projector)
            hamming_dist = np.mean(projected != memory)

            target_position = 0.0

            # Bear mode uses stricter threshold (0.60)
            if hamming_dist > 0.60:
                pass  # Blocked by HyperDUM
            else:
                risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
                target_position = np.sign(pred) * risk * KELLY_FRACTION

                if abs(target_position - abs(position)) > 0.001:
                    mode_stats['BEAR']['trades'] += 1
                    trades.append({
                        'date': indices[i],
                        'mode': 'BEAR',
                        'action': 'TRADE',
                        'reason': 'Mean-reversion signal'
                    })

        # Calculate return
        period_return = position * actual_return
        mode_stats[mode]['return'] += period_return

        equity *= (1 + period_return)
        equity_curve.append(equity)

        # Update position
        position = target_position

    # Calculate metrics
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]

    total_return = (equity_array[-1] / equity_array[0] - 1) * 100
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    cum_returns = np.cumprod(1 + returns)
    max_dd = np.min(cum_returns / np.maximum.accumulate(cum_returns) - 1) * 100

    win_rate = np.mean(returns > 0) * 100 if len(returns) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'mode_stats': mode_stats,
        'mode_switches': mode_switches,
        'equity_curve': equity_array
    }


def main():
    print("="*80)
    print("DUAL-STRATEGY BACKTEST")
    print("BULL: Trend-Following | BEAR: Mean-Reversion (EKF+FFNN+HyperDUM)")
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

    print("Running dual-strategy backtest...\n")

    # Run dual strategy
    results = run_dual_strategy_backtest(prices, model, scaler, projector, memory)

    # Calculate buy & hold
    btc_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

    # Display results
    print("\n" + "="*80)
    print("DUAL-STRATEGY RESULTS")
    print("="*80)

    print(f"\nTotal Return:     {results['total_return']:+.2f}%")
    print(f"Sharpe Ratio:     {results['sharpe']:.3f}")
    print(f"Max Drawdown:     {results['max_dd']:.2f}%")
    print(f"Win Rate:         {results['win_rate']:.1f}%")
    print(f"Total Trades:     {results['total_trades']}")

    print(f"\n{'─'*80}")
    print("STRATEGY MODE BREAKDOWN")
    print(f"{'─'*80}")

    for mode in ['BULL', 'BEAR']:
        stats = results['mode_stats'][mode]
        mode_return = (1 + stats['return']) - 1
        print(f"\n{mode} MODE:")
        print(f"  Days:   {stats['days']}")
        print(f"  Trades: {stats['trades']}")
        print(f"  Return: {mode_return*100:+.2f}%")

    print(f"\n{'─'*80}")
    print(f"Mode Switches: {len(results['mode_switches'])}")
    if results['mode_switches']:
        print("Recent switches:")
        for date, mode in results['mode_switches'][-5:]:
            print(f"  {date:%Y-%m-%d}: Switched to {mode}")

    print(f"\n{'─'*80}")
    print("COMPARISON")
    print(f"{'─'*80}")
    print(f"Dual Strategy:    {results['total_return']:+.2f}%")
    print(f"Buy & Hold BTC:   {btc_return:+.2f}%")

    improvement = results['total_return'] / btc_return if btc_return != 0 else 0
    print(f"\nCapture Ratio:    {improvement:.1%} of buy & hold")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
