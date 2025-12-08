"""
Multi-Asset Backtest for EKF + FFNN + HyperDUM Strategy

Backtests stocks (NVDA, TSLA) using trained models.
Includes comparison with buy-and-hold.

Usage:
    python backtest_stocks.py NVDA      # Backtest NVIDIA
    python backtest_stocks.py TSLA      # Backtest Tesla
    python backtest_stocks.py NVDA TSLA # Backtest multiple tickers

Prerequisites: Run train_stock_models.py first to generate model files.
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Default tickers
DEFAULT_TICKERS = ['NVDA', 'TSLA']

# Get tickers from command line
if len(sys.argv) > 1:
    TICKERS = [t.upper() for t in sys.argv[1:]]
else:
    TICKERS = DEFAULT_TICKERS

# Configuration
VOL_TARGET = 0.20
UNCERTAINTY_THRESHOLD = 0.385
MAX_GROSS_EXPOSURE = 0.50
KELLY_FRACTION = 0.25
INITIAL_USD = 10000.0
START_DATE = "2020-01-01"

# -----------------------------
# FFNN Model Definition
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

# -----------------------------
# EKF
# -----------------------------
def run_ekf(price_series, dt=1.0):
    """Extended Kalman Filter for state estimation"""
    if isinstance(price_series, pd.Series):
        values = price_series.values
        index = price_series.index
    else:
        values = np.array(price_series)
        index = pd.RangeIndex(len(values))

    values = np.asarray(values).flatten()
    if len(values) == 0:
        raise ValueError("Cannot run EKF on empty price series")

    n = len(values)
    x = np.zeros((n, 3))
    P = np.zeros((n, 3, 3))

    x[0] = np.array([float(values[0]), 0.0, -5.0])
    P[0] = np.eye(3) * 1.0

    Q = np.diag([0.01, 1e-4, 1e-4])
    R = 0.5

    smoothed = np.zeros(n)

    for t in range(1, n):
        F = np.array([[1, dt, 0],
                      [0,  1, 0],
                      [0,  0, 1]])
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
# Get VIX data
# -----------------------------
def get_vix_data(start_date, end_date, index):
    """Download VIX data as sentiment proxy"""
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d", progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix['Close']
            if isinstance(vix, pd.DataFrame):
                vix = vix.iloc[:, 0]
        else:
            vix = vix['Close']

        vix = vix.reindex(index, method='ffill')
        vix_normalized = (vix - 20) / 20
        vix_normalized = np.clip(vix_normalized, -1, 1)
        return vix_normalized
    except Exception as e:
        print(f"  Warning: Could not download VIX: {e}")
        np.random.seed(42)
        return pd.Series(np.tanh(np.cumsum(np.random.normal(0, 0.05, len(index)))), index=index)

# -----------------------------
# Backtest single ticker
# -----------------------------
def backtest_ticker(ticker):
    """Run comprehensive backtest for a single ticker"""
    print("\n" + "="*80)
    print(f"BACKTEST: {ticker}")
    print("="*80)

    # Handle crypto tickers
    is_crypto = ticker.upper() in ['BTC-USD', 'ETH-USD', 'BTC', 'ETH']
    ticker_original = ticker
    if ticker.upper() == 'BTC':
        ticker = 'BTC-USD'
    elif ticker.upper() == 'ETH':
        ticker = 'ETH-USD'

    ticker_clean = ticker.replace('-', '_').lower()

    # Load models
    print("\nLoading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = torch.load(f"{ticker_clean}_model.pth", map_location=device, weights_only=False)
        scaler = torch.load(f"{ticker_clean}_scaler.pth", map_location=device, weights_only=False)
        projector = np.load(f"{ticker_clean}_projector.npy")
        memory_vector = np.load(f"{ticker_clean}_memory.npy")
        model.eval()
        model.to(device)
        print(f"  Models loaded for {ticker}")
    except FileNotFoundError as e:
        print(f"  Error: Model files not found for {ticker}")
        print(f"  Run: python train_stock_models.py {ticker_original}")
        return None

    # Download data
    print(f"\nDownloading {ticker} data...")
    df = yf.download(ticker, start=START_DATE, end=None, interval="1d", progress=False)

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]

    df = pd.DataFrame(df, columns=['close'])
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()

    print(f"  Downloaded {len(df)} days ({df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d})")

    # Get sentiment proxy
    if is_crypto:
        np.random.seed(42)
        funding = np.tanh(np.cumsum(np.random.normal(0, 0.0005, len(df))) +
                        0.0002 * np.sin(np.arange(len(df)) * 2 * np.pi / 180))
        df['sentiment'] = funding + 0.0001 * np.random.randn(len(df))
    else:
        df['sentiment'] = get_vix_data(START_DATE, None, df.index)

    # Initialize backtest
    cash = INITIAL_USD
    position = 0.0
    equity_curve = []
    trade_log = []

    START_IDX = 60

    print("\nRunning backtest...")

    for i in range(START_IDX, len(df)):
        current_date = df.index[i]
        price = df['close'].iloc[i]
        actual_return = df['return'].iloc[i]

        # Get historical data
        hist_close = df['close'].iloc[:i+1]
        hist_returns = df['return'].iloc[:i+1]

        # Run EKF
        level_series, velocity_series = run_ekf(hist_close)
        level = level_series.iloc[-1]
        velocity = velocity_series.iloc[-1]

        # Get sentiment
        sentiment = df['sentiment'].iloc[i]

        # Calculate features
        recent_ret = hist_returns.iloc[-5:].mean() if len(hist_returns) >= 5 else 0.0
        rel_price = price / hist_close.iloc[-30:].mean() - 1 if len(hist_close) >= 30 else 0.0

        # Build feature vector
        feat = np.array([[level, velocity, sentiment, recent_ret, rel_price]])
        feat_s = scaler.transform(feat)

        # FFNN prediction
        with torch.no_grad():
            pred = model(torch.FloatTensor(feat_s).to(device)).item()

        # HyperDUM uncertainty
        projected = np.sign(feat_s @ projector)
        hamming_dist = np.mean(projected != memory_vector)

        # Calculate realized volatility
        recent_vol = hist_returns.iloc[-60:].std() * np.sqrt(252) if len(hist_returns) >= 60 else 0.20

        # Risk gates
        gross_exposure = abs(position * price) / max(cash + position * price, 1.0)

        target = 0.0
        skip_reason = None

        if hamming_dist > UNCERTAINTY_THRESHOLD:
            skip_reason = "HyperDUM"
        elif gross_exposure > MAX_GROSS_EXPOSURE and abs(pred) > 0:
            skip_reason = "RiskGate"
        else:
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            target = np.sign(pred) * risk * KELLY_FRACTION

        # Calculate position
        equity = cash + position * price
        target_position_value = equity * target
        target_position_size = target_position_value / price

        # Execute trade
        position_diff = target_position_size - position
        trade_side = None

        if abs(position_diff) > 0.001:
            trade_side = "BUY" if position_diff > 0 else "SELL"

            trade_log.append({
                'date': current_date,
                'side': trade_side,
                'size': abs(position_diff),
                'price': price,
                'pred': pred,
                'hamming': hamming_dist,
                'volatility': recent_vol
            })

            position += position_diff
            cash -= position_diff * price

        # Apply return
        if i < len(df) - 1:
            position_return = position * price * actual_return
            cash += position_return

        # Update equity
        equity = cash + position * price
        equity_curve.append({
            'date': current_date,
            'equity': equity,
            'position': position,
            'price': price,
            'hamming': hamming_dist,
            'pred': pred,
            'skip_reason': skip_reason,
            'target': target
        })

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    trades_df = pd.DataFrame(trade_log)

    equity_df['equity_return'] = equity_df['equity'].pct_change()
    equity_df['asset_return'] = equity_df['price'].pct_change()

    equity_df['strategy_cumulative'] = (1 + equity_df['equity_return']).cumprod()
    equity_df['asset_cumulative'] = (1 + equity_df['asset_return']).cumprod()

    returns = equity_df['equity_return'].dropna()
    asset_returns = equity_df['asset_return'].dropna()

    total_return = (equity_df['equity'].iloc[-1] / INITIAL_USD - 1) * 100
    asset_total_return = (equity_df['price'].iloc[-1] / equity_df['price'].iloc[0] - 1) * 100

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    asset_sharpe = np.mean(asset_returns) / np.std(asset_returns) * np.sqrt(252) if np.std(asset_returns) > 0 else 0

    # Sortino
    downside_returns = returns[returns < 0]
    sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0

    # Max drawdown
    cumulative = equity_df['strategy_cumulative']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() * 100

    asset_cumulative = equity_df['asset_cumulative']
    asset_running_max = asset_cumulative.expanding().max()
    asset_drawdown = (asset_cumulative - asset_running_max) / asset_running_max
    asset_max_dd = asset_drawdown.min() * 100

    # Calmar
    calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else 0

    # Win rate
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0

    # Trade stats
    num_trades = len(trades_df)
    days_skipped_hyperdum = len(equity_df[equity_df['skip_reason'] == 'HyperDUM'])
    days_skipped_risk = len(equity_df[equity_df['skip_reason'] == 'RiskGate'])
    days_with_position = len(equity_df[abs(equity_df['position']) > 0.001])

    # Print results
    print(f"\n{'='*80}")
    print(f"{ticker} BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"\nPeriod: {equity_df.index[0]:%Y-%m-%d} to {equity_df.index[-1]:%Y-%m-%d}")
    print(f"Trading Days: {len(equity_df)}")

    print(f"\n{'-'*80}")
    print("PERFORMANCE METRICS")
    print(f"{'-'*80}")
    print(f"{'Metric':<30} {'Strategy':>15} {'Buy & Hold':>15}")
    print(f"{'-'*80}")
    print(f"{'Total Return':<30} {total_return:>14.2f}% {asset_total_return:>14.2f}%")
    print(f"{'Sharpe Ratio':<30} {sharpe:>15.3f} {asset_sharpe:>15.3f}")
    print(f"{'Sortino Ratio':<30} {sortino:>15.3f} {'N/A':>15}")
    print(f"{'Calmar Ratio':<30} {calmar:>15.3f} {'N/A':>15}")
    print(f"{'Max Drawdown':<30} {max_dd:>14.2f}% {asset_max_dd:>14.2f}%")
    print(f"{'Win Rate':<30} {win_rate:>14.1%} {'N/A':>15}")
    print(f"{'Volatility (Annual)':<30} {returns.std() * np.sqrt(252):>14.1%} {asset_returns.std() * np.sqrt(252):>14.1%}")

    print(f"\n{'-'*80}")
    print("TRADE STATISTICS")
    print(f"{'-'*80}")
    print(f"Total Trades:             {num_trades}")
    print(f"Days with Position:       {days_with_position} ({days_with_position/len(equity_df)*100:.1f}%)")
    print(f"Days Skipped (HyperDUM):  {days_skipped_hyperdum} ({days_skipped_hyperdum/len(equity_df)*100:.1f}%)")
    print(f"Days Skipped (Risk Gate): {days_skipped_risk} ({days_skipped_risk/len(equity_df)*100:.1f}%)")

    # Year-by-year
    print(f"\n{'-'*80}")
    print("YEAR-BY-YEAR PERFORMANCE")
    print(f"{'-'*80}")
    equity_df['year'] = equity_df.index.year
    for year in equity_df['year'].unique():
        year_data = equity_df[equity_df['year'] == year]
        year_returns = year_data['equity_return'].dropna()
        asset_year_returns = year_data['asset_return'].dropna()

        if len(year_returns) > 0:
            year_total = (year_data['equity'].iloc[-1] / year_data['equity'].iloc[0] - 1) * 100
            asset_year_total = (year_data['price'].iloc[-1] / year_data['price'].iloc[0] - 1) * 100
            year_sharpe = np.mean(year_returns) / np.std(year_returns) * np.sqrt(252) if np.std(year_returns) > 0 else 0
            year_trades = len(trades_df[trades_df['date'].dt.year == year]) if len(trades_df) > 0 else 0

            print(f"{year}: Return={year_total:+.2f}% ({ticker}: {asset_year_total:+.2f}%), "
                  f"Sharpe={year_sharpe:.2f}, Trades={year_trades}")

    # Save results
    equity_df.to_csv(f'{ticker_clean}_backtest_results.csv')
    if len(trades_df) > 0:
        trades_df.to_csv(f'{ticker_clean}_backtest_trades.csv', index=False)

    print(f"\nResults saved to:")
    print(f"  - {ticker_clean}_backtest_results.csv")
    print(f"  - {ticker_clean}_backtest_trades.csv")

    return {
        'ticker': ticker,
        'total_return': total_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'asset_return': asset_total_return
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE BACKTEST - EKF + FFNN + HyperDUM")
    print("="*80)
    print(f"Tickers: {', '.join(TICKERS)}")

    results = []
    for ticker in TICKERS:
        result = backtest_ticker(ticker)
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"{'Ticker':<10} {'Return':>12} {'Sharpe':>10} {'Max DD':>10} {'Win Rate':>10} {'Asset':>12}")
        print("-"*80)
        for r in results:
            print(f"{r['ticker']:<10} {r['total_return']:>11.2f}% {r['sharpe']:>10.3f} {r['max_dd']:>9.2f}% {r['win_rate']:>9.1%} {r['asset_return']:>11.2f}%")
