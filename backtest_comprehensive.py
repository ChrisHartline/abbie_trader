"""
Comprehensive Backtest for EKF + FFNN + HyperDUM Strategy

Tests the exact logic from main.py with detailed metrics and analysis.
Includes:
- Full performance metrics (Sharpe, Sortino, Calmar, Max DD)
- Trade-by-trade analysis
- Period-by-period breakdown
- Comparison with buy-and-hold
- HyperDUM effectiveness analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# ============================
# Configuration (from main.py)
# ============================
VOL_TARGET = 0.20
UNCERTAINTY_THRESHOLD = 0.385
MAX_GROSS_EXPOSURE = 0.50
KELLY_FRACTION = 0.5
INITIAL_USD = 10000.0

# ============================
# FFNN class (must match training)
# ============================
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

# ============================
# EKF (exact from main.py)
# ============================
def run_ekf(price_series, dt=1.0):
    """Extended Kalman Filter - exact from main.py"""
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

# ============================
# Download BTC data from Alpha Vantage
# ============================
def download_btc_alphavantage(api_key, start_date="2022-01-01"):
    """
    Download BTC daily data from Alpha Vantage

    Args:
        api_key: Alpha Vantage API key
        start_date: Start date for historical data (YYYY-MM-DD)

    Returns:
        DataFrame with close prices
    """
    print(f"Fetching BTC data from Alpha Vantage (from {start_date})...")

    # Alpha Vantage Digital Currency Daily endpoint
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'BTC',
        'market': 'USD',
        'apikey': api_key,
        'outputsize': 'full'  # Get full history
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")

        if 'Note' in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

        if 'Time Series (Digital Currency Daily)' not in data:
            raise ValueError(f"Unexpected API response: {data}")

        # Parse the time series data
        time_series = data['Time Series (Digital Currency Daily)']

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Extract close price (USD)
        # Alpha Vantage may return different column names depending on endpoint
        # Try multiple possible column names
        possible_close_cols = ['4a. close (USD)', '4. close', 'close', '4b. close (USD)']
        close_col = None

        for col in possible_close_cols:
            if col in df.columns:
                close_col = col
                break

        # If still not found, try fuzzy matching
        if close_col is None:
            close_col = next((col for col in df.columns if 'close' in col.lower()), None)

        if close_col is None:
            raise ValueError(f"Could not find close price column. Available columns: {df.columns.tolist()}")

        print(f"Using column: '{close_col}'")
        close = df[close_col].astype(float)

        # Filter by start date
        close = close[close.index >= start_date]

        print(f"✓ Downloaded {len(close)} days from Alpha Vantage")
        return close

    except Exception as e:
        print(f"✗ Error downloading from Alpha Vantage: {e}")
        raise

# ============================
# Get historical funding rates
# ============================
def get_historical_funding(dates):
    """
    Generate synthetic funding rates for backtest
    In production, you'd fetch real historical funding from exchanges
    """
    np.random.seed(42)
    n = len(dates)

    # Synthetic funding that mimics real patterns:
    # - Mean-reverting around 0.01% (neutral)
    # - Spikes during high leverage periods
    # - Correlation with volatility
    base_funding = np.cumsum(np.random.normal(0, 0.0005, n))
    seasonal = 0.0002 * np.sin(np.arange(n) * 2 * np.pi / 180)
    noise = 0.0001 * np.random.randn(n)

    funding = np.tanh(base_funding + seasonal) + noise
    return pd.Series(funding, index=dates)

# ============================
# Load models
# ============================
print("="*80)
print("COMPREHENSIVE BACKTEST - EKF + FFNN + HyperDUM")
print("="*80)
print("\nLoading models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

try:
    model = torch.load("btc_model.pth", map_location=device, weights_only=False)
    scaler = torch.load("btc_scaler.pth", map_location=device, weights_only=False)
    projector = np.load("projector.npy")
    memory_vector = np.load("memory.npy")
    model.eval()
    model.to(device)
    print("✓ Models loaded successfully\n")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Run train_models.py first to generate model files")
    exit(1)

# ============================
# Download data
# ============================
print("Downloading BTC historical data...")
START_DATE = "2022-01-01"

# Get Alpha Vantage API key from environment variable or user
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

if not ALPHAVANTAGE_API_KEY:
    print("\n" + "="*80)
    print("ALPHA VANTAGE API KEY REQUIRED")
    print("="*80)
    print("\nSet your API key using one of these methods:")
    print("\n1. Add to .env file:")
    print("   ALPHAVANTAGE_API_KEY=your_key_here")
    print("\n2. Or enter it now:")
    ALPHAVANTAGE_API_KEY = input("Enter your Alpha Vantage API key: ").strip()

    if not ALPHAVANTAGE_API_KEY:
        print("\n✗ Error: API key required. Get one free at https://www.alphavantage.co/support/#api-key")
        exit(1)

print(f"✓ API key loaded from .env file\n")

# Download BTC data from Alpha Vantage
try:
    close = download_btc_alphavantage(ALPHAVANTAGE_API_KEY, START_DATE)
except Exception as e:
    print(f"\n✗ Failed to download data: {e}")
    print("\nTroubleshooting:")
    print("1. Check your API key is correct")
    print("2. Verify internet connection")
    print("3. Alpha Vantage free tier has 25 calls/day limit")
    exit(1)

# Create DataFrame
btc = pd.DataFrame({'close': close})
btc['return'] = np.log(btc['close'] / btc['close'].shift(1))
btc = btc.dropna()

if len(btc) == 0:
    print("✗ Error: No data downloaded. Check internet connection and try again.")
    exit(1)

print(f"Data range: {btc.index[0]:%Y-%m-%d} to {btc.index[-1]:%Y-%m-%d}\n")

# Add synthetic funding rate
btc['funding'] = get_historical_funding(btc.index)

# ============================
# Backtest loop
# ============================
print("Running backtest...")
print("="*80)

# Initialize
cash = INITIAL_USD
position = 0.0
equity_curve = []
trade_log = []
daily_metrics = []

# Need at least 60 days for vol calculation
START_IDX = 60

for i in range(START_IDX, len(btc)):
    current_date = btc.index[i]
    price = btc['close'].iloc[i]
    actual_return = btc['return'].iloc[i]

    # Get historical data up to current point
    hist_close = btc['close'].iloc[:i+1]
    hist_returns = btc['return'].iloc[:i+1]

    # Run EKF
    level_series, velocity_series = run_ekf(hist_close)
    level = level_series.iloc[-1]
    velocity = velocity_series.iloc[-1]

    # Get funding rate
    funding = btc['funding'].iloc[i]

    # Calculate features (same as main.py)
    recent_ret = hist_returns.iloc[-5:].mean() if len(hist_returns) >= 5 else 0.0
    rel_price = price / hist_close.iloc[-30:].mean() - 1 if len(hist_close) >= 30 else 0.0

    # Build feature vector
    feat = np.array([[level, velocity, funding, recent_ret, rel_price]])
    feat_s = scaler.transform(feat)

    # FFNN prediction
    with torch.no_grad():
        pred = model(torch.FloatTensor(feat_s).to(device)).item()

    # HyperDUM uncertainty
    projected = np.sign(feat_s @ projector)
    hamming_dist = np.mean(projected != memory_vector)

    # Calculate realized volatility (60-day)
    recent_vol = hist_returns.iloc[-60:].std() * np.sqrt(252) if len(hist_returns) >= 60 else 0.20

    # Risk gates
    gross_exposure = abs(position * price) / max(cash + position * price, 1.0)

    # Determine target position (same logic as main.py)
    target = 0.0
    skip_reason = None

    if hamming_dist > UNCERTAINTY_THRESHOLD:
        skip_reason = "HyperDUM"
    elif gross_exposure > MAX_GROSS_EXPOSURE and abs(pred) > 0:
        skip_reason = "RiskGate"
    else:
        # Calculate target
        risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
        target = np.sign(pred) * risk * KELLY_FRACTION

    # Calculate target position size
    equity = cash + position * price
    target_position_value = equity * target
    target_position_size = target_position_value / price

    # Execute trade
    position_diff = target_position_size - position
    trade_size = 0.0
    trade_side = None

    if abs(position_diff) > 0.001:  # Minimum trade size
        trade_size = position_diff
        trade_side = "BUY" if position_diff > 0 else "SELL"

        # Log trade
        trade_log.append({
            'date': current_date,
            'side': trade_side,
            'size': abs(trade_size),
            'price': price,
            'pred': pred,
            'hamming': hamming_dist,
            'volatility': recent_vol
        })

        # Update position
        position += position_diff
        cash -= position_diff * price

    # Apply return to position
    if i < len(btc) - 1:  # Don't apply return on last day
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

# ============================
# Calculate metrics
# ============================
print("\n" + "="*80)
print("BACKTEST RESULTS")
print("="*80)

# Convert to DataFrames
equity_df = pd.DataFrame(equity_curve).set_index('date')
trades_df = pd.DataFrame(trade_log)

# Calculate returns
equity_df['equity_return'] = equity_df['equity'].pct_change()
equity_df['btc_return'] = equity_df['price'].pct_change()

# Cumulative returns
equity_df['strategy_cumulative'] = (1 + equity_df['equity_return']).cumprod()
equity_df['btc_cumulative'] = (1 + equity_df['btc_return']).cumprod()

# Drop NaN
returns = equity_df['equity_return'].dropna()
btc_returns = equity_df['btc_return'].dropna()

# Performance metrics
total_return = (equity_df['equity'].iloc[-1] / INITIAL_USD - 1) * 100
btc_total_return = (equity_df['price'].iloc[-1] / equity_df['price'].iloc[0] - 1) * 100

sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
btc_sharpe = np.mean(btc_returns) / np.std(btc_returns) * np.sqrt(252) if np.std(btc_returns) > 0 else 0

# Sortino (downside deviation)
downside_returns = returns[returns < 0]
sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0

# Max drawdown
cumulative = equity_df['strategy_cumulative']
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min() * 100

btc_cumulative = equity_df['btc_cumulative']
btc_running_max = btc_cumulative.expanding().max()
btc_drawdown = (btc_cumulative - btc_running_max) / btc_running_max
btc_max_dd = btc_drawdown.min() * 100

# Calmar ratio
calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else 0

# Win rate
winning_days = len(returns[returns > 0])
total_days = len(returns)
win_rate = winning_days / total_days if total_days > 0 else 0

# Trade statistics
num_trades = len(trades_df)
days_skipped_hyperdum = len(equity_df[equity_df['skip_reason'] == 'HyperDUM'])
days_skipped_risk = len(equity_df[equity_df['skip_reason'] == 'RiskGate'])
days_with_position = len(equity_df[abs(equity_df['position']) > 0.001])

# Print results
print(f"\nPeriod: {equity_df.index[0]:%Y-%m-%d} to {equity_df.index[-1]:%Y-%m-%d}")
print(f"Trading Days: {len(equity_df)}")
print(f"\n{'─'*80}")
print("PERFORMANCE METRICS")
print(f"{'─'*80}")
print(f"{'Metric':<30} {'Strategy':>15} {'Buy & Hold':>15}")
print(f"{'─'*80}")
print(f"{'Total Return':<30} {total_return:>14.2f}% {btc_total_return:>14.2f}%")
print(f"{'Sharpe Ratio':<30} {sharpe:>15.3f} {btc_sharpe:>15.3f}")
print(f"{'Sortino Ratio':<30} {sortino:>15.3f} {'N/A':>15}")
print(f"{'Calmar Ratio':<30} {calmar:>15.3f} {'N/A':>15}")
print(f"{'Max Drawdown':<30} {max_dd:>14.2f}% {btc_max_dd:>14.2f}%")
print(f"{'Win Rate':<30} {win_rate:>14.1%} {'N/A':>15}")
print(f"{'Volatility (Annual)':<30} {returns.std() * np.sqrt(252):>14.1%} {btc_returns.std() * np.sqrt(252):>14.1%}")

print(f"\n{'─'*80}")
print("TRADE STATISTICS")
print(f"{'─'*80}")
print(f"Total Trades:             {num_trades}")
print(f"Days with Position:       {days_with_position} ({days_with_position/len(equity_df)*100:.1f}%)")
print(f"Days Skipped (HyperDUM):  {days_skipped_hyperdum} ({days_skipped_hyperdum/len(equity_df)*100:.1f}%)")
print(f"Days Skipped (Risk Gate): {days_skipped_risk} ({days_skipped_risk/len(equity_df)*100:.1f}%)")

if num_trades > 0:
    print(f"\n{'─'*80}")
    print("TRADE BREAKDOWN")
    print(f"{'─'*80}")
    buys = trades_df[trades_df['side'] == 'BUY']
    sells = trades_df[trades_df['side'] == 'SELL']
    print(f"Buy Trades:   {len(buys)}")
    print(f"Sell Trades:  {len(sells)}")
    print(f"Avg Trade Size: {trades_df['size'].mean():.6f} BTC")

# Year-by-year breakdown
print(f"\n{'─'*80}")
print("YEAR-BY-YEAR PERFORMANCE")
print(f"{'─'*80}")
equity_df['year'] = equity_df.index.year
for year in equity_df['year'].unique():
    year_data = equity_df[equity_df['year'] == year]
    year_returns = year_data['equity_return'].dropna()
    btc_year_returns = year_data['btc_return'].dropna()

    if len(year_returns) > 0:
        year_total = (year_data['equity'].iloc[-1] / year_data['equity'].iloc[0] - 1) * 100
        btc_year_total = (year_data['price'].iloc[-1] / year_data['price'].iloc[0] - 1) * 100
        year_sharpe = np.mean(year_returns) / np.std(year_returns) * np.sqrt(252) if np.std(year_returns) > 0 else 0
        year_trades = len(trades_df[trades_df['date'].dt.year == year])

        print(f"{year}: Return={year_total:+.2f}% (BTC: {btc_year_total:+.2f}%), "
              f"Sharpe={year_sharpe:.2f}, Trades={year_trades}")

# HyperDUM effectiveness
print(f"\n{'─'*80}")
print("HYPERDUM EFFECTIVENESS")
print(f"{'─'*80}")
# Compare returns on days when HyperDUM allowed vs blocked
allowed_days = equity_df[equity_df['skip_reason'].isna()]
blocked_days = equity_df[equity_df['skip_reason'] == 'HyperDUM']

if len(allowed_days) > 0:
    allowed_returns = allowed_days['equity_return'].dropna()
    avg_allowed_return = allowed_returns.mean() * 100
    print(f"Avg daily return (HyperDUM allowed): {avg_allowed_return:+.4f}%")
    print(f"Days allowed: {len(allowed_days)}")

if len(blocked_days) > 0:
    # What would have happened if we traded on blocked days?
    blocked_btc_returns = blocked_days['btc_return'].dropna()
    avg_blocked_return = blocked_btc_returns.mean() * 100
    print(f"Avg BTC return (HyperDUM blocked): {avg_blocked_return:+.4f}%")
    print(f"Days blocked: {len(blocked_days)}")
    print(f"\n✓ HyperDUM protected from {len(blocked_days)} potentially risky days")

# Save results
equity_df.to_csv('backtest_results.csv')
if len(trades_df) > 0:
    trades_df.to_csv('backtest_trades.csv', index=False)

print(f"\n{'─'*80}")
print(f"Results saved to:")
print(f"  - backtest_results.csv (daily equity curve)")
print(f"  - backtest_trades.csv (trade log)")
print("="*80)
