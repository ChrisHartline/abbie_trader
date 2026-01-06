"""
Backtest for Regime-Aware Trading Strategy

Tests main_with_regime.py logic:
- EKF + FFNN + HyperDUM (same as before)
- PLUS: Regime detection with dynamic position sizing
  * FAVORABLE (mean-rev + stable): 100% position size
  * CAUTION (trending + stable): 50% position size
  * WARNING (unstable): 0% - no trading
  * NEUTRAL: 75% position size

This should improve bull market participation while maintaining bear market protection.
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

# Load environment variables
load_dotenv()

# ============================
# Configuration
# ============================
VOL_TARGET = 0.20
UNCERTAINTY_THRESHOLD = 0.35
MAX_GROSS_EXPOSURE = 0.50
KELLY_FRACTION = 0.5
INITIAL_USD = 10000.0
STABILITY_FAVORABLE = 0.90
STABILITY_WARNING = 0.75

# ============================
# FFNN class
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
# EKF
# ============================
def run_ekf(price_series, dt=1.0):
    """Extended Kalman Filter"""
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
# REGIME DETECTION (from main_with_regime.py)
# ============================
def detect_trend_regime(prices, window=20):
    """Detect trending vs mean-reverting regime"""
    returns = prices.pct_change()
    up_move = returns.clip(lower=0)
    down_move = -returns.clip(upper=0)

    up_smooth = up_move.rolling(window).mean()
    down_smooth = down_move.rolling(window).mean()

    total_move = up_smooth + down_smooth
    dx = np.abs(up_smooth - down_smooth) / (total_move + 1e-10)
    adx = dx.rolling(window).mean()

    regime = (adx > 0.4).astype(int)
    return regime, adx

def detect_volatility_regime(prices, window=20, threshold=1.0):
    """Detect high vs low volatility regime"""
    returns = prices.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(365)
    median_vol = vol.rolling(window*5).median()
    regime = (vol > threshold * median_vol).astype(int)
    return regime, vol

def detect_regime_stability(trend_regime, vol_regime, window=20):
    """Detect regime stability"""
    trend_changes = (trend_regime != trend_regime.shift(1)).astype(int)
    vol_changes = (vol_regime != vol_regime.shift(1)).astype(int)

    trend_stability = 1 - (trend_changes.rolling(window).sum() / window)
    vol_stability = 1 - (vol_changes.rolling(window).sum() / window)

    stability = (trend_stability + vol_stability) / 2
    return stability

def check_regime_status(prices):
    """
    Check current market regime and return multiplier

    Returns:
        multiplier: 0.0 (WARNING), 0.5 (CAUTION), 0.75 (NEUTRAL), 1.0 (FAVORABLE)
        status: regime status string
    """
    trend_regime, trend_strength = detect_trend_regime(prices)
    vol_regime, volatility = detect_volatility_regime(prices)
    stability = detect_regime_stability(trend_regime, vol_regime)

    current_trend = trend_regime.iloc[-1]
    current_stability = stability.iloc[-1]

    if current_trend == 0 and current_stability >= STABILITY_FAVORABLE:
        # FAVORABLE: Mean-reverting + stable
        return 1.0, "FAVORABLE"
    elif current_trend == 1 and current_stability >= STABILITY_FAVORABLE:
        # CAUTION: Trending but stable
        return 0.5, "CAUTION"
    elif current_stability < STABILITY_WARNING:
        # WARNING: Unstable
        return 0.0, "WARNING"
    else:
        # NEUTRAL: Mixed
        return 0.75, "NEUTRAL"

# ============================
# Download BTC data from Alpha Vantage
# ============================
def download_btc_alphavantage(api_key, start_date="2022-01-01"):
    """Download BTC from Alpha Vantage"""
    print(f"Fetching BTC data from Alpha Vantage (from {start_date})...")

    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'BTC',
        'market': 'USD',
        'apikey': api_key,
        'outputsize': 'full'
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

        time_series = data['Time Series (Digital Currency Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Find close column
        possible_close_cols = ['4a. close (USD)', '4. close', 'close', '4b. close (USD)']
        close_col = None

        for col in possible_close_cols:
            if col in df.columns:
                close_col = col
                break

        if close_col is None:
            close_col = next((col for col in df.columns if 'close' in col.lower()), None)

        if close_col is None:
            raise ValueError(f"Could not find close price column. Available: {df.columns.tolist()}")

        print(f"Using column: '{close_col}'")
        close = df[close_col].astype(float)
        close = close[close.index >= start_date]

        print(f"✓ Downloaded {len(close)} days from Alpha Vantage")
        return close

    except Exception as e:
        print(f"✗ Error downloading from Alpha Vantage: {e}")
        raise

def get_historical_funding(dates):
    """Generate synthetic funding rates"""
    np.random.seed(42)
    n = len(dates)
    base_funding = np.cumsum(np.random.normal(0, 0.0005, n))
    seasonal = 0.0002 * np.sin(np.arange(n) * 2 * np.pi / 180)
    noise = 0.0001 * np.random.randn(n)
    funding = np.tanh(base_funding + seasonal) + noise
    return pd.Series(funding, index=dates)

# ============================
# Load models
# ============================
print("="*80)
print("REGIME-AWARE BACKTEST - EKF + FFNN + HyperDUM + Regime Detection")
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
    print("Run train_models.py first")
    exit(1)

# ============================
# Download data
# ============================
print("Downloading BTC historical data...")
START_DATE = "2022-01-01"

ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

if not ALPHAVANTAGE_API_KEY:
    print("\nALPHA VANTAGE API KEY REQUIRED")
    print("Add to .env file: ALPHAVANTAGE_API_KEY=your_key_here")
    ALPHAVANTAGE_API_KEY = input("Or enter now: ").strip()
    if not ALPHAVANTAGE_API_KEY:
        exit(1)

print(f"✓ API key loaded\n")

try:
    close = download_btc_alphavantage(ALPHAVANTAGE_API_KEY, START_DATE)
except Exception as e:
    print(f"\n✗ Failed to download data: {e}")
    exit(1)

btc = pd.DataFrame({'close': close})
btc['return'] = np.log(btc['close'] / btc['close'].shift(1))
btc = btc.dropna()

if len(btc) == 0:
    print("✗ Error: No data downloaded")
    exit(1)

print(f"Data range: {btc.index[0]:%Y-%m-%d} to {btc.index[-1]:%Y-%m-%d}\n")
btc['funding'] = get_historical_funding(btc.index)

# ============================
# Backtest loop with REGIME DETECTION
# ============================
print("Running regime-aware backtest...")
print("="*80)

cash = INITIAL_USD
position = 0.0
equity_curve = []
trade_log = []
regime_log = []

START_IDX = 60

for i in range(START_IDX, len(btc)):
    current_date = btc.index[i]
    price = btc['close'].iloc[i]
    actual_return = btc['return'].iloc[i]

    hist_close = btc['close'].iloc[:i+1]
    hist_returns = btc['return'].iloc[:i+1]

    # Run EKF
    level_series, velocity_series = run_ekf(hist_close)
    level = level_series.iloc[-1]
    velocity = velocity_series.iloc[-1]

    funding = btc['funding'].iloc[i]
    recent_ret = hist_returns.iloc[-5:].mean() if len(hist_returns) >= 5 else 0.0
    rel_price = price / hist_close.iloc[-30:].mean() - 1 if len(hist_close) >= 30 else 0.0

    feat = np.array([[level, velocity, funding, recent_ret, rel_price]])
    feat_s = scaler.transform(feat)

    # FFNN prediction
    with torch.no_grad():
        pred = model(torch.FloatTensor(feat_s).to(device)).item()

    # HyperDUM
    projected = np.sign(feat_s @ projector)
    hamming_dist = np.mean(projected != memory_vector)

    # Volatility
    recent_vol = hist_returns.iloc[-60:].std() * np.sqrt(252) if len(hist_returns) >= 60 else 0.20

    # REGIME CHECK (NEW!)
    regime_multiplier, regime_status = check_regime_status(hist_close)

    # Track regime
    regime_log.append({
        'date': current_date,
        'regime_status': regime_status,
        'regime_multiplier': regime_multiplier
    })

    # Risk gates
    gross_exposure = abs(position * price) / max(cash + position * price, 1.0)

    target = 0.0
    skip_reason = None

    # REGIME GATE (first check)
    if regime_multiplier == 0.0:
        skip_reason = "Regime-WARNING"
    # HYPERDUM GATE
    elif hamming_dist > UNCERTAINTY_THRESHOLD:
        skip_reason = "HyperDUM"
    # RISK GATE
    elif gross_exposure > MAX_GROSS_EXPOSURE and abs(pred) > 0:
        skip_reason = "RiskGate"
    else:
        # Calculate target with REGIME MULTIPLIER
        risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
        base_target = np.sign(pred) * risk * KELLY_FRACTION
        target = base_target * regime_multiplier  # APPLY REGIME MULTIPLIER

    # Calculate position
    equity = cash + position * price
    target_position_value = equity * target
    target_position_size = target_position_value / price

    # Execute
    position_diff = target_position_size - position
    if abs(position_diff) > 0.001:
        trade_side = "BUY" if position_diff > 0 else "SELL"
        trade_log.append({
            'date': current_date,
            'side': trade_side,
            'size': abs(position_diff),
            'price': price,
            'regime_status': regime_status,
            'regime_mult': regime_multiplier
        })
        position += position_diff
        cash -= position_diff * price

    # Apply return
    if i < len(btc) - 1:
        position_return = position * price * actual_return
        cash += position_return

    # Track equity
    equity = cash + position * price
    equity_curve.append({
        'date': current_date,
        'equity': equity,
        'position': position,
        'price': price,
        'skip_reason': skip_reason,
        'regime_status': regime_status,
        'regime_mult': regime_multiplier
    })

# ============================
# Calculate metrics
# ============================
print("\n" + "="*80)
print("REGIME-AWARE BACKTEST RESULTS")
print("="*80)

equity_df = pd.DataFrame(equity_curve).set_index('date')
trades_df = pd.DataFrame(trade_log)
regime_df = pd.DataFrame(regime_log).set_index('date')

equity_df['equity_return'] = equity_df['equity'].pct_change()
equity_df['btc_return'] = equity_df['price'].pct_change()

equity_df['strategy_cumulative'] = (1 + equity_df['equity_return']).cumprod()
equity_df['btc_cumulative'] = (1 + equity_df['btc_return']).cumprod()

returns = equity_df['equity_return'].dropna()
btc_returns = equity_df['btc_return'].dropna()

total_return = (equity_df['equity'].iloc[-1] / INITIAL_USD - 1) * 100
btc_total_return = (equity_df['price'].iloc[-1] / equity_df['price'].iloc[0] - 1) * 100

sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
btc_sharpe = np.mean(btc_returns) / np.std(btc_returns) * np.sqrt(252) if np.std(btc_returns) > 0 else 0

downside_returns = returns[returns < 0]
sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0

cumulative = equity_df['strategy_cumulative']
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min() * 100

btc_cumulative = equity_df['btc_cumulative']
btc_running_max = btc_cumulative.expanding().max()
btc_drawdown = (btc_cumulative - btc_running_max) / btc_running_max
btc_max_dd = btc_drawdown.min() * 100

calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else 0

winning_days = len(returns[returns > 0])
win_rate = winning_days / len(returns) if len(returns) > 0 else 0

num_trades = len(trades_df)
days_skipped_regime = len(equity_df[equity_df['skip_reason'] == 'Regime-WARNING'])
days_skipped_hyperdum = len(equity_df[equity_df['skip_reason'] == 'HyperDUM'])
days_with_position = len(equity_df[abs(equity_df['position']) > 0.001])

# Print results
print(f"\nPeriod: {equity_df.index[0]:%Y-%m-%d} to {equity_df.index[-1]:%Y-%m-%d}")
print(f"Trading Days: {len(equity_df)}")
print(f"\n{'─'*80}")
print("PERFORMANCE METRICS")
print(f"{'─'*80}")
print(f"{'Metric':<30} {'Strategy':>15} {'Buy & Hold':>15} {'Improvement':>15}")
print(f"{'─'*80}")
print(f"{'Total Return':<30} {total_return:>14.2f}% {btc_total_return:>14.2f}% {total_return-btc_total_return:>+14.2f}%")
print(f"{'Sharpe Ratio':<30} {sharpe:>15.3f} {btc_sharpe:>15.3f}")
print(f"{'Sortino Ratio':<30} {sortino:>15.3f} {'N/A':>15}")
print(f"{'Calmar Ratio':<30} {calmar:>15.3f} {'N/A':>15}")
print(f"{'Max Drawdown':<30} {max_dd:>14.2f}% {btc_max_dd:>14.2f}% {max_dd-btc_max_dd:>+14.2f}%")
print(f"{'Win Rate':<30} {win_rate:>14.1%} {'N/A':>15}")

print(f"\n{'─'*80}")
print("REGIME STATISTICS")
print(f"{'─'*80}")
regime_counts = regime_df['regime_status'].value_counts()
for status in ['FAVORABLE', 'CAUTION', 'NEUTRAL', 'WARNING']:
    count = regime_counts.get(status, 0)
    pct = (count / len(regime_df) * 100) if len(regime_df) > 0 else 0
    print(f"{status:<20} {count:>6} days ({pct:>5.1f}%)")

print(f"\n{'─'*80}")
print("TRADE STATISTICS")
print(f"{'─'*80}")
print(f"Total Trades:             {num_trades}")
print(f"Days with Position:       {days_with_position} ({days_with_position/len(equity_df)*100:.1f}%)")
print(f"Days Skipped (Regime):    {days_skipped_regime} ({days_skipped_regime/len(equity_df)*100:.1f}%)")
print(f"Days Skipped (HyperDUM):  {days_skipped_hyperdum} ({days_skipped_hyperdum/len(equity_df)*100:.1f}%)")

# Year-by-year
print(f"\n{'─'*80}")
print("YEAR-BY-YEAR PERFORMANCE (Regime-Aware)")
print(f"{'─'*80}")
equity_df['year'] = equity_df.index.year
for year in equity_df['year'].unique():
    year_data = equity_df[equity_df['year'] == year]
    if len(year_data) > 0:
        year_total = (year_data['equity'].iloc[-1] / year_data['equity'].iloc[0] - 1) * 100
        btc_year_total = (year_data['price'].iloc[-1] / year_data['price'].iloc[0] - 1) * 100
        year_trades = len(trades_df[trades_df['date'].dt.year == year])
        print(f"{year}: Return={year_total:+.2f}% (BTC: {btc_year_total:+.2f}%), Trades={year_trades}")

# Save results
equity_df.to_csv('backtest_regime_aware_results.csv')
if len(trades_df) > 0:
    trades_df.to_csv('backtest_regime_aware_trades.csv', index=False)

print(f"\n{'─'*80}")
print("Results saved to:")
print("  - backtest_regime_aware_results.csv")
print("  - backtest_regime_aware_trades.csv")
print("="*80)
