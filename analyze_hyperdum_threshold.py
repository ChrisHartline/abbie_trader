"""
HyperDUM Threshold Calibration using Historical Bull/Bear Market Data

Uses actual Bitcoin market cycle history to find optimal UNCERTAINTY_THRESHOLD:
- Analyzes HyperDUM scores during known bull vs bear periods
- Suggests threshold that maximizes protection in bears while allowing participation in bulls
- Data-driven approach to parameter tuning

Historical Market Regimes (2022-2025):
- Bear: Jan 2022 - Nov 2022 (~10 months, -78%)
- Bull: Nov 2022 - Dec 2025 (~37 months, +704%)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
import warnings
import requests
warnings.filterwarnings('ignore')

load_dotenv()

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
# Download BTC data
# ============================
def download_btc_alphavantage(api_key, start_date="2022-01-01"):
    """Download BTC from Alpha Vantage"""
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
        raise ValueError(f"API error: {data}")

    time_series = data['Time Series (Digital Currency Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Find close column
    possible_cols = ['4a. close (USD)', '4. close', 'close']
    close_col = next((col for col in possible_cols if col in df.columns), None)

    if close_col is None:
        close_col = next((col for col in df.columns if 'close' in col.lower()), None)

    close = df[close_col].astype(float)
    close = close[close.index >= start_date]

    print(f"✓ Downloaded {len(close)} days")
    return close

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
# Label historical regimes
# ============================
def label_market_regime(date):
    """
    Label date as bull or bear based on historical Bitcoin market cycles

    Historical Regimes:
    - Bear: 2022-01-01 to 2022-11-30 (BTC: $69k → $15.5k, -78%)
    - Bull: 2022-12-01 to 2025-12-31 (BTC: $15.5k → $108k+, +704%)
    """
    if pd.Timestamp('2022-01-01') <= date <= pd.Timestamp('2022-11-30'):
        return 'BEAR'
    elif pd.Timestamp('2022-12-01') <= date <= pd.Timestamp('2025-12-31'):
        return 'BULL'
    else:
        return 'UNKNOWN'

# ============================
# Main Analysis
# ============================
print("="*80)
print("HYPERDUM THRESHOLD CALIBRATION")
print("Using Historical Bull/Bear Market Data")
print("="*80)

# Load models
print("\nLoading models...")
device = torch.device("cpu")
model = torch.load("btc_model.pth", map_location=device, weights_only=False)
scaler = torch.load("btc_scaler.pth", map_location=device, weights_only=False)
projector = np.load("projector.npy")
memory_vector = np.load("memory.npy")
model.eval()
print("✓ Models loaded\n")

# Download data
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
if not ALPHAVANTAGE_API_KEY:
    ALPHAVANTAGE_API_KEY = input("Enter Alpha Vantage API key: ").strip()

close = download_btc_alphavantage(ALPHAVANTAGE_API_KEY, "2022-01-01")
btc = pd.DataFrame({'close': close})
btc['return'] = np.log(btc['close'] / btc['close'].shift(1))
btc = btc.dropna()
btc['funding'] = get_historical_funding(btc.index)

# Label regimes
print("Labeling historical regimes...")
btc['regime'] = btc.index.map(label_market_regime)

bear_days = len(btc[btc['regime'] == 'BEAR'])
bull_days = len(btc[btc['regime'] == 'BULL'])
print(f"Bear market days: {bear_days} ({bear_days/len(btc)*100:.1f}%)")
print(f"Bull market days: {bull_days} ({bull_days/len(btc)*100:.1f}%)")

# Calculate HyperDUM scores for each day
print("\nCalculating HyperDUM scores...")
hyperdum_scores = []

for i in range(60, len(btc)):
    current_date = btc.index[i]
    price = btc['close'].iloc[i]

    hist_close = btc['close'].iloc[:i+1]
    hist_returns = btc['return'].iloc[:i+1]

    # Run EKF
    level_series, velocity_series = run_ekf(hist_close)
    level = level_series.iloc[-1]
    velocity = velocity_series.iloc[-1]

    funding = btc['funding'].iloc[i]
    recent_ret = hist_returns.iloc[-5:].mean()
    rel_price = price / hist_close.iloc[-30:].mean() - 1

    feat = np.array([[level, velocity, funding, recent_ret, rel_price]])
    feat_s = scaler.transform(feat)

    # HyperDUM score
    projected = np.sign(feat_s @ projector)
    hamming_dist = np.mean(projected != memory_vector)

    hyperdum_scores.append({
        'date': current_date,
        'hamming_dist': hamming_dist,
        'regime': btc['regime'].iloc[i],
        'price': price
    })

scores_df = pd.DataFrame(hyperdum_scores).set_index('date')

# Analyze scores by regime
print("\n" + "="*80)
print("HYPERDUM SCORE ANALYSIS BY REGIME")
print("="*80)

bear_scores = scores_df[scores_df['regime'] == 'BEAR']['hamming_dist']
bull_scores = scores_df[scores_df['regime'] == 'BULL']['hamming_dist']

print(f"\n🐻 BEAR MARKET (2022-01 to 2022-11):")
print(f"  Mean HyperDUM: {bear_scores.mean():.4f}")
print(f"  Median:        {bear_scores.median():.4f}")
print(f"  Min:           {bear_scores.min():.4f}")
print(f"  Max:           {bear_scores.max():.4f}")
print(f"  25th %ile:     {bear_scores.quantile(0.25):.4f}")
print(f"  75th %ile:     {bear_scores.quantile(0.75):.4f}")

print(f"\n🐂 BULL MARKET (2022-12 to 2025-12):")
print(f"  Mean HyperDUM: {bull_scores.mean():.4f}")
print(f"  Median:        {bull_scores.median():.4f}")
print(f"  Min:           {bull_scores.min():.4f}")
print(f"  Max:           {bull_scores.max():.4f}")
print(f"  25th %ile:     {bull_scores.quantile(0.25):.4f}")
print(f"  75th %ile:     {bull_scores.quantile(0.75):.4f}")

# Test different thresholds
print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION")
print("="*80)

thresholds = [0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50]

print(f"\n{'Threshold':<12} {'Bear Block %':<15} {'Bull Block %':<15} {'Total Block %':<15}")
print("-" * 60)

best_threshold = None
best_score = -999

for thresh in thresholds:
    bear_blocked = (bear_scores > thresh).sum()
    bear_blocked_pct = bear_blocked / len(bear_scores) * 100

    bull_blocked = (bull_scores > thresh).sum()
    bull_blocked_pct = bull_blocked / len(bull_scores) * 100

    total_blocked = (scores_df['hamming_dist'] > thresh).sum()
    total_blocked_pct = total_blocked / len(scores_df) * 100

    # Score: Want high bear block, low bull block
    # Good score = (bear_blocked_pct - bull_blocked_pct)
    score = bear_blocked_pct - bull_blocked_pct

    print(f"{thresh:<12.2f} {bear_blocked_pct:<14.1f}% {bull_blocked_pct:<14.1f}% {total_blocked_pct:<14.1f}%")

    if score > best_score:
        best_score = score
        best_threshold = thresh

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print(f"\n🎯 OPTIMAL THRESHOLD: {best_threshold:.2f}")
print(f"   (Maximizes bear protection while minimizing bull blocking)")

print(f"\n📊 THRESHOLD GUIDELINES:")
print(f"   • Current (0.35): Blocks {(scores_df['hamming_dist'] > 0.35).sum()/len(scores_df)*100:.1f}% of all days")
print(f"   • Conservative (0.35-0.45): more block rate, safer")
print(f"   • Balanced (0.30-0.35): ~20-40% of days blocked ← RECOMMENDED")
print(f"   • Aggressive (0.25-0.30): ~10-20% of days blocked")

# Show what happens with optimal threshold
optimal_bear_block = (bear_scores > best_threshold).sum() / len(bear_scores) * 100
optimal_bull_block = (bull_scores > best_threshold).sum() / len(bull_scores) * 100

print(f"\n✅ With threshold = {best_threshold}:")
print(f"   • Bear market: Blocks {optimal_bear_block:.1f}% of days (protection)")
print(f"   • Bull market: Blocks {optimal_bull_block:.1f}% of days (participation)")
print(f"   • Net benefit: {optimal_bear_block - optimal_bull_block:.1f}% more protection than restriction")

# Save results
scores_df.to_csv('hyperdum_threshold_analysis.csv')
print(f"\n💾 Results saved to: hyperdum_threshold_analysis.csv")
print("="*80)
