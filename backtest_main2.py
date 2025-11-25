"""
Backtest script for main2.py trading logic
Tests the same strategy with proper risk management
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import requests
import warnings
warnings.filterwarnings('ignore')

# ============================
# Define FFNN class (must match training)
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
# EKF state estimator (same as main2.py)
# ============================
def run_ekf(prices):
    n = len(prices)
    x = np.zeros((n, 3))
    x[0] = [prices[0], 0.0, np.log(0.02)]
    P = np.eye(3)
    Q = np.diag([1e-4, 1e-6, 1e-8])
    R = 0.5**2
    
    level = np.zeros(n)
    velocity = np.zeros(n)
    
    for t in range(1, n):
        F = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        x_pred = F @ x[t-1]
        P_pred = F @ P @ F.T + Q
        
        y = prices[t] - x_pred[0]
        S = P_pred[0,0] + R
        K = P_pred[:,0] / S
        
        x[t] = x_pred + K * y
        P = (np.eye(3) - np.outer(K, [1,0,0])) @ P_pred
        
        level[t] = x[t][0]
        velocity[t] = x[t][1]
    
    return level[-1], velocity[-1]

# ============================
# HyperDUM uncertainty
# ============================
def get_uncertainty(feat_scaled, projector, memory_vector):
    hd = np.sign(feat_scaled @ projector)
    hamming = np.mean(hd != memory_vector)
    return hamming

# ============================
# Load models
# ============================
print("Loading models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.load("btc_model.pth", map_location=device, weights_only=False)
scaler = torch.load("btc_scaler.pth", map_location=device, weights_only=False)
projector = np.load("projector.npy")
memory_vector = np.load("memory.npy")
model.eval()
model.to(device)

# ============================
# Configuration (same as main2.py)
# ============================
VOL_TARGET = 0.20
UNCERTAINTY_THRESHOLD = 0.385
MAX_TRADE_USD = 5.0
INITIAL_USD = 100.0

# ============================
# Download test data
# ============================
print("\nDownloading BTC data for backtest...")
btc = yf.download("BTC-USD", start="2022-01-01", end="2025-11-20", interval="1d")

if isinstance(btc.columns, pd.MultiIndex):
    btc = btc.xs('Close', axis=1, level=0)

if len(btc.columns) == 1:
    btc.columns = ['Close']

btc['return'] = np.log(btc['Close'] / btc['Close'].shift(1))
btc = btc.dropna()

# Use synthetic funding rate for backtest (or you could fetch historical)
# In live trading, main2.py fetches real funding rates
np.random.seed(42)
funding = np.tanh(np.cumsum(np.random.normal(0, 0.0005, len(btc))) +
                 0.0002 * np.sin(np.arange(len(btc)) * 2 * np.pi / 180))
btc['funding_rate'] = funding + 0.0001 * np.random.randn(len(btc))

# ============================
# Backtest (using main2.py logic)
# ============================
print("\nRunning backtest with main2.py logic...")
print("="*60)

cash = INITIAL_USD
position_btc = 0.0
equity_history = [INITIAL_USD]
trades_taken = 0
trades_skipped_hyperdum = 0
trades_skipped_exposure = 0
position_history = [0.0]

# Need at least 60 days for vol calculation
for i in range(60, len(btc)):
    price = btc['Close'].iloc[i]
    actual_return = btc['return'].iloc[i]
    
    # Get price history up to current point
    price_history = btc['Close'].iloc[:i+1].values
    
    # EKF state
    level, velocity = run_ekf(price_history)
    
    # Build feature vector (same as main2.py)
    recent_ret = btc['return'].iloc[i-4:i+1].mean() if i >= 4 else 0.0
    rel_price = price / btc['Close'].iloc[i-30:i].mean() - 1 if i >= 30 else 0.0
    funding = btc['funding_rate'].iloc[i]
    
    feat = np.array([[level, velocity, funding, recent_ret, rel_price]])
    feat_s = scaler.transform(feat)
    
    # Prediction + uncertainty
    with torch.no_grad():
        pred_return = model(torch.FloatTensor(feat_s).to(device)).item()
    uncertainty = get_uncertainty(feat_s, projector, memory_vector)
    
    # Decision logic (same as main2.py)
    if uncertainty > UNCERTAINTY_THRESHOLD:
        target = 0.0
        trades_skipped_hyperdum += 1
    else:
        vol_60d = btc['return'].iloc[i-60:i].std() * np.sqrt(252) if i >= 60 else 0.20
        risk = min(0.5, VOL_TARGET / max(vol_60d, 0.01))
        target = np.sign(pred_return) * risk
    
    # Execute (simplified - no MAX_TRADE_USD cap in backtest for simplicity)
    # Calculate return from current position
    period_return = position_btc * actual_return / (cash + position_btc * price) if (cash + position_btc * price) > 0 else 0
    equity = cash + position_btc * price * (1 + actual_return)
    
    # Update position (simplified execution)
    if target > 0 and position_btc <= 0:
        # Buy
        size_btc = (equity * target) / price
        if size_btc >= 0.0001:
            trades_taken += 1
            position_btc = size_btc
            cash = equity - position_btc * price
    elif target < 0 and position_btc >= 0:
        # Sell
        size_btc = abs(position_btc * target)
        if size_btc >= 0.0001:
            trades_taken += 1
            position_btc = position_btc - size_btc
            cash = equity - position_btc * price
    else:
        # Hold or exit
        if abs(target) < 0.001 and position_btc > 0:
            # Exit position
            cash = equity
            position_btc = 0.0
    
    equity_history.append(equity)
    position_history.append(position_btc)

# ============================
# Calculate metrics
# ============================
equity_array = np.array(equity_history)
returns = np.diff(equity_array) / equity_array[:-1]
cum_returns = np.cumprod(1 + returns)
total_return = (equity_array[-1] / equity_array[0] - 1) * 100
sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
max_dd = np.min(cum_returns / np.maximum.accumulate(cum_returns) - 1)
win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
days_with_position = sum(1 for pos in position_history if abs(pos) > 0.001)

print("\n" + "="*60)
print("BACKTEST RESULTS (main2.py logic)")
print("="*60)
print(f"Period: {btc.index[60]:%Y-%m-%d} to {btc.index[-1]:%Y-%m-%d}")
print(f"Total Return: {total_return:+.2f}%")
print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"Max Drawdown: {max_dd*100:.2f}%")
print(f"Win Rate: {win_rate:.1%}")
print(f"\nTrade Statistics:")
print(f"  Trades Taken: {trades_taken}")
print(f"  Days Skipped (HyperDUM): {trades_skipped_hyperdum} ({trades_skipped_hyperdum/(len(btc)-60)*100:.1f}%)")
print(f"  Days with Position: {days_with_position} ({days_with_position/(len(btc)-60)*100:.1f}%)")
print("="*60)

if trades_taken == 0:
    print("\n⚠ WARNING: No trades taken! HyperDUM blocked all trades.")
    print(f"   Consider lowering UNCERTAINTY_THRESHOLD (currently {UNCERTAINTY_THRESHOLD})")
elif trades_skipped_hyperdum > (len(btc)-60) * 0.8:
    print(f"\n⚠ NOTE: HyperDUM blocked >80% of days ({trades_skipped_hyperdum/(len(btc)-60)*100:.1f}%)")
    print(f"   This may be too conservative for this period")


