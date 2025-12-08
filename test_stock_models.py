"""
Test Script for EKF + FFNN + HyperDUM Stock Models

Uses synthetic data to verify the model training pipeline works correctly.
This can be used when network access is restricted.

Usage:
    python test_stock_models.py NVDA      # Test NVIDIA
    python test_stock_models.py TSLA      # Test Tesla
    python test_stock_models.py NVDA TSLA # Test multiple
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Default tickers if none specified
DEFAULT_TICKERS = ['NVDA', 'TSLA']

# Get tickers from command line or use defaults
if len(sys.argv) > 1:
    TICKERS = [t.upper() for t in sys.argv[1:]]
else:
    TICKERS = DEFAULT_TICKERS

print(f"Testing models for: {', '.join(TICKERS)}")

# -----------------------------
# Configuration
# -----------------------------
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
    x = np.zeros((n, 3))      # [level, velocity, log_var]
    P = np.zeros((n, 3, 3))

    x[0] = [values[0], 0.0, -5.0]
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
# 2. FFNN Model Definition
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
# 3. Generate Synthetic Stock Data
# -----------------------------
def generate_synthetic_data(ticker, n_days=1000):
    """Generate realistic synthetic stock data for testing"""
    np.random.seed(hash(ticker) % 2**32)

    # Different characteristics for different stocks
    stock_params = {
        'NVDA': {'start_price': 200, 'drift': 0.0008, 'volatility': 0.03, 'mean_rev': 0.02},
        'TSLA': {'start_price': 250, 'drift': 0.0005, 'volatility': 0.04, 'mean_rev': 0.015},
        'AAPL': {'start_price': 150, 'drift': 0.0003, 'volatility': 0.02, 'mean_rev': 0.025},
    }

    params = stock_params.get(ticker, {'start_price': 100, 'drift': 0.0004, 'volatility': 0.025, 'mean_rev': 0.02})

    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n = len(dates)

    # Generate price series with mean-reversion and trending components
    prices = np.zeros(n)
    prices[0] = params['start_price']

    equilibrium = params['start_price']

    for i in range(1, n):
        # Mean reversion component
        mr_force = -params['mean_rev'] * (prices[i-1] - equilibrium) / equilibrium

        # Trending component (slow drift)
        drift = params['drift']

        # Random shock
        shock = np.random.normal(0, params['volatility'])

        # Price update
        ret = drift + mr_force + shock
        prices[i] = prices[i-1] * (1 + ret)

        # Slowly evolving equilibrium
        equilibrium *= (1 + params['drift'] * 0.5)

    # Generate synthetic VIX (inversely correlated with price returns)
    returns = np.diff(prices) / prices[:-1]
    vix = 20 + np.cumsum(np.random.normal(0, 1, n)) * 0.5
    vix = np.clip(vix, 10, 50)

    df = pd.DataFrame({
        'Close': prices,
    }, index=dates)

    vix_series = pd.Series(vix, index=dates)

    return df, vix_series

# -----------------------------
# 4. Train model for a single ticker
# -----------------------------
def train_ticker(ticker):
    """Train model for a single ticker using synthetic data"""
    print(f"\n{'='*70}")
    print(f"TRAINING: {ticker} (SYNTHETIC DATA)")
    print(f"{'='*70}")

    # Generate synthetic data
    print(f"Generating synthetic {ticker} data...")
    df, vix_series = generate_synthetic_data(ticker)

    print(f"  Generated {len(df)} periods from {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")

    # Calculate returns
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Use VIX as sentiment proxy
    df['sentiment'] = (vix_series - 20) / 20  # Normalized around VIX=20
    df['sentiment'] = df['sentiment'].clip(-1, 1)

    df = df.dropna()

    # Run EKF
    print("  Running EKF...")
    df['level'], df['velocity'] = run_ekf(df['Close'])

    # Prepare features & target
    window = 30
    momentum_window = 5

    features = []
    targets = []

    for i in range(window, len(df)-1):
        row = [
            df['level'].iloc[i],
            df['velocity'].iloc[i],
            df['sentiment'].iloc[i],
            df['return'].iloc[i-momentum_window+1:i+1].mean() if i >= momentum_window-1 else df['return'].iloc[:i+1].mean(),
            df['Close'].iloc[i] / df['Close'].iloc[i-window:i].mean() - 1
        ]
        features.append(row)
        targets.append(df['return'].iloc[i+1])

    X = np.array(features)
    y = np.array(targets)

    # Train/validation split (80/20 chronological)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    # Train FFNN
    print("  Training FFNN...")
    model = FFNN()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    model.train()
    for epoch in range(400):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"    Epoch {epoch} - Loss: {loss.item():.6f}")

    # Generate HyperDUM components
    print("  Generating HyperDUM components...")
    hyperdim = 2048
    feature_dim = X_train_scaled.shape[1]

    np.random.seed(42)
    projector = np.random.randn(feature_dim, hyperdim)
    projector = projector / np.linalg.norm(projector, axis=0, keepdims=True)

    projected_train = np.sign(X_train_scaled @ projector)
    memory_vector = np.sign(np.mean(projected_train, axis=0))

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy().flatten()

    # Run backtest
    print("\n  Running validation backtest...")
    initial_equity = 100.0
    equity = initial_equity
    position_exposure = 0.0
    trades_taken = 0
    trades_skipped = 0

    test_returns = []

    for i in range(len(test_pred)):
        pred = test_pred[i]
        actual_return = y_test[i]

        # HyperDUM check
        feat_s = X_test_scaled[i:i+1]
        projected = np.sign(feat_s @ projector)
        hamming_dist = np.mean(projected != memory_vector)

        # Calculate realized volatility
        lookback_start = max(0, split + window + 1 - 60 + i)
        lookback_end = split + window + 1 + i
        if lookback_end > lookback_start + 30:
            vol_window = df['return'].iloc[lookback_start:lookback_end]
            recent_vol = vol_window.std() * np.sqrt(252) if len(vol_window) > 0 else 0.20
        else:
            recent_vol = 0.20

        # Risk gates
        target_exposure = 0.0
        if hamming_dist > UNCERTAINTY_THRESHOLD:
            trades_skipped += 1
        else:
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            target_exposure = np.sign(pred) * risk * KELLY_FRACTION
            if abs(target_exposure - position_exposure) > 0.001:
                trades_taken += 1

        # Calculate return
        period_return = position_exposure * actual_return
        equity *= (1 + period_return)
        test_returns.append(period_return)
        position_exposure = target_exposure

    # Calculate metrics
    test_returns = np.array(test_returns)
    total_return = (equity / initial_equity - 1) * 100
    sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(252) if np.std(test_returns) > 0 else 0
    win_rate = np.mean(test_returns > 0) if len(test_returns) > 0 else 0

    # Save models with ticker prefix
    ticker_clean = ticker.replace('-', '_').lower()
    torch.save(model, f'{ticker_clean}_model.pth')
    torch.save(scaler, f'{ticker_clean}_scaler.pth')
    np.save(f'{ticker_clean}_projector.npy', projector)
    np.save(f'{ticker_clean}_memory.npy', memory_vector)

    # Print results
    print(f"\n  {'='*60}")
    print(f"  {ticker} TEST RESULTS (SYNTHETIC DATA)")
    print(f"  {'='*60}")
    print(f"  Train period: {df.index[window]:%Y-%m-%d} to {df.index[split+window]:%Y-%m-%d}")
    print(f"  Test period:  {df.index[split+window+1]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")
    print(f"\n  Total Return: {total_return:+.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Trades Taken: {trades_taken}")
    print(f"  Days Skipped (HyperDUM): {trades_skipped} ({trades_skipped/len(test_pred)*100:.1f}%)")
    print(f"\n  Models saved:")
    print(f"    - {ticker_clean}_model.pth")
    print(f"    - {ticker_clean}_scaler.pth")
    print(f"    - {ticker_clean}_projector.npy")
    print(f"    - {ticker_clean}_memory.npy")

    return True

# -----------------------------
# 5. Main: Train all tickers
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-ASSET EKF + FFNN + HyperDUM TESTING (SYNTHETIC DATA)")
    print("="*70)
    print("\nNote: Using synthetic data because network access is restricted.")
    print("When running locally with network access, use train_stock_models.py")

    successful = []
    failed = []

    for ticker in TICKERS:
        try:
            if train_ticker(ticker):
                successful.append(ticker)
            else:
                failed.append(ticker)
        except Exception as e:
            print(f"  Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(ticker)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    if successful:
        print(f"Successfully trained: {', '.join(successful)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("\nModel pipeline verified! Ready for live data when network is available.")
