"""
Test merged changes using cached regime_analysis.csv data.
Validates the threshold sweep results and shows detailed performance metrics.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Configuration from config.py (merged changes)
VOL_TARGET = 0.20
UNCERTAINTY_THRESHOLD = 0.35
STABILITY_FAVORABLE = 0.90
STABILITY_WARNING = 0.75
MAX_GROSS_EXPOSURE = 0.50
KELLY_FRACTION = 0.25
INITIAL_USD = 10_000.0

# Test period
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-12-01")
TRAIN_END = pd.Timestamp("2022-12-31")

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

def get_funding_rates(n, seed=42):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.0005, n))
    seasonal = 0.0002 * np.sin(np.arange(n) * 2 * np.pi / 180)
    noise = 0.0001 * rng.standard_normal(n)
    return np.tanh(base + seasonal) + noise

def detect_trend_regime(prices, window=20):
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
    returns = prices.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(365)
    median_vol = vol.rolling(window * 5).median()
    regime = (vol > threshold * median_vol).astype(int)
    return regime, vol

def detect_regime_stability(trend_regime, vol_regime, window=20):
    trend_changes = (trend_regime != trend_regime.shift(1)).astype(int)
    vol_changes = (vol_regime != vol_regime.shift(1)).astype(int)
    trend_stability = 1 - (trend_changes.rolling(window).sum() / window)
    vol_stability = 1 - (vol_changes.rolling(window).sum() / window)
    return (trend_stability + vol_stability) / 2

def get_regime_multiplier(trend_regime, stability, stable_cutoff, warning_cutoff):
    if stability < warning_cutoff:
        return 0.0, "WARNING"
    if stability >= stable_cutoff and trend_regime == 0:
        return 1.0, "FAVORABLE"
    if stability >= stable_cutoff and trend_regime == 1:
        return 0.5, "CAUTION"
    return 0.75, "NEUTRAL"

def main():
    print("="*70)
    print("TESTING MERGED CHANGES - Threshold Sweep & Risk Defaults")
    print("="*70)
    print(f"\nConfiguration (from merged config.py):")
    print(f"  UNCERTAINTY_THRESHOLD: {UNCERTAINTY_THRESHOLD}")
    print(f"  STABILITY_FAVORABLE:   {STABILITY_FAVORABLE}")
    print(f"  STABILITY_WARNING:     {STABILITY_WARNING}")
    print(f"  KELLY_FRACTION:        {KELLY_FRACTION}")
    print(f"  MAX_GROSS_EXPOSURE:    {MAX_GROSS_EXPOSURE}")

    # Load cached data
    print("\n" + "-"*70)
    print("Loading Data")
    print("-"*70)
    csv_path = Path("regime_analysis.csv")
    raw = pd.read_csv(csv_path, parse_dates=["Date"])
    raw = raw.rename(columns={"Date": "date", "price": "Close", "ekf_level": "level", "ekf_velocity": "velocity"})
    raw = raw.set_index("date").sort_index()
    raw["return"] = np.log(raw["Close"] / raw["Close"].shift(1))
    raw["funding_rate"] = get_funding_rates(len(raw))
    raw["momentum_5"] = raw["return"].rolling(5).mean()
    raw["rel_price"] = raw["Close"] / raw["Close"].rolling(30).mean() - 1
    btc = raw.dropna()

    print(f"Data range: {btc.index[0]:%Y-%m-%d} to {btc.index[-1]:%Y-%m-%d}")
    print(f"Total days: {len(btc)}")

    # Train model
    print("\n" + "-"*70)
    print("Training Model")
    print("-"*70)
    features = btc[["level", "velocity", "funding_rate", "momentum_5", "rel_price"]]
    targets = btc["return"].shift(-1)

    train = btc.index <= TRAIN_END
    X_train = features[train].dropna()
    y_train = targets.loc[X_train.index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = FFNN()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)

    model.train()
    for epoch in range(250):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    print(f"Training samples: {len(X_train)}")
    print(f"Final loss: {loss.item():.6f}")

    # HyperDUM components
    hyperdim = 2048
    rng = np.random.default_rng(42)
    projector = rng.standard_normal((X_train_scaled.shape[1], hyperdim))
    projector = projector / np.linalg.norm(projector, axis=0, keepdims=True)
    projected_train = np.sign(X_train_scaled @ projector)
    memory_vector = np.sign(np.mean(projected_train, axis=0))

    # Prepare test data
    X_all_scaled = scaler.transform(features.loc[X_train.index.union(btc.index[btc.index > TRAIN_END])])
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_all_scaled)).squeeze().numpy()

    feature_frame = features.loc[X_train.index.union(btc.index[btc.index > TRAIN_END])].copy()
    feature_frame["pred_return"] = preds

    projected_all = np.sign(X_all_scaled @ projector)
    hamming = np.mean(projected_all != memory_vector.reshape(1, -1), axis=1)
    feature_frame["hyperdum"] = hamming

    merged = btc.loc[feature_frame.index].copy()
    merged[["pred_return", "hyperdum"]] = feature_frame[["pred_return", "hyperdum"]]

    # Run backtest
    print("\n" + "-"*70)
    print("Running Backtest with Merged Config")
    print("-"*70)

    start_mask = (merged.index >= TEST_START) & (merged.index < TEST_END)
    data = merged.loc[start_mask].copy()

    print(f"Test period: {data.index[0]:%Y-%m-%d} to {data.index[-1]:%Y-%m-%d}")
    print(f"Test days: {len(data)}")

    # Compute regimes
    trend_regime, _ = detect_trend_regime(merged["Close"])
    vol_regime, _ = detect_volatility_regime(merged["Close"])
    stability = detect_regime_stability(trend_regime, vol_regime)

    # Backtest
    cash = INITIAL_USD
    position = 0.0
    equity_curve = []
    returns_list = []
    trades = 0
    blocked_hyperdum = 0
    blocked_regime = 0
    regime_stats = {"FAVORABLE": 0, "CAUTION": 0, "WARNING": 0, "NEUTRAL": 0}

    for idx in data.index:
        price = data.loc[idx, "Close"]
        actual_return = data.loc[idx, "return"]

        t_idx = trend_regime.index.get_loc(idx)
        regime_mult, regime_name = get_regime_multiplier(
            trend_regime.iloc[t_idx],
            stability.iloc[t_idx],
            STABILITY_FAVORABLE,
            STABILITY_WARNING
        )
        regime_stats[regime_name] += 1

        hamming_dist = data.loc[idx, "hyperdum"]

        if hamming_dist > UNCERTAINTY_THRESHOLD:
            target = 0.0
            blocked_hyperdum += 1
        elif regime_mult == 0.0:
            target = 0.0
            blocked_regime += 1
        else:
            recent_vol = data["return"].loc[:idx].tail(60).std() * np.sqrt(252)
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            target = np.sign(data.loc[idx, "pred_return"]) * risk * KELLY_FRACTION * regime_mult

        equity = cash + position * price
        target_value = equity * target
        target_position = target_value / price
        delta = target_position - position

        if abs(delta) > 0.0001:
            position += delta
            cash -= delta * price
            trades += 1

        equity = cash + position * price
        if not np.isnan(actual_return):
            cash += position * price * actual_return
            equity = cash + position * price
            equity_curve.append(equity)
            if len(equity_curve) > 1:
                returns_list.append((equity_curve[-1] - equity_curve[-2]) / equity_curve[-2])
            else:
                returns_list.append(0.0)

    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])
    eq_returns = pd.Series(returns_list, index=equity_series.index)

    cumulative = (1 + eq_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    total_return = (equity_series.iloc[-1] / INITIAL_USD - 1)
    sharpe = eq_returns.mean() / eq_returns.std() * np.sqrt(252) if eq_returns.std() > 0 else 0
    sortino = eq_returns.mean() / eq_returns[eq_returns < 0].std() * np.sqrt(252) if len(eq_returns[eq_returns < 0]) > 0 else 0
    calmar = (total_return * 252 / len(data)) / abs(max_dd) if max_dd != 0 else 0
    win_rate = (eq_returns > 0).mean()

    # Buy and hold comparison
    bh_return = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)

    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-"*42)
    print(f"{'Total Return':<25} {total_return*100:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {sharpe:>15.3f}")
    print(f"{'Sortino Ratio':<25} {sortino:>15.3f}")
    print(f"{'Max Drawdown':<25} {max_dd*100:>14.2f}%")
    print(f"{'Calmar Ratio':<25} {calmar:>15.3f}")
    print(f"{'Win Rate':<25} {win_rate*100:>14.1f}%")
    print(f"{'Buy & Hold':<25} {bh_return*100:>14.2f}%")
    print(f"{'Final Equity':<25} ${equity_series.iloc[-1]:>13,.2f}")

    print(f"\n{'Trading Activity':<25}")
    print("-"*42)
    print(f"{'Total Trades':<25} {trades:>15,}")
    print(f"{'Days Blocked (HyperDUM)':<25} {blocked_hyperdum:>15,}")
    print(f"{'Days Blocked (Regime)':<25} {blocked_regime:>15,}")
    print(f"{'Active Trading Days':<25} {len(data) - blocked_hyperdum - blocked_regime:>15,}")

    print(f"\n{'Regime Distribution':<25}")
    print("-"*42)
    for regime, count in regime_stats.items():
        pct = count / len(data) * 100
        print(f"{'  ' + regime:<25} {count:>10,} ({pct:>5.1f}%)")

    # Comparison with sweep results
    print("\n" + "="*70)
    print("VALIDATION vs THRESHOLD SWEEP")
    print("="*70)
    sweep_results = pd.read_csv("threshold_sweep_results.csv")
    best = sweep_results[
        (sweep_results["hyper_threshold"] == 0.35) &
        (sweep_results["stability_threshold"] == 0.90)
    ].iloc[0]

    print(f"\n{'Metric':<20} {'Sweep Result':>15} {'This Test':>15} {'Match':>10}")
    print("-"*62)
    print(f"{'Sharpe':<20} {best['sharpe']:>15.3f} {sharpe:>15.3f} {'✓' if abs(sharpe - best['sharpe']) < 0.5 else '✗':>10}")
    print(f"{'Total Return':<20} {best['total_return']*100:>14.1f}% {total_return*100:>14.1f}% {'✓' if abs(total_return - best['total_return']) < 0.05 else '~':>10}")
    print(f"{'Max Drawdown':<20} {best['max_drawdown']*100:>14.1f}% {max_dd*100:>14.1f}% {'✓' if abs(max_dd - best['max_drawdown']) < 0.05 else '~':>10}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if sharpe > 1.0 and max_dd > -0.10:
        print("\n✓ SUCCESS: Merged changes are performing well!")
        print(f"  - Sharpe Ratio of {sharpe:.2f} indicates good risk-adjusted returns")
        print(f"  - Max Drawdown of {max_dd*100:.1f}% is within acceptable limits")
        print(f"  - Total return of {total_return*100:.1f}% vs {bh_return*100:.1f}% buy & hold")
    elif sharpe > 0.5:
        print("\n~ MODERATE: Strategy is working but could be improved")
        print(f"  - Sharpe of {sharpe:.2f} is positive but not exceptional")
    else:
        print("\n✗ NEEDS REVIEW: Strategy performance is weak")
        print(f"  - Sharpe of {sharpe:.2f} suggests poor risk-adjusted returns")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
