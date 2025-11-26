"""
Regime and Pattern Detection Script for Q-Prime Trading System

Analyzes BTC market data to identify:
- Trending vs Mean-Reverting regimes
- High vs Low Volatility periods
- Regime transitions and stability
- Pattern statistics for strategy optimization

Usage: python regime_detector.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Extended Kalman Filter (EKF) - from main.py
# -----------------------------
def run_ekf(price_series, dt=1.0):
    """Extended Kalman Filter for state estimation"""
    if isinstance(price_series, pd.Series):
        values = price_series.values
        index = price_series.index
    else:
        values = np.array(price_series)
        index = pd.RangeIndex(len(values))

    # Ensure values is a 1D array
    values = np.asarray(values).flatten()

    if len(values) == 0:
        raise ValueError("Cannot run EKF on empty price series")

    n = len(values)
    x = np.zeros((n, 3))      # [level, velocity, log_var]
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
# 2. Regime Detection Functions
# -----------------------------
def detect_trend_regime(prices, window=20):
    """
    Detect trending vs mean-reverting regime

    Returns:
        - 1: Trending (strong directional move)
        - 0: Mean-reverting (range-bound)
    """
    # ADX-like calculation (simplified)
    returns = prices.pct_change()

    # Directional movement
    up_move = returns.clip(lower=0)
    down_move = -returns.clip(upper=0)

    # Smoothed directional indicators
    up_smooth = up_move.rolling(window).mean()
    down_smooth = down_move.rolling(window).mean()

    # Trend strength
    total_move = up_smooth + down_smooth
    dx = np.abs(up_smooth - down_smooth) / (total_move + 1e-10)

    # ADX approximation
    adx = dx.rolling(window).mean()

    # Regime: trending if ADX > 0.4, mean-reverting otherwise
    regime = (adx > 0.4).astype(int)

    return regime, adx


def detect_volatility_regime(prices, window=20, threshold=1.0):
    """
    Detect high vs low volatility regime

    Returns:
        - 1: High volatility
        - 0: Low volatility
    """
    returns = prices.pct_change()

    # Rolling volatility (annualized)
    vol = returns.rolling(window).std() * np.sqrt(365)

    # Historical median volatility
    median_vol = vol.rolling(window*5).median()

    # Regime: high vol if current > threshold * median
    regime = (vol > threshold * median_vol).astype(int)

    return regime, vol


def detect_mean_reversion_strength(prices, ekf_level, ekf_velocity, window=20):
    """
    Measure mean reversion strength using EKF

    Returns:
        - High values: Strong mean reversion
        - Low values: Weak/no mean reversion
    """
    # Deviation from EKF level
    deviation = (prices - ekf_level) / ekf_level

    # Mean reversion indicator: price crossing back to level
    crosses = (deviation * deviation.shift(1) < 0).astype(int)

    # Mean reversion rate (crossings per period)
    mr_rate = crosses.rolling(window).sum() / window

    # Velocity magnitude (high velocity = stretched, likely to revert)
    vel_mag = np.abs(ekf_velocity)
    vel_normalized = (vel_mag - vel_mag.rolling(window*5).mean()) / (vel_mag.rolling(window*5).std() + 1e-10)

    # Combined mean reversion strength
    mr_strength = mr_rate * (1 + vel_normalized.clip(lower=0))

    return mr_strength, deviation


def detect_regime_stability(trend_regime, vol_regime, window=20):
    """
    Detect how stable/persistent current regime is

    Returns:
        - High values: Stable regime (good for trading)
        - Low values: Unstable/transitioning (risky)
    """
    # Count regime changes
    trend_changes = (trend_regime != trend_regime.shift(1)).astype(int)
    vol_changes = (vol_regime != vol_regime.shift(1)).astype(int)

    # Stability = low change rate
    trend_stability = 1 - (trend_changes.rolling(window).sum() / window)
    vol_stability = 1 - (vol_changes.rolling(window).sum() / window)

    # Combined stability
    stability = (trend_stability + vol_stability) / 2

    return stability, trend_stability, vol_stability


# -----------------------------
# 3. Pattern Analysis
# -----------------------------
def analyze_patterns(df):
    """
    Analyze patterns and statistics in regime data
    """
    print("\n" + "="*80)
    print("REGIME PATTERN ANALYSIS")
    print("="*80)

    # Overall regime distribution
    print("\n📊 Regime Distribution:")
    trend_pct = df['trend_regime'].mean() * 100
    vol_pct = df['vol_regime'].mean() * 100
    print(f"  Trending:        {trend_pct:.1f}% of time")
    print(f"  Mean-reverting:  {100-trend_pct:.1f}% of time")
    print(f"  High volatility: {vol_pct:.1f}% of time")
    print(f"  Low volatility:  {100-vol_pct:.1f}% of time")

    # Regime combinations
    print("\n🔄 Regime Combinations:")
    combinations = df.groupby(['trend_regime', 'vol_regime']).size()
    total = len(df)
    for (trend, vol), count in combinations.items():
        trend_label = "Trending" if trend == 1 else "Mean-Rev"
        vol_label = "High-Vol" if vol == 1 else "Low-Vol"
        pct = (count / total) * 100
        print(f"  {trend_label} + {vol_label}: {pct:.1f}%")

    # Mean reversion strength by regime
    print("\n💪 Mean Reversion Strength by Regime:")
    for trend in [0, 1]:
        trend_label = "Trending" if trend == 1 else "Mean-Reverting"
        regime_data = df[df['trend_regime'] == trend]
        if len(regime_data) > 0:
            avg_mr = regime_data['mr_strength'].mean()
            print(f"  {trend_label}: {avg_mr:.4f}")

    # Volatility statistics
    print("\n📈 Volatility Statistics:")
    print(f"  Current volatility:  {df['volatility'].iloc[-1]:.2%}")
    print(f"  Average volatility:  {df['volatility'].mean():.2%}")
    print(f"  Median volatility:   {df['volatility'].median():.2%}")
    print(f"  Max volatility:      {df['volatility'].max():.2%}")
    print(f"  Min volatility:      {df['volatility'].min():.2%}")

    # Regime stability
    print("\n🎯 Regime Stability:")
    current_stability = df['stability'].iloc[-1]
    avg_stability = df['stability'].mean()
    print(f"  Current stability: {current_stability:.2%}")
    print(f"  Average stability: {avg_stability:.2%}")

    # Recent regime (last 30 days)
    print("\n📅 Recent Regime (Last 30 days):")
    recent = df.tail(30)
    if len(recent) > 0:
        recent_trend = recent['trend_regime'].mean()
        recent_vol = recent['vol_regime'].mean()
        recent_mr = recent['mr_strength'].mean()
        print(f"  Trending tendency:      {recent_trend:.2%}")
        print(f"  High-vol tendency:      {recent_vol:.2%}")
        print(f"  Mean reversion strength: {recent_mr:.4f}")

    # Current state
    print("\n🎲 Current State (Latest):")
    latest = df.iloc[-1]
    trend_label = "TRENDING" if latest['trend_regime'] == 1 else "MEAN-REVERTING"
    vol_label = "HIGH-VOL" if latest['vol_regime'] == 1 else "LOW-VOL"
    print(f"  Regime:           {trend_label} + {vol_label}")
    print(f"  Trend strength:   {latest['trend_strength']:.4f}")
    print(f"  Volatility:       {latest['volatility']:.2%}")
    print(f"  MR strength:      {latest['mr_strength']:.4f}")
    print(f"  Stability:        {latest['stability']:.2%}")
    print(f"  Price deviation:  {latest['deviation']:.2%}")

    # Trading recommendations
    print("\n💡 Trading Implications:")
    if latest['trend_regime'] == 0 and latest['stability'] > 0.7:
        print("  ✅ FAVORABLE: Mean-reverting regime with high stability")
        print("  → Strategy: Active mean-reversion trades likely profitable")
    elif latest['trend_regime'] == 1 and latest['stability'] > 0.7:
        print("  ⚠️  CAUTION: Trending regime detected")
        print("  → Strategy: Reduce position sizes, wait for mean-reversion signals")
    elif latest['stability'] < 0.5:
        print("  🛑 WARNING: Unstable regime (transitioning)")
        print("  → Strategy: Avoid trading until regime stabilizes")
    else:
        print("  ℹ️  NEUTRAL: Mixed conditions")
        print("  → Strategy: Use normal risk management")

    print("\n" + "="*80)


# -----------------------------
# 4. Main Execution
# -----------------------------
def main():
    print("="*80)
    print("Q-PRIME REGIME & PATTERN DETECTOR")
    print("="*80)

    # Download data
    print("\n📥 Downloading BTC data...")
    timeframe = "1d"  # Change to "4h" or "1h" if needed

    try:
        btc = yf.download("BTC-USD", start="2022-01-01", interval=timeframe, progress=False)
        if btc.empty:
            raise ValueError("No data downloaded")
        print(f"✅ Downloaded {len(btc)} periods from {btc.index[0]} to {btc.index[-1]}")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return

    # Prepare data - handle both single and multi-level column indices
    if isinstance(btc.columns, pd.MultiIndex):
        close = btc['Close']['BTC-USD'].copy()
    else:
        close = btc['Close'].copy()

    # Ensure it's a proper Series with numeric values
    close = pd.Series(close.values.flatten(), index=btc.index)

    # Run EKF
    print("\n🔧 Running Extended Kalman Filter...")
    ekf_level, ekf_velocity = run_ekf(close)

    # Detect regimes
    print("🔍 Detecting market regimes...")
    trend_regime, trend_strength = detect_trend_regime(close)
    vol_regime, volatility = detect_volatility_regime(close)
    mr_strength, deviation = detect_mean_reversion_strength(close, ekf_level, ekf_velocity)
    stability, trend_stability, vol_stability = detect_regime_stability(trend_regime, vol_regime)

    # Create results dataframe
    df = pd.DataFrame({
        'price': close,
        'ekf_level': ekf_level,
        'ekf_velocity': ekf_velocity,
        'trend_regime': trend_regime,
        'trend_strength': trend_strength,
        'vol_regime': vol_regime,
        'volatility': volatility,
        'mr_strength': mr_strength,
        'deviation': deviation,
        'stability': stability,
        'trend_stability': trend_stability,
        'vol_stability': vol_stability
    })

    # Drop NaN rows
    df = df.dropna()

    # Analyze patterns
    analyze_patterns(df)

    # Save results
    output_file = 'regime_analysis.csv'
    df.to_csv(output_file)
    print(f"\n💾 Results saved to: {output_file}")

    # Recent history (last 60 days)
    print(f"\n📋 Recent Regime History (Last 60 periods):")
    print("-" * 80)
    recent = df.tail(60)
    for idx, row in recent.iterrows():
        trend_symbol = "📈" if row['trend_regime'] == 1 else "↔️"
        vol_symbol = "🔴" if row['vol_regime'] == 1 else "🟢"
        stability_bar = "█" * int(row['stability'] * 10)
        print(f"{idx.strftime('%Y-%m-%d')} | {trend_symbol} {vol_symbol} | "
              f"Stab: {stability_bar:10s} {row['stability']:.2f} | "
              f"MR: {row['mr_strength']:.3f} | "
              f"Dev: {row['deviation']:+.2%}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
