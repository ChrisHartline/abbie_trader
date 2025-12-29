"""
Enhanced Feature Engineering for Regime-Specific Trading

Expands on the original 5 features with additional technical indicators:
- Bollinger Bands (%B, width)
- RSI (14)
- MACD (12, 26, 9)
- EMA ratios
- ATR (normalized)
- Volume indicators (if available)

These features feed into regime-specific FFNNs.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # EKF parameters
    ekf_dt: float = 1.0

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # RSI
    rsi_period: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # EMAs
    ema_fast: int = 20
    ema_slow: int = 50
    ema_very_slow: int = 200

    # ATR
    atr_period: int = 14

    # Momentum
    momentum_period: int = 5
    momentum_long: int = 20

    # Mean reversion
    mean_rev_period: int = 30


class FeatureEngineer:
    """
    Computes all features for the trading system.

    Output features (14 total):
    1. ekf_level - EKF smoothed price level
    2. ekf_velocity - EKF price velocity
    3. funding_rate - Funding rate (synthetic for backtest)
    4. momentum_5 - 5-day momentum
    5. rel_price - Price relative to 30-day mean
    6. bb_pct_b - Bollinger Band %B (0-1, where in band)
    7. bb_width - Bollinger Band width (volatility)
    8. rsi - RSI (0-100)
    9. macd_hist - MACD histogram
    10. ema_ratio_fast - EMA20/EMA50 ratio
    11. ema_ratio_slow - EMA50/EMA200 ratio
    12. atr_pct - ATR as % of price
    13. momentum_20 - 20-day momentum
    14. mean_rev_strength - Mean reversion strength (deviation from mean)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def run_ekf(self, price_series: pd.Series, dt: float = 1.0) -> Tuple[pd.Series, pd.Series]:
        """Extended Kalman Filter for state estimation"""
        values = price_series.values.astype(float)
        index = price_series.index
        n = len(values)

        if n == 0:
            raise ValueError("Cannot run EKF on empty price series")

        x = np.zeros((n, 3))  # [level, velocity, log_var]
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
            S = P_pred[0, 0] + R
            K = P_pred[:, 0] / S

            x[t] = x_pred + K * y
            P[t] = (np.eye(3) - np.outer(K, np.array([1, 0, 0]))) @ P_pred
            smoothed[t] = x[t][0]

        level = pd.Series(smoothed, index=index)
        velocity = pd.Series(x[:, 1], index=index)
        return level, velocity

    def compute_bollinger_bands(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: middle, upper, lower, %B, width"""
        cfg = self.config

        middle = close.rolling(cfg.bb_period).mean()
        std = close.rolling(cfg.bb_period).std()
        upper = middle + cfg.bb_std * std
        lower = middle - cfg.bb_std * std

        # %B: where price is within the bands (0 = lower, 1 = upper)
        pct_b = (close - lower) / (upper - lower + 1e-10)

        # Width: band width as % of middle (volatility indicator)
        width = (upper - lower) / (middle + 1e-10)

        return middle, upper, lower, pct_b, width

    def compute_rsi(self, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        """Relative Strength Index"""
        period = period or self.config.rsi_period

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def compute_macd(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD: line, signal, histogram"""
        cfg = self.config

        ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=cfg.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Average True Range"""
        period = self.config.atr_period

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr

    def compute_ema_ratios(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """EMA ratios for trend detection"""
        cfg = self.config

        ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()
        ema_very_slow = close.ewm(span=cfg.ema_very_slow, adjust=False).mean()

        # Ratio - 1 (so 0 = equal, positive = bullish, negative = bearish)
        ratio_fast = ema_fast / ema_slow - 1
        ratio_slow = ema_slow / ema_very_slow - 1

        return ratio_fast, ratio_slow

    def get_funding_rate(self, n: int, seed: int = 42) -> np.ndarray:
        """Synthetic funding rate for backtesting"""
        rng = np.random.default_rng(seed)
        base = np.cumsum(rng.normal(0, 0.0005, n))
        seasonal = 0.0002 * np.sin(np.arange(n) * 2 * np.pi / 180)
        noise = 0.0001 * rng.standard_normal(n)
        return np.tanh(base + seasonal) + noise

    def compute_features(self, df: pd.DataFrame, include_funding: bool = True) -> pd.DataFrame:
        """
        Compute all features from OHLC data.

        Args:
            df: DataFrame with 'close' (and optionally 'high', 'low', 'open', 'volume')
            include_funding: Whether to include synthetic funding rate

        Returns:
            DataFrame with all computed features
        """
        # Get price columns
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df.get('high', df.get('High', close * 1.02))
        low = df.get('low', df.get('Low', close * 0.98))

        features = pd.DataFrame(index=df.index)

        # 1-2. EKF features
        ekf_level, ekf_velocity = self.run_ekf(close, self.config.ekf_dt)
        features['ekf_level'] = ekf_level
        features['ekf_velocity'] = ekf_velocity

        # 3. Funding rate (synthetic)
        if include_funding:
            features['funding_rate'] = self.get_funding_rate(len(df))

        # 4. Short-term momentum
        features['momentum_5'] = close.pct_change(self.config.momentum_period)

        # 5. Relative price (mean reversion)
        features['rel_price'] = close / close.rolling(self.config.mean_rev_period).mean() - 1

        # 6-7. Bollinger Bands
        _, _, _, bb_pct_b, bb_width = self.compute_bollinger_bands(close)
        features['bb_pct_b'] = bb_pct_b
        features['bb_width'] = bb_width

        # 8. RSI
        features['rsi'] = self.compute_rsi(close) / 100  # Normalize to 0-1

        # 9. MACD histogram (normalized by price)
        _, _, macd_hist = self.compute_macd(close)
        features['macd_hist'] = macd_hist / close  # Normalize

        # 10-11. EMA ratios
        ema_ratio_fast, ema_ratio_slow = self.compute_ema_ratios(close)
        features['ema_ratio_fast'] = ema_ratio_fast
        features['ema_ratio_slow'] = ema_ratio_slow

        # 12. ATR as % of price
        atr = self.compute_atr(high, low, close)
        features['atr_pct'] = atr / close

        # 13. Long-term momentum
        features['momentum_20'] = close.pct_change(self.config.momentum_long)

        # 14. Mean reversion strength (z-score)
        rolling_mean = close.rolling(self.config.mean_rev_period).mean()
        rolling_std = close.rolling(self.config.mean_rev_period).std()
        features['mean_rev_zscore'] = (close - rolling_mean) / (rolling_std + 1e-10)

        # Compute returns for target
        features['return'] = np.log(close / close.shift(1))
        features['close'] = close

        return features

    def get_feature_names(self) -> list:
        """Get list of feature names (excluding target and close)"""
        return [
            'ekf_level', 'ekf_velocity', 'funding_rate',
            'momentum_5', 'rel_price',
            'bb_pct_b', 'bb_width', 'rsi', 'macd_hist',
            'ema_ratio_fast', 'ema_ratio_slow', 'atr_pct',
            'momentum_20', 'mean_rev_zscore'
        ]


def test_feature_engineering():
    """Test feature engineering on historical data"""
    from pathlib import Path

    print("="*70)
    print("FEATURE ENGINEERING TEST")
    print("="*70)

    # Load data
    csv_path = Path("regime_analysis.csv")
    if not csv_path.exists():
        print("Error: regime_analysis.csv not found")
        return

    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.rename(columns={'Date': 'date', 'price': 'close'})
    df = df.set_index('date').sort_index()
    df['high'] = df['close'] * 1.02
    df['low'] = df['close'] * 0.98

    print(f"Data: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} ({len(df)} days)")

    # Compute features
    engineer = FeatureEngineer()
    features = engineer.compute_features(df)

    # Drop NaN
    features_clean = features.dropna()
    print(f"Features computed: {len(features_clean)} samples (after dropping NaN)")

    # Summary statistics
    print("\n" + "-"*70)
    print("FEATURE STATISTICS")
    print("-"*70)

    feature_names = engineer.get_feature_names()
    print(f"\n{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-"*70)

    for feat in feature_names:
        if feat in features_clean.columns:
            col = features_clean[feat]
            print(f"{feat:<20} {col.mean():>12.4f} {col.std():>12.4f} {col.min():>12.4f} {col.max():>12.4f}")

    # Correlation with returns
    print("\n" + "-"*70)
    print("FEATURE CORRELATION WITH NEXT-DAY RETURNS")
    print("-"*70)

    features_clean['next_return'] = features_clean['return'].shift(-1)
    correlations = []

    for feat in feature_names:
        if feat in features_clean.columns:
            corr = features_clean[feat].corr(features_clean['next_return'])
            correlations.append((feat, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<20} {'Correlation':>12}")
    print("-"*35)
    for feat, corr in correlations:
        print(f"{feat:<20} {corr:>12.4f}")

    print("\n" + "="*70)

    return features_clean


if __name__ == "__main__":
    features = test_feature_engineering()
