"""
Regime Classification System

Two-stage classification:
1. CRISIS DETECTOR ("Oh Shit" Gate) - Circuit breaker for dangerous markets
2. REGIME CLASSIFIER - Classifies safe markets into BULL/BEAR/SIDEWAYS

Crisis conditions (any triggers sit-out):
- Volatility explosion (ATR > 3x normal)
- Severe drawdown (> 20% in 30 days)
- Crash velocity (> 10% drop in 5 days)
- Volatility regime instability

Regime classification:
- BULL: Trending up, EMA20 > EMA50, positive momentum
- BEAR: Trending down, EMA20 < EMA50, negative momentum
- SIDEWAYS: No clear trend, mean-reverting, low ADX
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional


class Regime(Enum):
    CRISIS = "CRISIS"       # Sit out completely
    BULL = "BULL"           # Trending up
    BEAR = "BEAR"           # Trending down
    SIDEWAYS = "SIDEWAYS"   # Mean-reverting / choppy


@dataclass
class RegimeState:
    regime: Regime
    confidence: float       # 0-1, how confident in classification
    crisis_flags: dict      # Which crisis conditions are triggered
    metrics: dict           # Underlying metrics for debugging


class CrisisDetector:
    """
    "Oh Shit" Gate - Circuit breaker for dangerous market conditions

    If ANY condition triggers, we sit out completely.
    """

    def __init__(
        self,
        vol_explosion_threshold: float = 3.0,    # ATR > 3x median
        severe_dd_threshold: float = -0.20,      # -20% drawdown
        crash_velocity_threshold: float = -0.10, # -10% in 5 days
        vol_instability_threshold: float = 0.6,  # Volatility regime changes
        lookback_dd: int = 30,                   # Drawdown lookback
        lookback_vol: int = 90,                  # Volatility baseline lookback
    ):
        self.vol_explosion_threshold = vol_explosion_threshold
        self.severe_dd_threshold = severe_dd_threshold
        self.crash_velocity_threshold = crash_velocity_threshold
        self.vol_instability_threshold = vol_instability_threshold
        self.lookback_dd = lookback_dd
        self.lookback_vol = lookback_vol

    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range - measures volatility"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def detect(self, df: pd.DataFrame) -> Tuple[bool, dict]:
        """
        Check if we're in crisis mode.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns (or just 'close')

        Returns:
            (is_crisis, flags_dict)
        """
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df.get('high', df.get('High', close * 1.01))
        low = df.get('low', df.get('Low', close * 0.99))

        flags = {}

        # 1. Volatility Explosion
        atr = self.compute_atr(high, low, close, 14)
        atr_median = atr.rolling(self.lookback_vol).median()
        if len(atr) > 0 and len(atr_median.dropna()) > 0:
            current_atr = atr.iloc[-1]
            median_atr = atr_median.iloc[-1]
            if pd.notna(current_atr) and pd.notna(median_atr) and median_atr > 0:
                vol_ratio = current_atr / median_atr
                flags['volatility_explosion'] = vol_ratio > self.vol_explosion_threshold
                flags['vol_ratio'] = vol_ratio
            else:
                flags['volatility_explosion'] = False
                flags['vol_ratio'] = 0
        else:
            flags['volatility_explosion'] = False
            flags['vol_ratio'] = 0

        # 2. Severe Drawdown
        rolling_max = close.rolling(self.lookback_dd).max()
        drawdown = (close / rolling_max - 1)
        current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        flags['severe_drawdown'] = current_dd < self.severe_dd_threshold
        flags['current_drawdown'] = current_dd

        # 3. Crash Velocity (fast drop)
        if len(close) >= 5:
            five_day_return = close.iloc[-1] / close.iloc[-5] - 1
            flags['crash_velocity'] = five_day_return < self.crash_velocity_threshold
            flags['five_day_return'] = five_day_return
        else:
            flags['crash_velocity'] = False
            flags['five_day_return'] = 0

        # 4. Volatility Instability (regime flip-flopping)
        returns = close.pct_change()
        vol = returns.rolling(20).std() * np.sqrt(365)
        vol_median = vol.rolling(100).median()
        high_vol = (vol > vol_median).astype(int)
        vol_changes = (high_vol != high_vol.shift(1)).astype(int)
        vol_instability = vol_changes.rolling(20).sum() / 20
        current_instability = vol_instability.iloc[-1] if len(vol_instability) > 0 else 0
        flags['volatility_unstable'] = current_instability > self.vol_instability_threshold
        flags['vol_instability'] = current_instability

        # Crisis = ANY flag triggered
        is_crisis = any([
            flags.get('volatility_explosion', False),
            flags.get('severe_drawdown', False),
            flags.get('crash_velocity', False),
            flags.get('volatility_unstable', False),
        ])

        return is_crisis, flags


class RegimeClassifier:
    """
    Classifies market into BULL / BEAR / SIDEWAYS

    Only called when NOT in crisis mode.
    """

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        adx_period: int = 14,
        adx_trend_threshold: float = 15,    # ADX > 15 = trending (tuned for BTC)
        momentum_period: int = 20,
        min_trend_strength: float = 0.02,   # 2% difference for trend
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.momentum_period = momentum_period
        self.min_trend_strength = min_trend_strength

    def compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index - measures trend strength"""
        # +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()

        # Keep as pandas Series with proper index
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0),
            index=high.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0),
            index=high.index
        )

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed values
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / atr
        minus_di = 100 * minus_dm.rolling(period).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx, plus_di, minus_di

    def classify(self, df: pd.DataFrame) -> Tuple[Regime, float, dict]:
        """
        Classify current market regime.

        Returns:
            (regime, confidence, metrics)
        """
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df.get('high', df.get('High', close * 1.01))
        low = df.get('low', df.get('Low', close * 0.99))

        metrics = {}

        # 1. EMA crossover (trend direction)
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()
        ema_ratio = (ema_fast / ema_slow - 1).iloc[-1]
        metrics['ema_ratio'] = ema_ratio

        # 2. ADX (trend strength)
        adx, plus_di, minus_di = self.compute_adx(high, low, close, self.adx_period)
        current_adx = adx.iloc[-1] if len(adx) > 0 else 0
        metrics['adx'] = current_adx
        metrics['plus_di'] = plus_di.iloc[-1] if len(plus_di) > 0 else 0
        metrics['minus_di'] = minus_di.iloc[-1] if len(minus_di) > 0 else 0

        # 3. Momentum
        momentum = (close.iloc[-1] / close.iloc[-self.momentum_period] - 1) if len(close) >= self.momentum_period else 0
        metrics['momentum'] = momentum

        # 4. Price vs EMAs
        price_vs_ema_fast = (close.iloc[-1] / ema_fast.iloc[-1] - 1)
        price_vs_ema_slow = (close.iloc[-1] / ema_slow.iloc[-1] - 1)
        metrics['price_vs_ema_fast'] = price_vs_ema_fast
        metrics['price_vs_ema_slow'] = price_vs_ema_slow

        # Classification logic
        is_trending = current_adx > self.adx_trend_threshold

        if not is_trending:
            # Low ADX = sideways/choppy market
            regime = Regime.SIDEWAYS
            confidence = min(1.0, (self.adx_trend_threshold - current_adx) / self.adx_trend_threshold)
        elif ema_ratio > self.min_trend_strength and momentum > 0:
            # Trending up
            regime = Regime.BULL
            confidence = min(1.0, ema_ratio / 0.10)  # Normalize to ~10% as max
        elif ema_ratio < -self.min_trend_strength and momentum < 0:
            # Trending down
            regime = Regime.BEAR
            confidence = min(1.0, abs(ema_ratio) / 0.10)
        else:
            # Trending but mixed signals
            regime = Regime.SIDEWAYS
            confidence = 0.5

        return regime, confidence, metrics


class RegimeSystem:
    """
    Complete regime detection system combining crisis detection and classification.
    """

    def __init__(
        self,
        crisis_detector: Optional[CrisisDetector] = None,
        regime_classifier: Optional[RegimeClassifier] = None,
    ):
        self.crisis_detector = crisis_detector or CrisisDetector()
        self.regime_classifier = regime_classifier or RegimeClassifier()

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Full regime detection pipeline.

        Returns RegimeState with regime, confidence, and debug info.
        """
        # Stage 1: Crisis check
        is_crisis, crisis_flags = self.crisis_detector.detect(df)

        if is_crisis:
            return RegimeState(
                regime=Regime.CRISIS,
                confidence=1.0,
                crisis_flags=crisis_flags,
                metrics={'triggered_by': [k for k, v in crisis_flags.items() if v is True]}
            )

        # Stage 2: Regime classification
        regime, confidence, metrics = self.regime_classifier.classify(df)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            crisis_flags=crisis_flags,
            metrics=metrics
        )

    def classify_history(self, df: pd.DataFrame, min_lookback: int = 100) -> pd.DataFrame:
        """
        Classify regime for each day in historical data.

        Returns DataFrame with regime labels for each day.
        """
        results = []

        for i in range(min_lookback, len(df)):
            window = df.iloc[:i+1].copy()
            state = self.detect(window)

            results.append({
                'date': df.index[i] if hasattr(df.index[i], 'date') else df.index[i],
                'regime': state.regime.value,
                'confidence': state.confidence,
                **{f'crisis_{k}': v for k, v in state.crisis_flags.items()},
                **{f'metric_{k}': v for k, v in state.metrics.items() if not isinstance(v, list)},
            })

        return pd.DataFrame(results)


# =============================================================================
# Testing and Visualization
# =============================================================================

def test_regime_system():
    """Test the regime system on historical BTC data"""
    from pathlib import Path

    print("="*70)
    print("REGIME CLASSIFICATION SYSTEM TEST")
    print("="*70)

    # Load data
    csv_path = Path("regime_analysis.csv")
    if not csv_path.exists():
        print("Error: regime_analysis.csv not found")
        return

    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.rename(columns={'Date': 'date', 'price': 'Close'})
    df = df.set_index('date').sort_index()
    df['close'] = df['Close']
    df['high'] = df['Close'] * 1.02  # Approximate
    df['low'] = df['Close'] * 0.98   # Approximate

    print(f"Data: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} ({len(df)} days)")

    # Initialize system
    system = RegimeSystem()

    # Classify history
    print("\nClassifying historical regimes...")
    history = system.classify_history(df, min_lookback=100)

    # Summary
    print("\n" + "-"*70)
    print("REGIME DISTRIBUTION")
    print("-"*70)

    regime_counts = history['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(history) * 100
        print(f"  {regime:<12} {count:>6} days ({pct:>5.1f}%)")

    # Show regime transitions
    print("\n" + "-"*70)
    print("REGIME PERIODS")
    print("-"*70)

    # Find contiguous regime periods
    history['regime_change'] = history['regime'] != history['regime'].shift(1)
    history['period_id'] = history['regime_change'].cumsum()

    periods = history.groupby('period_id').agg({
        'date': ['first', 'last'],
        'regime': 'first',
    })
    periods.columns = ['start', 'end', 'regime']
    periods['days'] = (pd.to_datetime(periods['end']) - pd.to_datetime(periods['start'])).dt.days + 1

    print(f"\n{'Start':<12} {'End':<12} {'Regime':<12} {'Days':>6}")
    print("-"*46)
    for _, row in periods.tail(20).iterrows():
        print(f"{str(row['start'])[:10]:<12} {str(row['end'])[:10]:<12} {row['regime']:<12} {row['days']:>6}")

    # Crisis periods
    crisis_periods = periods[periods['regime'] == 'CRISIS']
    print(f"\n" + "-"*70)
    print(f"CRISIS PERIODS DETECTED: {len(crisis_periods)}")
    print("-"*70)

    if len(crisis_periods) > 0:
        for _, row in crisis_periods.iterrows():
            print(f"  {str(row['start'])[:10]} to {str(row['end'])[:10]} ({row['days']} days)")
    else:
        print("  No crisis periods detected in data")

    # Show current state
    print("\n" + "-"*70)
    print("CURRENT STATE")
    print("-"*70)

    current = system.detect(df)
    print(f"  Regime:     {current.regime.value}")
    print(f"  Confidence: {current.confidence:.2%}")
    print(f"  Crisis Flags:")
    for k, v in current.crisis_flags.items():
        if isinstance(v, bool):
            print(f"    {k}: {'TRIGGERED' if v else 'OK'}")
        else:
            print(f"    {k}: {v:.4f}")

    print("\n" + "="*70)

    return history


if __name__ == "__main__":
    history = test_regime_system()
