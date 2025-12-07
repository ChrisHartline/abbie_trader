"""
Market Regime Router - Multi-Indicator Bull/Bear/Transition Classifier

Uses multiple technical indicators to robustly classify market regime:
1. Bull Market Support Band (20-week SMA + 21-week EMA)
2. 200-day Moving Average (long-term trend)
3. 50-day Moving Average (medium-term trend)
4. Golden/Death Cross (50/200 MA crossover)
5. Trend Strength (consecutive higher highs/lows)

Returns: "BULL", "BEAR", or "TRANSITION" with confidence score
"""

import numpy as np
import pandas as pd


class MarketRegimeRouter:
    """
    Router that classifies market regime using multiple indicators.

    Usage:
        router = MarketRegimeRouter()
        regime, confidence = router.classify(prices)
    """

    def __init__(self,
                 ma_short=50,
                 ma_long=200,
                 bull_band_sma=140,  # 20 weeks * 7 days ≈ 140 days
                 bull_band_ema=147): # 21 weeks * 7 days ≈ 147 days
        """
        Initialize router with configurable parameters.

        Args:
            ma_short: Short-term moving average period (default 50)
            ma_long: Long-term moving average period (default 200)
            bull_band_sma: Bull market support band SMA (default 140 days ≈ 20 weeks)
            bull_band_ema: Bull market support band EMA (default 147 days ≈ 21 weeks)
        """
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.bull_band_sma = bull_band_sma
        self.bull_band_ema = bull_band_ema

    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()

    def detect_trend_strength(self, prices, lookback=20):
        """
        Detect trend strength using consecutive higher highs/lows.

        Returns:
            score: +1 (strong uptrend) to -1 (strong downtrend)
        """
        if len(prices) < lookback:
            return 0.0

        recent = prices.iloc[-lookback:]

        # Count higher highs and higher lows
        higher_highs = 0
        lower_lows = 0

        for i in range(1, len(recent)):
            if recent.iloc[i] > recent.iloc[i-1]:
                higher_highs += 1
            elif recent.iloc[i] < recent.iloc[i-1]:
                lower_lows += 1

        # Normalize to -1 to +1
        net_trend = (higher_highs - lower_lows) / lookback
        return net_trend

    def classify(self, prices):
        """
        Classify current market regime.

        Args:
            prices: pandas Series of prices (indexed by date)

        Returns:
            regime: "BULL", "BEAR", or "TRANSITION"
            confidence: 0.0 to 1.0 (how confident the classification is)
            signals: dict of individual signal values (for debugging)
        """
        if len(prices) < max(self.ma_long, self.bull_band_sma, self.bull_band_ema):
            return "TRANSITION", 0.5, {}

        current_price = prices.iloc[-1]

        # Calculate indicators
        ma_50 = self.calculate_sma(prices, self.ma_short).iloc[-1]
        ma_200 = self.calculate_sma(prices, self.ma_long).iloc[-1]
        bull_band_sma = self.calculate_sma(prices, self.bull_band_sma).iloc[-1]
        bull_band_ema = self.calculate_ema(prices, self.bull_band_ema).iloc[-1]
        trend_strength = self.detect_trend_strength(prices)

        # Individual signals (each contributes to bull/bear score)
        signals = {}
        bull_score = 0
        bear_score = 0

        # Signal 1: Price vs 200-day MA (strongest signal, weight = 2)
        if current_price > ma_200:
            bull_score += 2
            signals['ma_200'] = 'BULL'
        else:
            bear_score += 2
            signals['ma_200'] = 'BEAR'

        # Signal 2: Price vs 50-day MA (weight = 1.5)
        if current_price > ma_50:
            bull_score += 1.5
            signals['ma_50'] = 'BULL'
        else:
            bear_score += 1.5
            signals['ma_50'] = 'BEAR'

        # Signal 3: Bull Market Support Band (weight = 2)
        # Price above BOTH 20-week SMA and 21-week EMA = strong bull
        above_sma = current_price > bull_band_sma
        above_ema = current_price > bull_band_ema

        if above_sma and above_ema:
            bull_score += 2
            signals['bull_band'] = 'BULL'
        elif not above_sma and not above_ema:
            bear_score += 2
            signals['bull_band'] = 'BEAR'
        else:
            # Mixed signal = transition
            signals['bull_band'] = 'TRANSITION'

        # Signal 4: Golden/Death Cross (weight = 1.5)
        if ma_50 > ma_200:
            bull_score += 1.5
            signals['ma_cross'] = 'BULL'
        else:
            bear_score += 1.5
            signals['ma_cross'] = 'BEAR'

        # Signal 5: Trend Strength (weight = 1)
        if trend_strength > 0.3:
            bull_score += 1
            signals['trend'] = 'BULL'
        elif trend_strength < -0.3:
            bear_score += 1
            signals['trend'] = 'BEAR'
        else:
            signals['trend'] = 'NEUTRAL'

        # Total possible score: 2 + 1.5 + 2 + 1.5 + 1 = 8
        total_score = bull_score + bear_score

        # Classify based on scores
        if bull_score > bear_score * 1.5:  # Strong bull
            regime = "BULL"
            confidence = bull_score / total_score
        elif bear_score > bull_score * 1.5:  # Strong bear
            regime = "BEAR"
            confidence = bear_score / total_score
        else:  # Mixed signals
            regime = "TRANSITION"
            confidence = 1.0 - abs(bull_score - bear_score) / total_score

        signals['bull_score'] = bull_score
        signals['bear_score'] = bear_score
        signals['current_price'] = current_price
        signals['ma_50'] = ma_50
        signals['ma_200'] = ma_200

        return regime, confidence, signals


def classify_regime(prices):
    """
    Convenience function for quick regime classification.

    Args:
        prices: pandas Series of prices

    Returns:
        regime: "BULL", "BEAR", or "TRANSITION"
        confidence: 0.0 to 1.0
    """
    router = MarketRegimeRouter()
    regime, confidence, _ = router.classify(prices)
    return regime, confidence


if __name__ == "__main__":
    # Test the router
    print("Testing Market Regime Router...")

    # Create sample data: bull market followed by bear
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Simulate bull market (uptrend)
    bull_prices = pd.Series(
        100 + np.cumsum(np.random.randn(250) * 2 + 0.5),
        index=dates[:250]
    )

    # Simulate bear market (downtrend)
    bear_prices = pd.Series(
        bull_prices.iloc[-1] + np.cumsum(np.random.randn(250) * 2 - 0.5),
        index=dates[250:]
    )

    all_prices = pd.concat([bull_prices, bear_prices])

    # Test at different points
    router = MarketRegimeRouter()

    print("\n" + "="*60)
    print("Testing on simulated data:")
    print("="*60)

    # Test during bull phase
    regime, conf, signals = router.classify(all_prices.iloc[:300])
    print(f"\nDay 300 (bull phase):")
    print(f"  Regime: {regime}")
    print(f"  Confidence: {conf:.2%}")
    print(f"  Bull Score: {signals['bull_score']:.1f}")
    print(f"  Bear Score: {signals['bear_score']:.1f}")

    # Test during bear phase
    regime, conf, signals = router.classify(all_prices.iloc[:450])
    print(f"\nDay 450 (bear phase):")
    print(f"  Regime: {regime}")
    print(f"  Confidence: {conf:.2%}")
    print(f"  Bull Score: {signals['bull_score']:.1f}")
    print(f"  Bear Score: {signals['bear_score']:.1f}")

    print("\n" + "="*60)
    print("✓ Router test complete")
    print("="*60)
