"""
Trend-Following Strategy for Bull Markets

Simple trend-following approach for confirmed bull markets:
- Buy and hold during uptrends
- Trail stops at bull market support bands
- Exit when support breaks

This complements the mean-reversion strategy (bear markets).
"""

import numpy as np
import pandas as pd


class TrendFollowingStrategy:
    """
    Trend-following strategy for bull markets.

    Strategy:
    - Enter long when in confirmed bull (price > support bands)
    - Hold position with trailing stop at support bands
    - Exit when price breaks below BOTH support bands
    - Use volatility targeting for position sizing
    """

    def __init__(self,
                 vol_target=0.20,
                 max_exposure=0.50,
                 trail_buffer=0.02):  # 2% buffer below support bands
        """
        Initialize trend-following strategy.

        Args:
            vol_target: Target annual volatility (default 0.20 = 20%)
            max_exposure: Maximum position size as fraction of equity
            trail_buffer: Buffer below support bands for stop (default 0.02 = 2%)
        """
        self.vol_target = vol_target
        self.max_exposure = max_exposure
        self.trail_buffer = trail_buffer

    def calculate_support_bands(self, prices):
        """
        Calculate bull market support bands.

        Args:
            prices: pandas Series of prices

        Returns:
            sma_20w: 20-week (140-day) simple moving average
            ema_21w: 21-week (147-day) exponential moving average
        """
        # 20 weeks ≈ 140 days, 21 weeks ≈ 147 days
        sma_20w = prices.rolling(window=140).mean()
        ema_21w = prices.ewm(span=147, adjust=False).mean()

        return sma_20w, ema_21w

    def is_bull_confirmed(self, price, sma, ema):
        """
        Check if bull market is confirmed.

        Bull confirmed when price > BOTH support bands.

        Args:
            price: Current price
            sma: 20-week SMA value
            ema: 21-week EMA value

        Returns:
            bool: True if bull confirmed
        """
        if pd.isna(sma) or pd.isna(ema):
            return False

        return price > sma and price > ema

    def calculate_stop_loss(self, sma, ema):
        """
        Calculate trailing stop loss based on support bands.

        Stop is set slightly below the higher of the two support bands.

        Args:
            sma: 20-week SMA value
            ema: 21-week EMA value

        Returns:
            float: Stop loss price
        """
        if pd.isna(sma) or pd.isna(ema):
            return None

        # Use the higher support band
        support = max(sma, ema)

        # Set stop slightly below with buffer
        stop = support * (1 - self.trail_buffer)

        return stop

    def calculate_position_size(self, equity, price, volatility):
        """
        Calculate position size using volatility targeting.

        Args:
            equity: Current portfolio equity
            price: Current asset price
            volatility: Current annualized volatility

        Returns:
            float: Position size (number of units to hold)
        """
        # Volatility-based position sizing
        target_exposure = min(self.max_exposure, self.vol_target / max(volatility, 0.01))

        # Convert to position size
        position_value = equity * target_exposure
        position_size = position_value / price

        return position_size

    def generate_signal(self, prices, current_price, current_position, equity, volatility):
        """
        Generate trading signal for trend-following strategy.

        Args:
            prices: pandas Series of historical prices
            current_price: Current price
            current_position: Current position size (>0 = long, 0 = no position)
            equity: Current portfolio equity
            volatility: Current annualized volatility

        Returns:
            dict with keys:
                - action: 'BUY', 'HOLD', 'SELL'
                - target_position: Target position size
                - reason: Explanation of signal
                - stop_loss: Current stop loss level (if applicable)
        """
        # Calculate support bands
        sma, ema = self.calculate_support_bands(prices)
        current_sma = sma.iloc[-1]
        current_ema = ema.iloc[-1]

        # Check if bull market is confirmed
        bull_confirmed = self.is_bull_confirmed(current_price, current_sma, current_ema)

        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(current_sma, current_ema)

        # Generate signal
        if current_position == 0:
            # No position - check if we should enter
            if bull_confirmed:
                # Enter long position
                target_position = self.calculate_position_size(equity, current_price, volatility)
                return {
                    'action': 'BUY',
                    'target_position': target_position,
                    'reason': 'Bull confirmed (price > support bands)',
                    'stop_loss': stop_loss
                }
            else:
                # Stay out
                return {
                    'action': 'HOLD',
                    'target_position': 0,
                    'reason': 'Bull not confirmed',
                    'stop_loss': None
                }
        else:
            # Have position - check if we should exit or hold
            if stop_loss and current_price < stop_loss:
                # Stop hit - exit
                return {
                    'action': 'SELL',
                    'target_position': 0,
                    'reason': f'Stop loss hit (price {current_price:.2f} < stop {stop_loss:.2f})',
                    'stop_loss': stop_loss
                }
            elif bull_confirmed:
                # Bull still confirmed - hold and trail stop
                target_position = self.calculate_position_size(equity, current_price, volatility)
                return {
                    'action': 'HOLD',
                    'target_position': target_position,
                    'reason': 'Holding long (bull trend continues)',
                    'stop_loss': stop_loss
                }
            else:
                # Support broken - exit
                return {
                    'action': 'SELL',
                    'target_position': 0,
                    'reason': 'Support bands broken',
                    'stop_loss': stop_loss
                }


if __name__ == "__main__":
    # Test the trend strategy
    print("Testing Trend-Following Strategy...")

    # Create sample data: uptrend
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(300) * 2 + 0.3),  # Uptrend
        index=dates
    )

    strategy = TrendFollowingStrategy()

    # Test at different points
    print("\n" + "="*60)
    print("Testing on simulated uptrend:")
    print("="*60)

    # Early in trend
    signal = strategy.generate_signal(
        prices.iloc[:200],
        prices.iloc[199],
        current_position=0,
        equity=10000,
        volatility=0.30
    )
    print(f"\nDay 200 (no position):")
    print(f"  Action: {signal['action']}")
    print(f"  Target Position: {signal['target_position']:.2f}")
    print(f"  Reason: {signal['reason']}")
    print(f"  Stop Loss: {signal['stop_loss']:.2f if signal['stop_loss'] else 'N/A'}")

    # Later in trend (with position)
    signal = strategy.generate_signal(
        prices.iloc[:250],
        prices.iloc[249],
        current_position=10,
        equity=10000,
        volatility=0.30
    )
    print(f"\nDay 250 (with position):")
    print(f"  Action: {signal['action']}")
    print(f"  Reason: {signal['reason']}")
    print(f"  Stop Loss: {signal['stop_loss']:.2f if signal['stop_loss'] else 'N/A'}")

    print("\n" + "="*60)
    print("✓ Trend strategy test complete")
    print("="*60)
