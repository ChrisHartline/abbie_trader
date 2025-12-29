"""
Q-Prime Trend-Following System
==============================

TREND-FOLLOWING for momentum stocks (TSLA, growth stocks, ETFs)

Strategy:
- LONG when EMA20 > EMA50 (bull market)
- EXIT when trend breaks or crisis triggers
- NO SHORTING (long-only for momentum assets)

Gate Order:
1. CRISIS DETECTOR - "Oh Shit" gate for dangerous markets
2. TREND SIGNAL - EMA crossover for direction
3. RISK GATES - Position limits

Usage:
    python main_trend.py              # Uses config_tsla.py
    python main_trend.py --config config_tsll.py  # For 2x ETF

Broker: Webull (stocks) - requires webull package
"""

import argparse
import numpy as np
import pandas as pd
import time
import requests
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Parse command line arguments
# -----------------------------
parser = argparse.ArgumentParser(description='Q-Prime Trend-Following System')
parser.add_argument('--config', type=str, default='config_tsla',
                    help='Config module to use (default: config_tsla)')
parser.add_argument('--dry-run', action='store_true',
                    help='Run without executing trades')
args = parser.parse_args()

# Import config dynamically
import importlib
config = importlib.import_module(args.config.replace('.py', ''))

# Extract config values with defaults
SYMBOL = getattr(config, 'SYMBOL', 'TSLA')
INITIAL_USD = getattr(config, 'INITIAL_USD', 10000.0)
VOL_TARGET = getattr(config, 'VOL_TARGET', 0.25)
MAX_GROSS_EXPOSURE = getattr(config, 'MAX_GROSS_EXPOSURE', 0.80)
KELLY_FRACTION = getattr(config, 'KELLY_FRACTION', 0.50)
EMA_FAST = getattr(config, 'EMA_FAST', 20)
EMA_SLOW = getattr(config, 'EMA_SLOW', 50)
EMA_BULL_THRESHOLD = getattr(config, 'EMA_BULL_THRESHOLD', 0.0)
EMA_EXIT_THRESHOLD = getattr(config, 'EMA_EXIT_THRESHOLD', -0.03)
CRISIS_VOL_THRESHOLD = getattr(config, 'CRISIS_VOL_THRESHOLD', 2.5)
CRISIS_DD_THRESHOLD = getattr(config, 'CRISIS_DD_THRESHOLD', -0.15)
CRISIS_CRASH_THRESHOLD = getattr(config, 'CRISIS_CRASH_THRESHOLD', -0.08)
LIVE = getattr(config, 'LIVE', False)

# -----------------------------
# 0. Crisis Detector ("Oh Shit" Gate)
# -----------------------------
class CrisisDetector:
    """
    Circuit breaker for dangerous market conditions.

    Triggers on:
    - Volatility explosion (ATR > threshold × median)
    - Severe drawdown (price drop from recent high)
    - Crash velocity (fast drop in 5 days)
    """

    def __init__(
        self,
        vol_explosion_threshold: float = 2.5,
        severe_dd_threshold: float = -0.15,
        crash_velocity_threshold: float = -0.08,
        lookback_dd: int = 30,
        lookback_vol: int = 90,
    ):
        self.vol_explosion_threshold = vol_explosion_threshold
        self.severe_dd_threshold = severe_dd_threshold
        self.crash_velocity_threshold = crash_velocity_threshold
        self.lookback_dd = lookback_dd
        self.lookback_vol = lookback_vol

    def compute_atr(self, high, low, close, period=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def detect(self, df):
        """Check if we're in crisis mode."""
        close = df['close'].astype(float)
        high = df.get('high', close * 1.01).astype(float)
        low = df.get('low', close * 0.99).astype(float)

        reasons = []

        # 1. Volatility Explosion
        atr = self.compute_atr(high, low, close, 14)
        atr_median = atr.rolling(self.lookback_vol).median()
        if len(atr) > 0 and len(atr_median.dropna()) > 0:
            current_atr = atr.iloc[-1]
            median_atr = atr_median.iloc[-1]
            if pd.notna(current_atr) and pd.notna(median_atr) and median_atr > 0:
                vol_ratio = current_atr / median_atr
                if vol_ratio > self.vol_explosion_threshold:
                    reasons.append(f"Volatility explosion ({vol_ratio:.1f}x)")

        # 2. Severe Drawdown
        if len(close) >= self.lookback_dd:
            rolling_max = close.rolling(self.lookback_dd).max()
            drawdown = close / rolling_max - 1
            current_dd = drawdown.iloc[-1]
            if current_dd < self.severe_dd_threshold:
                reasons.append(f"Drawdown ({current_dd*100:.1f}%)")

        # 3. Crash Velocity
        if len(close) >= 5:
            five_day_return = close.iloc[-1] / close.iloc[-5] - 1
            if five_day_return < self.crash_velocity_threshold:
                reasons.append(f"Crash ({five_day_return*100:.1f}% in 5d)")

        is_crisis = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "All clear"

        return is_crisis, reason_str


# -----------------------------
# 1. Trend Signal Generator
# -----------------------------
class TrendSignal:
    """
    Trend-following signal based on EMA crossover.

    BULL: EMA_fast > EMA_slow (uptrend)
    BEAR: EMA_fast < EMA_slow by threshold (downtrend)
    """

    def __init__(self, fast_period=20, slow_period=50,
                 bull_threshold=0.0, exit_threshold=-0.03):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.bull_threshold = bull_threshold
        self.exit_threshold = exit_threshold

    def compute(self, df):
        """Compute trend signal and return (is_bull, ema_ratio)."""
        close = df['close'].astype(float)

        ema_fast = close.ewm(span=self.fast_period).mean()
        ema_slow = close.ewm(span=self.slow_period).mean()

        ema_ratio = (ema_fast / ema_slow - 1).iloc[-1]

        is_bull = ema_ratio > self.bull_threshold
        is_exit = ema_ratio < self.exit_threshold

        return is_bull, is_exit, ema_ratio


# -----------------------------
# 2. Broker Interface (Abstract)
# -----------------------------
class BrokerInterface:
    """Abstract broker interface - implement for specific brokers."""

    def get_balance(self) -> float:
        raise NotImplementedError

    def get_position(self, symbol: str) -> float:
        raise NotImplementedError

    def get_price(self, symbol: str) -> float:
        raise NotImplementedError

    def get_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        raise NotImplementedError

    def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        raise NotImplementedError


class WebullBroker(BrokerInterface):
    """
    Webull broker implementation.

    Requires: pip install webull
    Auth: wb.login(email, password) or wb.get_trade_token(password)
    """

    def __init__(self, email=None, password=None, trade_pin=None):
        self.email = email or os.getenv('WEBULL_EMAIL')
        self.password = password or os.getenv('WEBULL_PASSWORD')
        self.trade_pin = trade_pin or os.getenv('WEBULL_TRADE_PIN')
        self.wb = None
        self._connect()

    def _connect(self):
        try:
            from webull import webull
            self.wb = webull()

            if self.email and self.password:
                self.wb.login(self.email, self.password)
                if self.trade_pin:
                    self.wb.get_trade_token(self.trade_pin)
                print(f"✓ Connected to Webull")
            else:
                print("⚠ Webull credentials not set - running in read-only mode")
        except ImportError:
            print("⚠ webull package not installed - running in simulation mode")
            self.wb = None
        except Exception as e:
            print(f"⚠ Webull connection failed: {e}")
            self.wb = None

    def get_balance(self) -> float:
        if not self.wb:
            return INITIAL_USD
        try:
            account = self.wb.get_account()
            return float(account.get('accountMembers', [{}])[0].get('value', 0))
        except:
            return INITIAL_USD

    def get_position(self, symbol: str) -> float:
        if not self.wb:
            return 0.0
        try:
            positions = self.wb.get_positions()
            for pos in positions:
                if pos.get('ticker', {}).get('symbol') == symbol:
                    return float(pos.get('position', 0))
            return 0.0
        except:
            return 0.0

    def get_price(self, symbol: str) -> float:
        if not self.wb:
            return 0.0
        try:
            quote = self.wb.get_quote(symbol)
            return float(quote.get('close', 0))
        except:
            return 0.0

    def get_historical_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        if not self.wb:
            return pd.DataFrame()
        try:
            bars = self.wb.get_bars(symbol, interval='d', count=days)
            df = pd.DataFrame(bars)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        except:
            return pd.DataFrame()

    def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        if not self.wb:
            return {'status': 'simulated', 'quantity': quantity, 'side': side}
        try:
            if side == 'buy':
                result = self.wb.place_order(stock=symbol, qty=int(quantity),
                                            action='BUY', orderType='MKT')
            else:
                result = self.wb.place_order(stock=symbol, qty=int(quantity),
                                            action='SELL', orderType='MKT')
            return result
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


class SimulatedBroker(BrokerInterface):
    """Simulated broker for backtesting and dry runs."""

    def __init__(self, initial_cash=10000.0, price_source='yahoo'):
        self.cash = initial_cash
        self.positions = {}
        self.price_source = price_source
        self._price_cache = {}

    def get_balance(self) -> float:
        return self.cash

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def get_price(self, symbol: str) -> float:
        # Return last cached price or fetch new
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        return 0.0

    def get_historical_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Fetch from Yahoo Finance (or return empty for simulation)."""
        try:
            # Try to fetch from Yahoo Finance API directly
            end = datetime.now()
            start = end - pd.Timedelta(days=days)

            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': int(start.timestamp()),
                'period2': int(end.timestamp()),
                'interval': '1d'
            }
            headers = {'User-Agent': 'Mozilla/5.0'}

            resp = requests.get(url, params=params, headers=headers, timeout=10)
            data = resp.json()

            if 'chart' in data and 'result' in data['chart']:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]

                df = pd.DataFrame({
                    'open': quotes['open'],
                    'high': quotes['high'],
                    'low': quotes['low'],
                    'close': quotes['close'],
                    'volume': quotes['volume']
                }, index=pd.to_datetime(timestamps, unit='s'))

                self._price_cache[symbol] = df['close'].iloc[-1]
                return df
        except Exception as e:
            print(f"⚠ Could not fetch data for {symbol}: {e}")

        return pd.DataFrame()

    def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        price = self.get_price(symbol)
        if price <= 0:
            return {'status': 'error', 'message': 'No price available'}

        if side == 'buy':
            cost = quantity * price
            if cost > self.cash:
                return {'status': 'error', 'message': 'Insufficient funds'}
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:  # sell
            current_pos = self.positions.get(symbol, 0)
            if quantity > current_pos:
                quantity = current_pos  # Can only sell what we have
            self.cash += quantity * price
            self.positions[symbol] = current_pos - quantity

        return {'status': 'filled', 'quantity': quantity, 'price': price, 'side': side}


# -----------------------------
# 3. Initialize Components
# -----------------------------
print(f"\n{'='*60}")
print(f"Q-PRIME TREND-FOLLOWING SYSTEM")
print(f"{'='*60}")
print(f"Symbol: {SYMBOL}")
print(f"Config: {args.config}")
print(f"Mode: {'DRY RUN' if args.dry_run else ('LIVE' if LIVE else 'PAPER')}")
print(f"{'='*60}")

# Initialize crisis detector
crisis_detector = CrisisDetector(
    vol_explosion_threshold=CRISIS_VOL_THRESHOLD,
    severe_dd_threshold=CRISIS_DD_THRESHOLD,
    crash_velocity_threshold=CRISIS_CRASH_THRESHOLD,
)

# Initialize trend signal
trend_signal = TrendSignal(
    fast_period=EMA_FAST,
    slow_period=EMA_SLOW,
    bull_threshold=EMA_BULL_THRESHOLD,
    exit_threshold=EMA_EXIT_THRESHOLD,
)

# Initialize broker
if args.dry_run:
    broker = SimulatedBroker(initial_cash=INITIAL_USD)
    print("Using: Simulated Broker (dry run)")
elif LIVE:
    broker = WebullBroker()
    print("Using: Webull Broker (LIVE)")
else:
    broker = SimulatedBroker(initial_cash=INITIAL_USD)
    print("Using: Simulated Broker (paper trade)")

print(f"{'='*60}\n")

# -----------------------------
# 4. Trading Loop
# -----------------------------
initial_cash = INITIAL_USD
equity_curve = []
iteration = 0

try:
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        # Fetch historical data
        df = broker.get_historical_data(SYMBOL, days=100)

        if df.empty or len(df) < EMA_SLOW + 10:
            print(f"⚠ Insufficient data: {len(df)} candles (need {EMA_SLOW + 10}+)")
            time.sleep(300)
            continue

        price = df['close'].iloc[-1]
        position = broker.get_position(SYMBOL)
        cash = broker.get_balance()
        equity = cash + position * price

        # Compute signals
        is_crisis, crisis_reason = crisis_detector.detect(df)
        is_bull, is_exit, ema_ratio = trend_signal.compute(df)

        # Calculate volatility for position sizing
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        recent_vol = df['return'].iloc[-60:].std() * np.sqrt(252) if len(df) >= 60 else 0.25

        # Display status
        print(f"Price: ${price:.2f} | Position: {position:.2f} shares")
        print(f"EMA Ratio: {ema_ratio*100:.2f}% | Trend: {'BULL' if is_bull else 'BEAR/NEUTRAL'}")
        print(f"60d Vol: {recent_vol:.1%} | Equity: ${equity:,.2f}")

        # Determine target position
        target = 0.0

        # Gate 1: Crisis Detector
        if is_crisis:
            print(f"🚨 CRISIS GATE: {crisis_reason}")
            print(f"   → EXIT ALL (dangerous market conditions)")
            target = 0.0

        # Gate 2: Trend Signal
        elif is_bull:
            # Calculate position size based on volatility
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            target = risk * KELLY_FRACTION
            print(f"✓ BULL TREND: EMA{EMA_FAST} > EMA{EMA_SLOW}")
            print(f"   → Target exposure: {target:.1%}")

        elif is_exit:
            print(f"📉 TREND BREAK: EMA ratio {ema_ratio*100:.1f}% < {EMA_EXIT_THRESHOLD*100:.0f}%")
            print(f"   → EXIT (trend broken)")
            target = 0.0

        else:
            print(f"⏸ NEUTRAL: No clear trend")
            target = 0.0

        # Calculate target position in shares
        target_value = equity * target
        target_shares = target_value / price if price > 0 else 0
        position_diff = target_shares - position

        # Execute trades
        if abs(position_diff) > 0.5:  # Minimum trade size (0.5 shares)
            if args.dry_run:
                print(f"[DRY RUN] Would {'BUY' if position_diff > 0 else 'SELL'} {abs(position_diff):.2f} shares")
            else:
                if position_diff > 0:
                    print(f"📈 BUY {position_diff:.2f} shares @ ${price:.2f}")
                    result = broker.place_order(SYMBOL, 'buy', position_diff)
                else:
                    print(f"📉 SELL {abs(position_diff):.2f} shares @ ${price:.2f}")
                    result = broker.place_order(SYMBOL, 'sell', abs(position_diff))
                print(f"   Order result: {result}")
        else:
            print("→ No trade (position already optimal)")

        # Update tracking
        equity_curve.append(equity)
        pnl_pct = (equity / initial_cash - 1) * 100
        print(f"\n💰 EQUITY: ${equity:,.2f} | PnL: {pnl_pct:+.2f}%")

        # Sleep until next check (daily for stocks)
        print(f"\nNext check in 24 hours...")
        time.sleep(86400)

except KeyboardInterrupt:
    print("\n\n⚠ Trading stopped by user")

# Final stats
print(f"\n{'='*60}")
print("FINAL STATS")
print(f"{'='*60}")
if equity_curve:
    final_equity = equity_curve[-1]
    total_return = (final_equity / initial_cash - 1) * 100
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Iterations: {len(equity_curve)}")
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"{'='*60}")
