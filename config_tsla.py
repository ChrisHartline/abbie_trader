"""
Q-Prime Configuration: TSLA Trend-Following System

Strategy: Trend-following for momentum stocks
- Stay LONG when EMA20 > EMA50 (bull market)
- EXIT when trend breaks (EMA20 < EMA50 by 3%)
- Crisis detector protects during crashes/volatility spikes

This is DIFFERENT from the BTC mean-reversion system.
For momentum stocks, we ride the trend instead of fading extremes.

Expected performance on momentum stocks (TSLA simulation):
- Total Return: ~80% of Buy & Hold
- Max Drawdown: -15% (vs ~-35% B&H during corrections)
- Sharpe Ratio: ~1.25
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Asset configuration
ASSET_TYPE = "STOCK"  # STOCK or CRYPTO
SYMBOL = "TSLA"

# API configuration (for future broker integration)
LIVE = False
API_KEY = os.getenv("BROKER_KEY", "")
API_SECRET = os.getenv("BROKER_SECRET", "")

INITIAL_USD = 10_000.0

# Risk Management Parameters - AGGRESSIVE for trend-following
VOL_TARGET = 0.30                    # Higher vol target for growth stocks
MAX_GROSS_EXPOSURE = 0.80            # Higher max exposure
KELLY_FRACTION = 0.50                # More aggressive Kelly

# HyperDUM - DISABLED for trend-following stocks
# (Mean-reversion patterns don't apply to momentum stocks)
UNCERTAINTY_THRESHOLD = 1.0          # Effectively disabled
USE_HYPERDUM = False                 # Skip HyperDUM check entirely

# Trend-Following Parameters
EMA_FAST = 20                        # Fast EMA period
EMA_SLOW = 50                        # Slow EMA period
EMA_BULL_THRESHOLD = 0.0             # EMA20 > EMA50 = bull
EMA_EXIT_THRESHOLD = -0.03           # Exit when EMA20 < EMA50 by 3%

# Crisis Detector Parameters - RELAXED for stocks (more volatile)
CRISIS_VOL_THRESHOLD = 3.0           # ATR > 3x median (stocks are more volatile)
CRISIS_DD_THRESHOLD = -0.20          # -20% drawdown (stocks can drop more)
CRISIS_CRASH_THRESHOLD = -0.12       # -12% in 5 days

# Strategy mode
STRATEGY_MODE = "TREND_FOLLOWING"    # TREND_FOLLOWING or MEAN_REVERSION
LONG_ONLY = True                     # Don't short momentum stocks
