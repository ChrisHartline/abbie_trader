"""
Q-Prime Configuration: TSLL (2x TSLA) Trend-Following System

Strategy: Trend-following for 2x leveraged ETFs
- Crisis detector is CRITICAL (drawdowns are amplified)
- Tighter thresholds to exit early
- Accept volatility decay as cost of leverage

Key differences from TSLA config:
- Tighter crisis thresholds (2x amplifies everything)
- Lower Kelly fraction (already levered)
- HyperDUM disabled (trend-following, not mean-reversion)

IMPORTANT: 2x ETFs suffer volatility decay in choppy markets.
This strategy works best in strong trending environments.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Asset configuration
ASSET_TYPE = "LEVERAGED_ETF"
SYMBOL = "TSLL"                      # 2x TSLA ETF
LEVERAGE_FACTOR = 2.0                # Built-in leverage

# API configuration (for future broker integration)
LIVE = False
API_KEY = os.getenv("BROKER_KEY", "")
API_SECRET = os.getenv("BROKER_SECRET", "")

INITIAL_USD = 10_000.0

# Risk Management Parameters - CONSERVATIVE for leveraged products
# The ETF already provides 2x, so we size conservatively
VOL_TARGET = 0.25                    # Moderate vol target
MAX_GROSS_EXPOSURE = 0.70            # Not 100% - leave room for crisis exit
KELLY_FRACTION = 0.40                # Conservative - leverage is built in

# HyperDUM - DISABLED for trend-following
# Mean-reversion patterns don't apply to momentum/trend strategies
UNCERTAINTY_THRESHOLD = 1.0          # Effectively disabled
USE_HYPERDUM = False

# Trend-Following Parameters
EMA_FAST = 20                        # Fast EMA period
EMA_SLOW = 50                        # Slow EMA period
EMA_BULL_THRESHOLD = 0.0             # EMA20 > EMA50 = bull
EMA_EXIT_THRESHOLD = -0.02           # Exit earlier on 2x products

# Crisis Detector Parameters - TIGHTER for leveraged products
# 2x amplifies drawdowns, so we need to exit EARLIER
CRISIS_VOL_THRESHOLD = 2.0           # Tighter: ATR > 2x median (was 2.5)
CRISIS_DD_THRESHOLD = -0.10          # Tighter: -10% drawdown (was -0.15)
CRISIS_CRASH_THRESHOLD = -0.06       # Tighter: -6% in 5 days (was -0.08)

# Strategy mode
STRATEGY_MODE = "TREND_FOLLOWING"
LONG_ONLY = True                     # Don't short leveraged ETFs

# Decay awareness
EXPECT_ANNUAL_DECAY = 0.10           # ~10% annual decay in choppy markets
# This is the "cost" of using leveraged ETF vs margin
