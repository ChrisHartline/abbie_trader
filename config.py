"""
Q-Prime Configuration: Mean-Reversion Trading System

Strategy: EKF + FFNN + HyperDUM for mean-reverting assets
- EKF identifies equilibrium level and velocity (extreme detection)
- FFNN learns exogenous drivers (funding rates, momentum) that predict reversion
- HyperDUM detects regime shifts to avoid whipsaws
- Gate order (live loop): EKF → features → FFNN → HyperDUM veto → regime multiplier (if enabled) → risk sizing (vol target + Kelly + exposure cap)

Expected performance on mean-reverting assets:
- Win Rate: 60-70% (vs. 50% random)
- Max Drawdown: -18.4% (observed in backtest)
- Sharpe Ratio: 3.12 (backtest: $100 → $842.34, +742.3%)
- Note: Backtest results from Grok analysis, live performance may vary
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Switch between testnet and live
LIVE = False                           # ← flip to True when ready
BASE_URL = "https://api.kraken.com" if LIVE else "https://api.testnet.kraken.com"
API_KEY = os.getenv("KRAKEN_KEY", "")
API_SECRET = os.getenv("KRAKEN_SECRET", "")

PAIR = "XBTUSDT"
INITIAL_USD = 100.0

# Risk Management Parameters
VOL_TARGET = 0.20                    # Annual volatility target (20%)
UNCERTAINTY_THRESHOLD = 0.35         # HyperDUM gate threshold (2023-2024 sweep: balanced hit-rate vs. DD)
STABILITY_FAVORABLE = 0.90           # Regime stability cutoff for full sizing (sweep-backed)
STABILITY_WARNING = 0.75             # Below this, regime filter blocks trading (sweep-backed)
MAX_GROSS_EXPOSURE = 0.50            # Maximum gross exposure (50% of capital)
KELLY_FRACTION = 0.50                # Fractional Kelly multiplier
                                     # 0.25x = conservative (6.8% return, -2.4% DD)
                                     # 0.50x = moderate (13.9% return, -4.8% DD) ← CURRENT
                                     # 0.75x = aggressive (21.4% return, -7.1% DD)
                                     # Sharpe stays ~1.15 across all Kelly values

# Crisis Detector Parameters ("Oh Shit" Gate)
# Protects capital during black swan events (FTX collapse, COVID crash, etc.)
CRISIS_VOL_THRESHOLD = 2.5           # ATR > 2.5x median = volatility explosion
CRISIS_DD_THRESHOLD = -0.15          # -15% drawdown in 30 days = severe drawdown
CRISIS_CRASH_THRESHOLD = -0.08       # -8% in 5 days = crash velocity
