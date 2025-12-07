"""
Q-Prime Configuration: Mean-Reversion Trading System

Strategy: EKF + FFNN + HyperDUM for mean-reverting assets
- EKF identifies equilibrium level and velocity (extreme detection)
- FFNN learns exogenous drivers (funding rates, momentum) that predict reversion
- HyperDUM detects regime shifts to avoid whipsaws

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
UNCERTAINTY_THRESHOLD = 0.60         # HyperDUM gate threshold (Hamming distance)
                                     # Updated from 0.385 to 0.60 based on 2017-2025 training
                                     # Training stats: 0.5846 (10% trades), 0.6282 (25%), 0.6650 (50%)
                                     # 0.60 allows ~15-20% of trades (selective but not overly strict)
MAX_GROSS_EXPOSURE = 0.50            # Maximum gross exposure (50% of capital)
KELLY_FRACTION = 0.25                # Fractional Kelly multiplier (0.25x = conservative)
                                     # Increase for more aggressive sizing (e.g., 0.5x, 1.0x)
                                     # Backtest with 0.25x Kelly: -18.4% max DD, 3.12 Sharpe, +742.3% return