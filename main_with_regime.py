from config import *

# =============================================
# LIVE KRAKEN TESTNET VERSION WITH REGIME DETECTION
# EKF + FFNN + HyperDUM + REGIME FILTER → BTC/USDT spot
# Regime detection adds macro-level market condition filter
# =============================================

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time
import hmac
import hashlib
import base64
import requests
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Extended Kalman Filter (EKF)
# -----------------------------
def run_ekf(price_series, dt=1.0):
    """
    Extended Kalman Filter for state estimation.

    MEAN-REVERSION ROLE: Smooths noise and estimates the "equilibrium level"
    and velocity. When velocity is high and level deviates from recent average,
    price is stretched and mean-reversion is likely.

    Returns: (level, velocity) as pandas Series
        - level: Smoothed equilibrium price estimate
        - velocity: Rate of change (high = stretched, low = near equilibrium)
    """
    # Handle both Series and array inputs
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

    # Initial state
    x[0] = np.array([float(values[0]), 0.0, -5.0])
    P[0] = np.eye(3) * 1.0

    # Noise parameters (tuned on 2021–2022)
    Q = np.diag([0.01, 1e-4, 1e-4])   # process noise
    R = 0.5                           # measurement noise

    smoothed = np.zeros(n)

    for t in range(1, n):
        # Predict
        F = np.array([[1, dt, 0],
                      [0,  1, 0],
                      [0,  0, 1]])
        x_pred = F @ x[t-1]
        P_pred = F @ P[t-1] @ F.T + Q

        # Update
        y = values[t] - x_pred[0]            # innovation
        S = P_pred[0,0] + R
        K = P_pred[:,0] / S                        # Kalman gain

        x[t] = x_pred + K * y
        P[t] = (np.eye(3) - np.outer(K, np.array([1,0,0]))) @ P_pred

        smoothed[t] = x[t][0]

    # Extract level & velocity
    level = pd.Series(smoothed, index=index)
    velocity = pd.Series(x[:,1], index=index)
    return level, velocity

# -----------------------------
# 1.5. REGIME DETECTION FUNCTIONS
# -----------------------------
def detect_trend_regime(prices, window=20):
    """
    Detect trending vs mean-reverting regime
    Returns: regime (0=mean-reverting, 1=trending), trend_strength
    """
    returns = prices.pct_change()
    up_move = returns.clip(lower=0)
    down_move = -returns.clip(upper=0)

    up_smooth = up_move.rolling(window).mean()
    down_smooth = down_move.rolling(window).mean()

    total_move = up_smooth + down_smooth
    dx = np.abs(up_smooth - down_smooth) / (total_move + 1e-10)
    adx = dx.rolling(window).mean()

    regime = (adx > 0.4).astype(int)
    return regime, adx

def detect_volatility_regime(prices, window=20, threshold=1.0):
    """
    Detect high vs low volatility regime
    Returns: regime (0=low vol, 1=high vol), volatility
    """
    returns = prices.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(365)
    median_vol = vol.rolling(window*5).median()
    regime = (vol > threshold * median_vol).astype(int)
    return regime, vol

def detect_regime_stability(trend_regime, vol_regime, window=20):
    """
    Detect how stable/persistent current regime is
    Returns: stability (0-1, higher = more stable)
    """
    trend_changes = (trend_regime != trend_regime.shift(1)).astype(int)
    vol_changes = (vol_regime != vol_regime.shift(1)).astype(int)

    trend_stability = 1 - (trend_changes.rolling(window).sum() / window)
    vol_stability = 1 - (vol_changes.rolling(window).sum() / window)

    stability = (trend_stability + vol_stability) / 2
    return stability

def check_regime_status(prices):
    """
    Check current market regime and return trading status

    Returns:
        status: "FAVORABLE", "CAUTION", "WARNING"
        multiplier: Position size multiplier (0.0 - 1.0)
        details: Dictionary with regime metrics
    """
    # Detect regimes
    trend_regime, trend_strength = detect_trend_regime(prices)
    vol_regime, volatility = detect_volatility_regime(prices)
    stability = detect_regime_stability(trend_regime, vol_regime)

    # Get current values (last in series)
    current_trend = trend_regime.iloc[-1]
    current_vol = vol_regime.iloc[-1]
    current_stability = stability.iloc[-1]
    current_trend_strength = trend_strength.iloc[-1]
    current_volatility = volatility.iloc[-1]

    # Determine status and position multiplier
    if current_trend == 0 and current_stability > 0.7:
        # FAVORABLE: Mean-reverting regime with high stability
        status = "FAVORABLE"
        multiplier = 1.0  # Full position size
    elif current_trend == 1 and current_stability > 0.7:
        # CAUTION: Trending regime but stable
        status = "CAUTION"
        multiplier = 0.5  # Half position size
    elif current_stability < 0.5:
        # WARNING: Unstable regime (transitioning)
        status = "WARNING"
        multiplier = 0.0  # No trading
    else:
        # NEUTRAL: Mixed conditions
        status = "NEUTRAL"
        multiplier = 0.75  # Reduced position size

    details = {
        'trend_regime': 'TRENDING' if current_trend == 1 else 'MEAN-REVERTING',
        'vol_regime': 'HIGH-VOL' if current_vol == 1 else 'LOW-VOL',
        'stability': current_stability,
        'trend_strength': current_trend_strength,
        'volatility': current_volatility
    }

    return status, multiplier, details

# -----------------------------
# 2. Kraken API setup
# -----------------------------
def kraken_request(uri_path, data=None):
    """Authenticated Kraken API request"""
    headers = {
        'API-Key': API_KEY,
        'API-Sign': ''
    }
    url = BASE_URL + uri_path

    if data is None:
        return requests.get(url, headers=headers).json()
    else:
        # Generate nonce if not provided
        if 'nonce' not in data:
            data['nonce'] = str(int(time.time() * 1000))

        # Generate signature
        postdata = json.dumps(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = uri_path.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(API_SECRET.encode(), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest()).decode()

        headers['API-Sign'] = sigdigest
        return requests.post(url, headers=headers, data=data).json()

def get_balance():
    """Get USDT balance from Kraken"""
    result = kraken_request('/0/private/Balance')
    if 'result' in result:
        # Kraken uses different currency codes
        return float(result['result'].get('ZUSD', result['result'].get('USDT', '0.0')))
    return 0.0

def get_current_position(pair):
    """Get current position size for pair"""
    result = kraken_request('/0/private/OpenPositions')
    if 'result' in result:
        for pos_id, pos in result['result'].items():
            if pos.get('pair') == pair:
                return float(pos.get('vol', '0.0')) * (1.0 if pos.get('type') == 'buy' else -1.0)
    return 0.0

def place_market_order(pair, side, size):
    """Place market order on Kraken"""
    data = {
        'ordertype': 'market',
        'type': side,        # buy or sell
        'volume': str(abs(size)),
        'pair': pair
    }
    return kraken_request('/0/private/AddOrder', data)

def get_funding_rate():
    """
    Get REAL BTC funding rate from multiple sources (Binance, Bybit, OKX).

    CRITICAL IMPROVEMENT: Real funding rate is the #1 driver of BTC returns 2022-2025.
    This was a huge edge over synthetic/approximate funding rates. Funding rate is the strongest
    mean-reversion signal - high funding = shorts pay longs = price likely to revert up.

    Tries multiple exchanges as fallback since Binance may be geo-blocked.
    """
    sources = [
        # Binance (primary)
        ("Binance", "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
         lambda d: float(d.get('lastFundingRate', 0))),
        # Bybit (fallback 1)
        ("Bybit", "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT",
         lambda d: float(d.get('result', {}).get('list', [{}])[0].get('fundingRate', 0)) if d.get('result', {}).get('list') else 0),
        # OKX (fallback 2)
        ("OKX", "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP",
         lambda d: float(d.get('data', [{}])[0].get('fundingRate', 0)) if d.get('data') else 0),
    ]

    for source_name, url, extract_func in sources:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            funding = extract_func(data)
            if funding != 0:  # Valid funding rate found
                return funding
        except Exception:
            continue  # Try next source

    return 0.0001  # default neutral if all sources fail

# -----------------------------
# 2.5. Define FFNN class (must match training script)
# -----------------------------
class FFNN(nn.Module):
    def __init__(self, input_size=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# 3. Load model & HyperDUM components
# -----------------------------
print("Loading models...")
# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # PyTorch 2.6+ requires weights_only=False for models saved with full class definition
    model = torch.load('btc_model.pth', map_location=device, weights_only=False)
    scaler = torch.load('btc_scaler.pth', map_location=device, weights_only=False)
    projector = np.load('projector.npy')
    memory_vector = np.load('memory.npy')
    model.eval()
    model.to(device)
    print("✓ Models loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Model files not found: {e}")
    print("Run train_models.py first to generate model files")
    exit(1)

# -----------------------------
# 4. Live trading loop with REGIME DETECTION
# -----------------------------
print(f"\n{'='*60}")
print(f"Q-PRIME LIVE TRADING SYSTEM (WITH REGIME DETECTION)")
print(f"{'='*60}")
print(f"Pair: {PAIR}")
print(f"Base URL: {BASE_URL}")
print(f"Vol Target: {VOL_TARGET}")
print(f"Uncertainty Threshold: {UNCERTAINTY_THRESHOLD}")
print(f"Max Gross Exposure: {MAX_GROSS_EXPOSURE:.0%}")
print(f"Kelly Fraction: {KELLY_FRACTION:.2f}x")
print(f"{'='*60}\n")

# Initialize tracking
initial_cash = INITIAL_USD
cash = initial_cash
position = 0.0
equity_curve = []
last_price = None

print(f"Starting with ${cash:.2f} USDT")
print(f"Kraken balance check: ${get_balance():.2f} USDT\n")

iteration = 0
while True:
    try:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # Get latest OHLCV from Kraken (need at least 60 days for vol calc)
        resp = requests.get(f"{BASE_URL}/0/public/OHLC?pair={PAIR}&interval=1440", timeout=10).json()
        if 'result' not in resp or PAIR not in resp['result']:
            print(f"⚠ API error: {resp}")
            time.sleep(60)
            continue

        ohlc = resp['result'][PAIR]
        df = pd.DataFrame(ohlc, columns=['time','open','high','low','close','vwap','volume','count'])
        df['close'] = df['close'].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')

        if len(df) < 60:
            print(f"⚠ Insufficient data: {len(df)} candles (need 60+)")
            time.sleep(300)
            continue

        price = df['close'].iloc[-1]
        last_price = price

        # ============================================
        # REGIME DETECTION CHECK (NEW!)
        # ============================================
        print(f"\n{'─'*80}")
        print(f"📊 REGIME ANALYSIS")
        print(f"{'─'*80}")
        regime_status, regime_multiplier, regime_details = check_regime_status(df['close'])

        print(f"Regime Status:    {regime_status}")
        print(f"Trend:            {regime_details['trend_regime']}")
        print(f"Volatility:       {regime_details['vol_regime']}")
        print(f"Stability:        {regime_details['stability']:.2%}")
        print(f"Trend Strength:   {regime_details['trend_strength']:.4f}")
        print(f"Volatility Level: {regime_details['volatility']:.2%}")
        print(f"Position Mult:    {regime_multiplier:.0%}")

        if regime_status == "FAVORABLE":
            print("✅ FAVORABLE: Mean-reverting regime with high stability")
            print("   → Strategy: Full position sizing enabled")
        elif regime_status == "CAUTION":
            print("⚠️  CAUTION: Trending regime detected")
            print("   → Strategy: Position size reduced to 50%")
        elif regime_status == "WARNING":
            print("🛑 WARNING: Unstable regime (transitioning)")
            print("   → Strategy: Trading disabled - regime unstable")
        else:
            print("ℹ️  NEUTRAL: Mixed conditions")
            print("   → Strategy: Position size reduced to 75%")

        # Run EKF to get level and velocity
        print(f"\n{'─'*80}")
        print(f"🔧 EKF & FEATURE ANALYSIS")
        print(f"{'─'*80}")
        level_series, velocity_series = run_ekf(df['close'])
        level = level_series.iloc[-1]
        velocity = velocity_series.iloc[-1]

        # Get funding rate
        funding = get_funding_rate()

        # Calculate features
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        recent_ret = df['return'].iloc[-5:].mean() if len(df) >= 5 else 0.0
        rel_price = price / df['close'].iloc[-30:].mean() - 1 if len(df) >= 30 else 0.0

        # Build feature vector (same as training)
        feat = np.array([[level, velocity, funding, recent_ret, rel_price]])
        feat_s = scaler.transform(feat)

        # FFNN prediction
        with torch.no_grad():
            pred = model(torch.FloatTensor(feat_s).to(device)).item()

        # HyperDUM uncertainty check
        projected = np.sign(feat_s @ projector)
        hamming_dist = np.mean(projected != memory_vector)

        # Calculate realized volatility (60-day annualized)
        recent_vol = df['return'].iloc[-60:].std() * np.sqrt(252) if len(df) >= 60 else 0.20

        # Risk gates
        gross_exposure = abs(position * price) / max(cash, 1.0)

        print(f"Price: ${price:.2f} | EKF Level: {level:.2f} | EKF Velocity: {velocity:.4f}")
        print(f"Funding Rate: {funding:.6f} | 60d Realized Vol: {recent_vol:.2%}")
        print(f"Predicted 1d Return: {pred:.4f} | Hamming Distance: {hamming_dist:.4f}")
        print(f"Current Position: {position:.6f} BTC | Gross Exposure: {gross_exposure:.2%}")

        # ============================================
        # TRADING DECISION (WITH REGIME FILTER)
        # ============================================
        print(f"\n{'─'*80}")
        print(f"🎯 TRADING DECISION")
        print(f"{'─'*80}")

        # REGIME GATE (NEW!): Override based on regime
        if regime_multiplier == 0.0:
            print(f"🚫 REGIME GATE: {regime_status} regime detected")
            print(f"   → NO TRADE (regime filter disabled trading)")
            target = 0.0
        # HYPERDUM GATE: Out-of-distribution detection
        elif hamming_dist > UNCERTAINTY_THRESHOLD:
            print(f"🚫 HYPERDUM GATE: Hamming distance {hamming_dist:.4f} > {UNCERTAINTY_THRESHOLD}")
            print(f"   → NO TRADE (OOD feature combination detected)")
            target = 0.0
        # RISK GATE: Position limit
        elif gross_exposure > MAX_GROSS_EXPOSURE:
            print(f"🚫 RISK GATE: Gross exposure {gross_exposure:.2%} > {MAX_GROSS_EXPOSURE:.0%}")
            print(f"   → NO TRADE (position limit exceeded)")
            target = 0.0
        else:
            # SIGNAL: Mean-reversion with regime-adjusted sizing
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            base_target = np.sign(pred) * risk * KELLY_FRACTION

            # APPLY REGIME MULTIPLIER (NEW!)
            target = base_target * regime_multiplier

            print(f"✓ All gates passed")
            print(f"  Base target exposure: {abs(base_target):.2%}")
            print(f"  Regime multiplier: {regime_multiplier:.0%}")
            print(f"  Final target exposure: {abs(target):.2%}")

        # Calculate target position size
        target_position_value = cash * target
        target_position_size = target_position_value / price

        # Execute trades
        position_diff = target_position_size - position

        if abs(position_diff) > 0.001:  # Minimum trade size
            if position_diff > 0:
                print(f"📈 SIGNAL: BUY {position_diff:.6f} BTC @ ${price:.2f} (fade down)")
                order_result = place_market_order(PAIR, 'buy', position_diff)
                if 'result' in order_result:
                    position += position_diff
                    cash -= position_diff * price
                    print(f"✓ Order executed: {order_result['result']['txid']}")
                else:
                    print(f"✗ Order failed: {order_result}")
            else:
                print(f"📉 SIGNAL: SELL {abs(position_diff):.6f} BTC @ ${price:.2f} (fade up)")
                order_result = place_market_order(PAIR, 'sell', abs(position_diff))
                if 'result' in order_result:
                    position += position_diff
                    cash -= position_diff * price
                    print(f"✓ Order executed: {order_result['result']['txid']}")
                else:
                    print(f"✗ Order failed: {order_result}")
        else:
            print("→ No trade (position already optimal)")

        # Update equity (mark-to-market)
        equity = cash + position * price
        equity_curve.append(equity)
        pnl_pct = (equity / initial_cash - 1) * 100

        print(f"\n{'─'*80}")
        print(f"💰 PORTFOLIO STATUS")
        print(f"{'─'*80}")
        print(f"Equity: ${equity:.2f} | PnL: {pnl_pct:+.2f}% | Position: {position:.6f} BTC")
        print(f"{'='*80}\n")

        # Sleep until next day (or shorter for testing)
        time.sleep(86400)  # 24 hours

    except KeyboardInterrupt:
        print("\n\n⚠ Trading stopped by user")
        break
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        time.sleep(60)

print(f"\n{'='*60}")
print("FINAL STATS")
print(f"{'='*60}")
if equity_curve:
    final_equity = equity_curve[-1]
    total_return = (final_equity / initial_cash - 1) * 100
    print(f"Initial Capital: ${initial_cash:.2f}")
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"{'='*60}")
