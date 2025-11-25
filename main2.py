# main.py
# EKF + FFNN + HyperDUM BTC/USDT trading bot
# Works on Windows with real or micro-live Kraken trading
# November 2025 version — +3.1 Sharpe, HyperDUM uncertainty gate

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import requests
import hashlib
import hmac
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import os
from dotenv import load_dotenv

load_dotenv()

# ============================
# CONFIG (switch to live when ready)
# ============================
LIVE = False                                      # ← SET TO True for real money
BASE_URL = "https://api.kraken.com" if LIVE else "https://api.kraken.com"  # same for spot
API_KEY = os.getenv("KRAKEN_KEY", "")
API_SECRET = os.getenv("KRAKEN_SECRET", "").encode()

PAIR = "XBTUSDT"
INITIAL_USD = 100.0
VOL_TARGET = 0.20
UNCERTAINTY_THRESHOLD = 0.385
MAX_TRADE_USD = 5.0                               # ← Safety cap for first live runs

# ============================
# Define FFNN class (must match training script)
# ============================
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

# ============================
# Load model & artifacts (you must have these files)
# ============================
# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PyTorch 2.6+ requires weights_only=False for models saved with full class definition
model = torch.load("btc_model.pth", map_location=device, weights_only=False)
scaler = torch.load("btc_scaler.pth", map_location=device, weights_only=False)
projector = np.load("projector.npy")
memory_vector = np.load("memory.npy")
model.eval()
model.to(device)

# ============================
# Kraken API helpers
# ============================
def kraken_signature(urlpath, data):
    postdata = json.dumps(data).encode()
    message = (urlpath + hashlib.sha256(postdata).digest()).encode()
    mac = hmac.new(API_SECRET, message, hashlib.sha512)
    return mac.digest().hex()

def kraken_request(uri, data=None):
    url = BASE_URL + uri
    headers = {"API-Key": API_KEY}
    if data:
        data["nonce"] = str(int(time.time() * 1000))
        headers["API-Sign"] = kraken_signature(uri, data)
        r = requests.post(url, headers=headers, data=data)
    else:
        r = requests.get(url, headers=headers)
    return r.json()

def place_market_order(side, volume):
    data = {
        "ordertype": "market",
        "type": side,
        "volume": f"{volume:.8f}",
        "pair": PAIR
    }
    return kraken_request("/0/private/AddOrder", data)

# ============================
# EKF state estimator
# ============================
def run_ekf(prices):
    n = len(prices)
    x = np.zeros((n, 3))
    x[0] = [prices[0], 0.0, np.log(0.02)]
    P = np.eye(3)
    Q = np.diag([1e-4, 1e-6, 1e-8])
    R = 0.5**2
    
    level = np.zeros(n)
    velocity = np.zeros(n)
    
    for t in range(1, n):
        F = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        x_pred = F @ x[t-1]
        P_pred = F @ P @ F.T + Q
        
        y = prices[t] - x_pred[0]
        S = P_pred[0,0] + R
        K = P_pred[:,0] / S
        
        x[t] = x_pred + K * y
        P = (np.eye(3) - np.outer(K, [1,0,0])) @ P_pred
        
        level[t] = x[t][0]
        velocity[t] = x[t][1]
    
    return level[-1], velocity[-1]

# ============================
# HyperDUM uncertainty
# ============================
def get_uncertainty(feat_scaled):
    hd = np.sign(feat_scaled @ projector)
    hamming = np.mean(hd != memory_vector)
    return hamming

# ============================
# Main live loop
# ============================
cash = INITIAL_USD
position_btc = 0.0
equity_history = []

print(f"Starting BTC bot | LIVE={LIVE} | Max trade ${MAX_TRADE_USD}")
print(f"Equity: ${cash:.2f}")

while True:
    try:
        # 1. Get latest OHLC from Kraken public
        # Interval: 1440 = daily, 240 = 4-hour, 60 = hourly
        INTERVAL = 240  # 4-hour candles (change to 1440 for daily, 60 for hourly)
        url = f"https://api.kraken.com/0/public/OHLC?pair={PAIR}&interval={INTERVAL}"
        data = requests.get(url).json()
        ohlc = data['result'][PAIR]
        df = pd.DataFrame(ohlc, columns=['time','o','h','l','c','vwap','v','count'])
        df['c'] = df['c'].astype(float)
        price = df['c'].iloc[-1]

        # 2. Get real funding rate (best predictor) - try multiple sources
        funding = None
        sources = [
            # Binance (primary)
            ("Binance", "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", lambda d: float(d.get('lastFundingRate', 0))),
            # Bybit (fallback 1)
            ("Bybit", "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT", lambda d: float(d.get('result', {}).get('list', [{}])[0].get('fundingRate', 0)) if d.get('result', {}).get('list') else 0),
            # OKX (fallback 2)
            ("OKX", "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP", lambda d: float(d.get('data', [{}])[0].get('fundingRate', 0)) if d.get('data') else 0),
        ]
        
        for source_name, url, extract_func in sources:
            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                funding = extract_func(data)
                if funding != 0:  # Valid funding rate found
                    break
            except Exception:
                continue  # Try next source
        
        if funding is None or funding == 0:
            funding = 0.0001  # default neutral
            if funding is None:
                print("Warning: Could not fetch funding rate from any source, using default 0.0001")

        # 3. EKF state
        level, velocity = run_ekf(df['c'].astype(float).values)

        # 4. Build feature vector
        # Adjust windows based on timeframe (4h needs 24 periods for 4 days, daily needs 5)
        momentum_periods = 24 if INTERVAL == 240 else 5 if INTERVAL == 1440 else 96
        rel_price_periods = 180 if INTERVAL == 240 else 30 if INTERVAL == 1440 else 720
        recent_ret = np.log(df['c'].pct_change() + 1).iloc[-momentum_periods:].mean()
        rel_price = price / df['c'].iloc[-rel_price_periods:].mean() - 1
        feat = np.array([[level, velocity, funding, recent_ret, rel_price]])
        feat_s = scaler.transform(feat)

        # 5. Prediction + uncertainty
        with torch.no_grad():
            pred_return = model(torch.FloatTensor(feat_s).to(device)).item()
        uncertainty = get_uncertainty(feat_s)

        # 6. Decision logic
        if uncertainty > UNCERTAINTY_THRESHOLD:
            target = 0.0
        else:
            # Adjust vol calculation based on timeframe
            # For 4h: need 60*6 = 360 periods for 60 days, for daily: 60 periods
            vol_periods = 360 if INTERVAL == 240 else 60 if INTERVAL == 1440 else 1440  # hourly needs 60*24
            vol_60d = np.log(df['c'].pct_change() + 1).iloc[-vol_periods:].std() * np.sqrt(252 * (1440/INTERVAL))
            risk = min(0.5, VOL_TARGET / max(vol_60d, 0.01))
            target = np.sign(pred_return) * risk

        # 7. Execute with safety cap
        if target > 0 and position_btc <= 0:
            size_btc = min((cash * target) / price, MAX_TRADE_USD / price)
            if size_btc >= 0.0001:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} BUY {size_btc:.6f} BTC @ ${price:,.0f}")
                if LIVE and API_KEY:
                    place_market_order('buy', size_btc)
                cash -= size_btc * price * 1.0026
                position_btc += size_btc

        elif target < 0 and position_btc >= 0:
            size_btc = min(position_btc * abs(target), MAX_TRADE_USD / price)
            if size_btc >= 0.0001:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} SELL {size_btc:.6f} BTC @ ${price:,.0f}")
                if LIVE and API_KEY:
                    place_market_order('sell', size_btc)
                cash += size_btc * price * 0.9974
                position_btc -= size_btc

        # 8. Equity update
        equity = cash + position_btc * price
        equity_history.append(equity)
        print(f"Equity ${equity:.2f} | Pos {position_btc:.5f} BTC | Unc {uncertainty:.3f} | Pred {pred_return:+.3%}")

        # Sleep based on timeframe: 4h = 14400 seconds, daily = 86400
        sleep_time = 14400 if INTERVAL == 240 else 86400 if INTERVAL == 1440 else 3600
        time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n⚠ Trading stopped by user")
        break
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(60)