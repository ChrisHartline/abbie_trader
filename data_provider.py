"""
Data Provider Module - Unified data fetching for the trading bot

Supports multiple backends:
- OpenBB (primary) - aggregates 100+ sources
- yfinance (fallback)
- Alpha Vantage (optional, requires API key)

Usage:
    from data_provider import get_stock_data, get_vix_data

    df = get_stock_data("NVDA", start="2020-01-01")
    vix = get_vix_data(start="2020-01-01", index=df.index)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Try to import data providers
OPENBB_AVAILABLE = False
YFINANCE_AVAILABLE = False

try:
    from openbb import obb
    OPENBB_AVAILABLE = True

    # Authenticate with OpenBB if API key is available
    openbb_api_key = os.environ.get("OPENBB_API_KEY", "")
    if openbb_api_key:
        try:
            obb.account.login(pat=openbb_api_key)
            print("OpenBB authenticated successfully")
        except Exception as e:
            print(f"OpenBB authentication failed: {e}")
            print("Continuing without authentication (limited features)")
    else:
        print("OpenBB available (no API key - using free tier)")
except ImportError:
    pass

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    pass

# Configuration
DEFAULT_START_DATE = "2018-01-01"
PREFERRED_PROVIDER = os.environ.get("DATA_PROVIDER", "openbb")  # openbb, yfinance, alphavantage


def get_stock_data(ticker: str, start: str = None, end: str = None,
                   provider: str = None) -> pd.DataFrame:
    """
    Fetch historical stock data.

    Args:
        ticker: Stock symbol (e.g., "NVDA", "TSLA")
        start: Start date (YYYY-MM-DD), defaults to 2018-01-01
        end: End date (YYYY-MM-DD), defaults to today
        provider: Force specific provider ("openbb", "yfinance")

    Returns:
        DataFrame with columns: Close, return
    """
    start = start or DEFAULT_START_DATE
    end = end or datetime.now().strftime("%Y-%m-%d")
    provider = provider or PREFERRED_PROVIDER

    df = None
    error_msgs = []

    # Try OpenBB first
    if provider == "openbb" or (df is None and OPENBB_AVAILABLE):
        try:
            df = _fetch_openbb(ticker, start, end)
            if df is not None and len(df) > 0:
                print(f"  [{ticker}] Fetched {len(df)} days via OpenBB")
        except Exception as e:
            error_msgs.append(f"OpenBB: {e}")

    # Fallback to yfinance
    if df is None and YFINANCE_AVAILABLE:
        try:
            df = _fetch_yfinance(ticker, start, end)
            if df is not None and len(df) > 0:
                print(f"  [{ticker}] Fetched {len(df)} days via yfinance")
        except Exception as e:
            error_msgs.append(f"yfinance: {e}")

    # Check if we got data
    if df is None or len(df) == 0:
        raise ValueError(f"Failed to fetch data for {ticker}. Errors: {'; '.join(error_msgs)}")

    # Standardize output
    df = _standardize_dataframe(df)

    # Add returns
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()

    return df


def _fetch_openbb(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch data using OpenBB"""
    if not OPENBB_AVAILABLE:
        return None

    output = obb.equity.price.historical(
        symbol=ticker,
        start_date=start,
        end_date=end,
        provider="yfinance"  # Use yfinance as default provider within OpenBB
    )

    df = output.to_dataframe()

    if df is None or len(df) == 0:
        return None

    return df


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch data using yfinance directly"""
    if not YFINANCE_AVAILABLE:
        return None

    df = yf.download(ticker, start=start, end=end, progress=False)

    if df is None or len(df) == 0:
        return None

    return df


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame to have consistent column names"""
    # Handle MultiIndex columns (yfinance sometimes returns this)
    if isinstance(df.columns, pd.MultiIndex):
        # Try to get Close column
        if 'Close' in df.columns.get_level_values(0):
            df = df['Close']
            if isinstance(df, pd.DataFrame):
                df = df.iloc[:, 0]
            df = pd.DataFrame(df, columns=['Close'])
        else:
            df.columns = df.columns.get_level_values(0)

    # Rename columns to standard format
    col_mapping = {
        'close': 'Close',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume',
        'adj_close': 'Adj Close',
        'adjusted_close': 'Adj Close',
    }

    df.columns = [col_mapping.get(c.lower(), c) for c in df.columns]

    # Ensure Close column exists
    if 'Close' not in df.columns:
        close_cols = [c for c in df.columns if 'close' in c.lower()]
        if close_cols:
            df['Close'] = df[close_cols[0]]
        else:
            raise ValueError("No 'Close' column found in data")

    return df


def get_vix_data(start: str = None, end: str = None,
                 index: pd.DatetimeIndex = None) -> pd.Series:
    """
    Fetch VIX data as sentiment proxy for stocks.

    Args:
        start: Start date
        end: End date
        index: Optional DatetimeIndex to align VIX data to

    Returns:
        Series of normalized VIX values [-1, 1]
    """
    start = start or DEFAULT_START_DATE
    end = end or datetime.now().strftime("%Y-%m-%d")

    vix = None

    # Try OpenBB
    if OPENBB_AVAILABLE:
        try:
            output = obb.equity.price.historical(
                symbol="^VIX",
                start_date=start,
                end_date=end,
                provider="yfinance"
            )
            vix_df = output.to_dataframe()
            if vix_df is not None and len(vix_df) > 0:
                vix = vix_df['close'] if 'close' in vix_df.columns else vix_df['Close']
        except Exception:
            pass

    # Fallback to yfinance
    if vix is None and YFINANCE_AVAILABLE:
        try:
            vix_df = yf.download("^VIX", start=start, end=end, progress=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix = vix_df['Close'].iloc[:, 0] if isinstance(vix_df['Close'], pd.DataFrame) else vix_df['Close']
            else:
                vix = vix_df['Close']
        except Exception:
            pass

    # Generate synthetic VIX if all else fails
    if vix is None:
        print("  Warning: Could not fetch VIX data, using synthetic values")
        if index is not None:
            np.random.seed(42)
            vix = pd.Series(20 + np.cumsum(np.random.normal(0, 0.5, len(index))), index=index)
            vix = vix.clip(10, 50)
        else:
            raise ValueError("Cannot generate synthetic VIX without index")

    # Reindex to match stock data if provided
    if index is not None:
        vix = vix.reindex(index, method='ffill')

    # Normalize VIX to [-1, 1] range (typical VIX: 10-40, centered at 20)
    vix_normalized = (vix - 20) / 20
    vix_normalized = vix_normalized.clip(-1, 1)

    return vix_normalized


def get_available_providers() -> dict:
    """Return dictionary of available data providers"""
    return {
        'openbb': OPENBB_AVAILABLE,
        'yfinance': YFINANCE_AVAILABLE,
    }


def test_providers():
    """Test all available data providers"""
    print("=" * 60)
    print("DATA PROVIDER TEST")
    print("=" * 60)

    providers = get_available_providers()
    print(f"\nAvailable providers: {providers}")

    test_ticker = "AAPL"
    test_start = "2024-01-01"

    for provider, available in providers.items():
        if not available:
            print(f"\n{provider}: Not installed")
            continue

        print(f"\nTesting {provider}...")
        try:
            df = get_stock_data(test_ticker, start=test_start, provider=provider)
            print(f"  Success! Got {len(df)} rows")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"  Failed: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_providers()
