"""
Kraken Historical Data Downloader & Processor

Downloads and processes historical OHLCVT data from Kraken for model training.

Data Sources:
1. Kraken CSV files (manual download from their support page)
2. Kraken API (for recent data updates)

Usage:
    # First time: Download CSV from Kraken support page and place in data/ folder
    # Then run:
    python download_kraken_data.py

    # To update with recent data:
    python download_kraken_data.py --update

CSV Download Instructions:
    1. Go to: https://support.kraken.com/hc/en-us/articles/360047124832
    2. Download the OHLCVT ZIP file for your pairs
    3. Extract XBTUSD_240.csv (4hr) to data/ folder
    4. Run this script to process and update

Available timeframes in Kraken CSV (suffix in filename):
    - 1 = 1 minute
    - 5 = 5 minutes
    - 15 = 15 minutes
    - 30 = 30 minutes
    - 60 = 1 hour
    - 240 = 4 hours
    - 1440 = 1 day
    - 10080 = 1 week
    - 21600 = 15 days
"""

import pandas as pd
import numpy as np
import requests
import os
import argparse
from datetime import datetime
from pathlib import Path


def load_kraken_csv(filepath):
    """Load and parse Kraken OHLCVT CSV file.

    Kraken CSV format: timestamp, open, high, low, close, volume, trades
    - timestamp: Unix timestamp
    - open/high/low/close: Price values
    - volume: Trade volume in base currency
    - trades: Number of trades in period (count)
    """
    print(f"Loading Kraken CSV from {filepath}...")

    df = pd.read_csv(filepath, header=None, names=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'count'
    ])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    print(f"  Loaded {len(df)} records")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    return df


def fetch_kraken_api(pair='XBTUSD', interval=240, since=None):
    """Fetch OHLC data from Kraken API.

    Args:
        pair: Trading pair (e.g., 'XBTUSD', 'ETHUSD')
        interval: Timeframe in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        since: Unix timestamp to fetch data since

    Returns:
        DataFrame with OHLCV data

    Note: Kraken API returns max 720 candles per request
    """
    print(f"Fetching from Kraken API: {pair} {interval}m since {since}...")

    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        'pair': pair,
        'interval': interval
    }
    if since:
        params['since'] = since

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    if data.get('error'):
        raise ValueError(f"Kraken API error: {data['error']}")

    # Get the result key (Kraken uses different keys like XXBTZUSD)
    result_key = None
    for key in data['result']:
        if key != 'last':
            result_key = key
            break

    if not result_key:
        raise ValueError("No data in API response")

    ohlc_data = data['result'][result_key]

    df = pd.DataFrame(ohlc_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
        df[col] = df[col].astype(float)
    df['count'] = df['count'].astype(int)

    df.set_index('timestamp', inplace=True)
    df = df.drop('vwap', axis=1)  # Remove vwap to match CSV format
    df = df.sort_index()

    print(f"  Fetched {len(df)} records from API")
    if len(df) > 0:
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    return df


def update_csv_with_api(csv_path, pair='XBTUSD', interval=240):
    """Update existing CSV with recent data from API.

    Args:
        csv_path: Path to existing CSV file
        pair: Trading pair
        interval: Timeframe in minutes

    Returns:
        Updated DataFrame
    """
    # Load existing data
    df = load_kraken_csv(csv_path)

    # Get last timestamp
    last_ts = int(df.index[-1].timestamp())

    # Fetch new data
    new_df = fetch_kraken_api(pair, interval, since=last_ts)

    if len(new_df) == 0:
        print("No new data available")
        return df

    # Combine and deduplicate
    combined = pd.concat([df, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()

    print(f"\nUpdated data:")
    print(f"  Total records: {len(combined)}")
    print(f"  Date range: {combined.index[0]} to {combined.index[-1]}")
    print(f"  New records added: {len(combined) - len(df)}")

    return combined


def process_for_training(df, output_path=None):
    """Process OHLCV data for model training.

    Creates a clean CSV with columns needed for train_models.py:
    - Date (index)
    - Close price

    Args:
        df: DataFrame with OHLCV data
        output_path: Optional path to save processed CSV

    Returns:
        Processed DataFrame
    """
    processed = df[['close']].copy()
    processed.columns = ['Close']

    # Remove any NaN values
    processed = processed.dropna()

    # Calculate returns for quality check
    processed['return'] = np.log(processed['Close'] / processed['Close'].shift(1))

    # Check for data quality
    suspicious = processed[processed['return'].abs() > 0.5]
    if len(suspicious) > 0:
        print(f"\nWarning: Found {len(suspicious)} periods with >50% return (possible data errors)")
        print(suspicious.head())

    # Remove return column for output
    processed = processed[['Close']]

    if output_path:
        processed.to_csv(output_path)
        print(f"\nSaved processed data to {output_path}")
        print(f"  Records: {len(processed)}")
        print(f"  Date range: {processed.index[0]} to {processed.index[-1]}")

    return processed


def main():
    parser = argparse.ArgumentParser(description='Download and process Kraken historical data')
    parser.add_argument('--csv', type=str, help='Path to Kraken CSV file')
    parser.add_argument('--pair', type=str, default='XBTUSD', help='Trading pair (default: XBTUSD)')
    parser.add_argument('--interval', type=int, default=240, help='Timeframe in minutes (default: 240 = 4hr)')
    parser.add_argument('--update', action='store_true', help='Update existing CSV with recent API data')
    parser.add_argument('--output', type=str, default='data/btc_4h_kraken.csv', help='Output path for processed data')
    parser.add_argument('--api-only', action='store_true', help='Fetch only from API (limited to 720 candles)')

    args = parser.parse_args()

    # Create data directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else 'data', exist_ok=True)

    if args.api_only:
        # Fetch from API only (limited data)
        print(f"\n{'='*60}")
        print("FETCHING FROM KRAKEN API ONLY")
        print("Note: API returns max 720 candles (~120 days for 4hr)")
        print("For full history, download CSV from Kraken support page")
        print(f"{'='*60}\n")

        df = fetch_kraken_api(args.pair, args.interval)
        process_for_training(df, args.output)

    elif args.csv:
        # Load from CSV file
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            print("\nTo get Kraken historical data:")
            print("1. Go to: https://support.kraken.com/hc/en-us/articles/360047124832")
            print("2. Download the OHLCVT ZIP file")
            print("3. Extract XBTUSD_240.csv (for 4hr data)")
            print("4. Run: python download_kraken_data.py --csv path/to/XBTUSD_240.csv")
            return

        df = load_kraken_csv(args.csv)

        if args.update:
            df = update_csv_with_api(args.csv, args.pair, args.interval)

        process_for_training(df, args.output)

    else:
        # Check for default CSV location
        default_csv = 'data/XBTUSD_240.csv'
        if os.path.exists(default_csv):
            print(f"Found default CSV at {default_csv}")
            df = load_kraken_csv(default_csv)

            if args.update:
                df = update_csv_with_api(default_csv, args.pair, args.interval)

            process_for_training(df, args.output)
        else:
            print("\n" + "="*60)
            print("KRAKEN DATA DOWNLOADER")
            print("="*60)
            print("\nNo CSV file specified and no default found.")
            print("\nTo get Kraken historical data (recommended):")
            print("  1. Go to: https://support.kraken.com/hc/en-us/articles/360047124832")
            print("  2. Download the OHLCVT ZIP file for BTC")
            print("  3. Extract XBTUSD_240.csv to data/ folder")
            print("  4. Run: python download_kraken_data.py --csv data/XBTUSD_240.csv")
            print("\nOr fetch limited data from API:")
            print("  python download_kraken_data.py --api-only")
            print("\nOptions:")
            print("  --csv PATH       Path to Kraken CSV file")
            print("  --update         Update CSV with recent API data")
            print("  --pair PAIR      Trading pair (default: XBTUSD)")
            print("  --interval MIN   Timeframe in minutes (default: 240)")
            print("  --output PATH    Output path (default: data/btc_4h_kraken.csv)")
            print("  --api-only       Fetch from API only (limited to 720 candles)")


if __name__ == '__main__':
    main()
