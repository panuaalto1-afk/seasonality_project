#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_regime_prices.py - Päivitä regime detection -hinnat

Lataa makro-indikaattorit yfinancesta:
- SPY, QQQ, IWM (osakeindeksit)
- GLD, TLT (turvasatamat)
- HYG, LQD (luottomarkkinat)
- USO, UUP (hyödykkeet, valuutta)

Käynnistetään päivittäin klo 16:15 (markkinoiden sulkeuduttua)
"""

import os
import sys

# UTF-8 encoding fix for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Indikaattorit
REGIME_TICKERS = [
    'SPY',   # S&P 500
    'QQQ',   # Nasdaq-100
    'IWM',   # Russell 2000
    'GLD',   # Gold
    'TLT',   # 20+ Year Treasury
    'HYG',   # High Yield Corporate Bonds
    'LQD',   # Investment Grade Corporate Bonds
    'USO',   # Oil
    'UUP',   # US Dollar Index
    'SHY',   # 1-3 Year Treasury
    'IEF',   # 7-10 Year Treasury
    'TIP',   # TIPS
    'CPER',  # Copper
    'RINF'   # Real Return Fund
]

# Lataa 2 vuoden historia (riittää regime detectioniin)
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
OUTPUT_DIR = Path("seasonality_reports/price_cache")


def download_ticker(ticker: str, start: str, retries: int = 3) -> pd.DataFrame:
    """Lataa yksittäinen ticker yfinancesta"""
    for attempt in range(retries):
        try:
            print(f"  Downloading {ticker}... ", end='')
            df = yf.download(ticker, start=start, progress=False, threads=False, auto_adjust=True)
            
            if df.empty:
                print(f"[FAIL] No data")
                return None
            
            # Tarkista että tarvittavat sarakkeet on olemassa
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"[FAIL] Missing columns")
                return None
            
            # Lisää Adj Close jos puuttuu
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # Reset index (Date -> sarake)
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            print(f"[OK] {len(df)} rows")
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            
        except Exception as e:
            print(f"[ERROR] attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                import time
                time.sleep(2)
    
    return None


def update_all_regime_prices():
    """Päivitä kaikki regime-indikaattorit"""
    print("\n" + "="*80)
    print("REGIME PRICES UPDATE")
    print("="*80)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Start date: {START_DATE}")
    print(f"Tickers:    {len(REGIME_TICKERS)}")
    print("="*80 + "\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = []
    
    for ticker in REGIME_TICKERS:
        df = download_ticker(ticker, START_DATE)
        
        if df is not None:
            output_file = OUTPUT_DIR / f"{ticker}.csv"
            df.to_csv(output_file, index=False)
            success += 1
        else:
            failed.append(ticker)
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Success: {success}/{len(REGIME_TICKERS)}")
    
    if failed:
        print(f"Failed:  {len(failed)}")
        print(f"   Tickers: {', '.join(failed)}")
    
    print("="*80)


if __name__ == "__main__":
    update_all_regime_prices()