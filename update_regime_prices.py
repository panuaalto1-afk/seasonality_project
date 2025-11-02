#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_regime_prices.py - Päivitä regime detection -hinnat

Lataa makro-indikaattorit yfinancesta:
- SPY, QQQ, IWM (osakeindeksit)
- GLD, TLT (turvasatamat)
- HYG, LQD (luottomarkkinat)
- USO, UUP (hyödykkeet, valuutta)

Käynnistetään päivittäin klo 10:50 (ennen ml_unified_pipeline)
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
            
            # ==================== KORJAUS: auto_adjust=False ====================
            # auto_adjust=True aiheuttaa tuplaheaderin!
            df = yf.download(
                ticker, 
                start=start, 
                progress=False, 
                threads=False, 
                auto_adjust=False  # ← KORJATTU: False
            )
            # ====================================================================
            
            if df.empty:
                print(f"[FAIL] No data")
                return None
            
            # ==================== KORJAUS: Puhdista MultiIndex ====================
            # Jos yfinance palauttaa MultiIndex columns, flattenna ne
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # ======================================================================
            
            # Tarkista että tarvittavat sarakkeet on olemassa
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"[FAIL] Missing columns: {df.columns.tolist()}")
                return None
            
            # Lisää Adj Close jos puuttuu
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # ==================== KORJAUS: Reset index oikein ====================
            df = df.reset_index()
            
            # Varmista että Date on string-muodossa YYYY-MM-DD
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            else:
                print(f"[FAIL] No Date column after reset_index")
                return None
            # =====================================================================
            
            # Järjestä sarakkeet
            final_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
            print(f"[OK] {len(final_df)} rows")
            return final_df
            
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
            # ==================== KORJAUS: Varmista clean write ====================
            df.to_csv(output_file, index=False, encoding='utf-8')
            # ========================================================================
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
    
    print("="*80 + "\n")
    
    # ==================== LISÄYS: Näytä esimerkkidata ====================
    if success > 0:
        print("Example: SPY.csv first 3 rows:")
        try:
            spy_path = OUTPUT_DIR / "SPY.csv"
            if spy_path.exists():
                spy_df = pd.read_csv(spy_path, nrows=3)
                print(spy_df.to_string(index=False))
        except:
            pass
        print("="*80)
    # =====================================================================


if __name__ == "__main__":
    update_all_regime_prices()