#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_exit_watchlist.py
======================
Luo exit watchlist nykyisille positioille.

Features:
- Trailing stop ATR-pohjaisesti
- Take profit -tasot
- P&L tracking
- Regime-aware exits
- Automaattinen arkistointi (timestamped CSV:t)
- KORJATTU: MultiIndex DataFrame handling

Usage:
    python make_exit_watchlist.py
    python make_exit_watchlist.py --stop_mult 2.0
    python make_exit_watchlist.py --price_cache_dir "path/to/cache"
"""

import argparse
import os
import sys
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

# Import logging system
try:
    from trades_logger import TradeLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    print("[WARN] trades_logger.py ei löydy. Logging pois käytöstä.")

# ======================== ARGUMENT PARSING ========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Luo exit watchlist nykyisille positioille",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--stop_mult", type=float, default=1.5,
                   help="Stop loss etäisyys (kerrottuna ATR:llä)")
    p.add_argument("--tp_mult", type=float, default=3.0,
                   help="Take profit etäisyys (kerrottuna ATR:llä)")
    p.add_argument("--atr_period", type=int, default=14,
                   help="ATR laskenta-periodi")
    p.add_argument("--stats_file", type=str, default=None,
                   help="Polku tilastotiedostoon (jos None, käytetään estimoituja)")
    p.add_argument("--price_cache_dir", type=str, default=None,
                   help="Price cache -kansio (jos None, auto-detect)")
    p.add_argument("--output", type=str, default="seasonality_reports/exit_watchlist.csv",
                   help="Output CSV path")
    return p.parse_args()

# ======================== PRICE DATA ========================

def find_price_cache_dir() -> Optional[str]:
    """Etsi viimeisin price cache -kansio"""
    base = "seasonality_reports/runs"
    
    if not os.path.exists(base):
        return None
    
    # Etsi kaikki run-kansiot
    runs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    
    if not runs:
        return None
    
    # Järjestä päivämäärän mukaan (uusin ensin)
    runs.sort(reverse=True)
    
    # Etsi ensimmäinen jossa on price_cache
    for run in runs:
        cache_dir = os.path.join(base, run, "price_cache")
        if os.path.exists(cache_dir):
            return cache_dir
    
    return None

def load_price_from_cache(ticker: str, cache_dir: str) -> Optional[pd.DataFrame]:
    """Lataa hinta-data cachesta"""
    cache_file = os.path.join(cache_dir, f"{ticker}.csv")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        df = pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
        
        # Flatten MultiIndex if needed (cache may have MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        return df
    except Exception as e:
        print(f"[WARN] Virhe luettaessa {ticker} cachesta: {e}")
        return None

def download_price_data(ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """Lataa hinta-data yfinancesta"""
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # Välttää FutureWarning
        )
        
        if df.empty:
            return None
        
        # ==================== KORJAUS: Flatten MultiIndex ====================
        # yfinance palauttaa MultiIndex DataFramen (ticker, column)
        # Flatten se yksinkertaiseksi DataFrameksi
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # =====================================================================
        
        # Varmista että sarakkeet ovat oikein
        if 'Close' not in df.columns:
            return None
        
        return df
    except Exception as e:
        print(f"[WARN] Virhe ladattaessa {ticker}: {e}")
        return None

def get_price_data(ticker: str, cache_dir: Optional[str] = None, 
                   lookback_days: int = 90) -> Optional[pd.DataFrame]:
    """Hae hinta-data (cache tai download)"""
    
    # Yritä ensin cachesta
    if cache_dir and os.path.exists(cache_dir):
        df = load_price_from_cache(ticker, cache_dir)
        if df is not None and not df.empty:
            # Tarkista että data on riittävän tuoretta
            if df.index[-1].date() >= (date.today() - timedelta(days=7)):
                return df
    
    # Jos cache ei toimi, lataa yfinancesta
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    
    df = download_price_data(ticker, start_date, end_date)
    
    # ==================== KORJAUS: Varmista että MultiIndex on flattened ====================
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # ========================================================================================
    
    return df

# ======================== TECHNICAL INDICATORS ========================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Laske ATR (Average True Range)"""
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR (EMA of TR)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr

def calculate_stops(df: pd.DataFrame, entry_price: float, 
                    stop_mult: float = 1.5, tp_mult: float = 3.0,
                    atr_period: int = 14) -> Dict:
    """Laske stop loss ja take profit tasot"""
    
    if df.empty:
        return {
            'current_price': None,
            'atr': None,
            'stop_loss': None,
            'take_profit': None,
            'stop_distance_pct': None,
            'tp_distance_pct': None
        }
    
    # Laske ATR
    atr = calculate_atr(df, period=atr_period)
    current_atr = atr.iloc[-1]
    current_price = df['Close'].iloc[-1]
    
    # ==================== KORJAUS: Varmista scalar values ====================
    # Muunna pandas scalar -> Python float
    if isinstance(current_atr, pd.Series):
        current_atr = float(current_atr.iloc[0])
    else:
        current_atr = float(current_atr)
    
    if isinstance(current_price, pd.Series):
        current_price = float(current_price.iloc[0])
    else:
        current_price = float(current_price)
    # ========================================================================
    
    # Trailing stop (entry price - ATR * multiplier)
    stop_loss = entry_price - (current_atr * stop_mult)
    
    # Take profit (entry price + ATR * multiplier)
    take_profit = entry_price + (current_atr * tp_mult)
    
    # Prosenttietäisyydet
    stop_distance_pct = ((stop_loss - current_price) / current_price) * 100
    tp_distance_pct = ((take_profit - current_price) / current_price) * 100
    
    return {
        'current_price': current_price,
        'atr': current_atr,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'stop_distance_pct': stop_distance_pct,
        'tp_distance_pct': tp_distance_pct
    }

# ======================== P&L CALCULATIONS ========================

def calculate_pnl(entry_price: float, current_price: float, 
                  quantity: int) -> Dict:
    """Laske realisoitumaton P&L"""
    
    if current_price is None or entry_price is None:
        return {
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0
        }
    
    # Dollar P&L
    unrealized_pnl = (current_price - entry_price) * quantity
    
    # Prosentti P&L
    unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
    
    return {
        'unrealized_pnl': unrealized_pnl,
        'unrealized_pnl_pct': unrealized_pnl_pct
    }

# ======================== PORTFOLIO STATE ========================

def load_portfolio_state(project_root: str = ".") -> Dict:
    """Lataa portfolio state"""
    path = os.path.join(project_root, "seasonality_reports", "portfolio_state.json")
    
    if not os.path.exists(path):
        print(f"[WARN] portfolio_state.json ei löydy: {path}")
        return {"positions": {}, "cash": 0.0}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
            # Varmista että positions on dict
            if not isinstance(state.get('positions'), dict):
                print(f"[WARN] portfolio_state.json positions ei ole dict")
                state['positions'] = {}
            
            return state
    except Exception as e:
        print(f"[ERROR] Virhe luettaessa portfolio_state.json: {e}")
        return {"positions": {}, "cash": 0.0}

# ======================== EXIT SIGNALS ========================

def check_exit_signals(ticker: str, position: Dict, price_data: pd.DataFrame,
                       stops: Dict, args) -> Dict:
    """Tarkista exit-signaalit"""
    
    signals = {
        'stop_loss_hit': False,
        'take_profit_hit': False,
        'recommendation': 'HOLD'
    }
    
    current_price = stops.get('current_price')
    stop_loss = stops.get('stop_loss')
    take_profit = stops.get('take_profit')
    
    if current_price is None or stop_loss is None:
        return signals
    
    # ==================== KORJAUS: Varmista scalar comparison ====================
    # Muunna pandas scalar -> Python float (jos tarvitaan)
    if isinstance(current_price, pd.Series):
        current_price = float(current_price.iloc[0])
    if isinstance(stop_loss, pd.Series):
        stop_loss = float(stop_loss.iloc[0])
    if isinstance(take_profit, pd.Series):
        take_profit = float(take_profit.iloc[0])
    # ===========================================================================
    
    # Stop loss
    if current_price <= stop_loss:
        signals['stop_loss_hit'] = True
        signals['recommendation'] = 'SELL (Stop Loss)'
    
    # Take profit
    if take_profit and current_price >= take_profit:
        signals['take_profit_hit'] = True
        signals['recommendation'] = 'SELL (Take Profit)'
    
    # Muut signaalit (voidaan lisätä myöhemmin)
    # - Regime change
    # - Seasonality window closed
    # - Technical breakdown
    
    return signals

# ======================== MAIN LOGIC ========================

def build_exit_watchlist(args):
    """Rakenna exit watchlist nykyisille positioille"""
    
    print("\n" + "="*80)
    print("🎯 EXIT WATCHLIST BUILDER")
    print("="*80)
    print(f"Päivämäärä:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stop mult:    {args.stop_mult} × ATR")
    print(f"Stats file:   {args.stats_file if args.stats_file else 'Estimoidut tilastot'}")
    print(f"Price cache:  {args.price_cache_dir if args.price_cache_dir else 'Auto-detect'}")
    print("="*80 + "\n")
    
    # Etsi price cache
    if args.price_cache_dir is None:
        cache_dir = find_price_cache_dir()
        if cache_dir:
            print(f"[INFO] Price cache löytyi: {cache_dir}")
        else:
            print(f"[WARN] Price cachea ei löytynyt, ladataan yfinancesta")
    else:
        cache_dir = args.price_cache_dir
    
    # Lataa portfolio state
    print(f"[INFO] Ladataan portfolio state...")
    portfolio = load_portfolio_state()
    positions = portfolio.get('positions', {})
    
    if not positions or not isinstance(positions, dict):
        print(f"[INFO] Ei avoimia positioita.")
        # Tallenna tyhjä exit watchlist
        empty_df = pd.DataFrame()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        empty_df.to_csv(args.output, index=False)
        print(f"[OK] Tyhjä exit watchlist tallennettu: {args.output}\n")
        return
    
    print(f"[INFO] Löytyi {len(positions)} positiota\n")
    
    # Käsittele jokainen positio
    watchlist_data = []
    
    for ticker, pos in positions.items():
        print(f"[{ticker}] Käsitellään...")
        
        # Hae position tiedot
        entry_price = pos.get('entry_price', 0.0)
        quantity = pos.get('quantity', 0)
        entry_date = pos.get('entry_date', 'Unknown')
        regime_at_entry = pos.get('regime_at_entry', 'Unknown')
        
        # Hae price data
        price_data = get_price_data(ticker, cache_dir=cache_dir)
        
        if price_data is None or price_data.empty:
            print(f"[{ticker}] ⚠️  Ei price dataa, skipataan")
            continue
        
        # Laske stops
        stops = calculate_stops(
            price_data,
            entry_price,
            stop_mult=args.stop_mult,
            tp_mult=args.tp_mult,
            atr_period=args.atr_period
        )
        
        current_price = stops.get('current_price')
        
        if current_price is None:
            print(f"[{ticker}] ⚠️  Ei current pricea, skipataan")
            continue
        
        # Laske P&L
        pnl = calculate_pnl(entry_price, current_price, quantity)
        
        # Tarkista exit-signaalit
        signals = check_exit_signals(ticker, pos, price_data, stops, args)
        
        # Laske päivien määrä
        try:
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d').date()
            days_held = (date.today() - entry_dt).days
        except:
            days_held = 0
        
        # Rakenna watchlist entry
        watchlist_entry = {
            'Ticker': ticker,
            'EntryDate': entry_date,
            'DaysHeld': days_held,
            'EntryPrice': entry_price,
            'CurrentPrice': current_price,
            'Quantity': quantity,
            'ATR': stops.get('atr', 0.0),
            'StopLoss': stops.get('stop_loss', 0.0),
            'TakeProfit': stops.get('take_profit', 0.0),
            'StopDistance%': stops.get('stop_distance_pct', 0.0),
            'TPDistance%': stops.get('tp_distance_pct', 0.0),
            'UnrealizedPnL': pnl['unrealized_pnl'],
            'UnrealizedPnL%': pnl['unrealized_pnl_pct'],
            'Recommendation': signals['recommendation'],
            'RegimeAtEntry': regime_at_entry
        }
        
        watchlist_data.append(watchlist_entry)
        
        # Print summary
        print(f"[{ticker}] ✅ OK")
        print(f"         Entry: ${entry_price:.2f} → Current: ${current_price:.2f} ({pnl['unrealized_pnl_pct']:+.2f}%)")
        print(f"         Stop: ${stops.get('stop_loss', 0):.2f} ({stops.get('stop_distance_pct', 0):.1f}%)")
        print(f"         TP:   ${stops.get('take_profit', 0):.2f} ({stops.get('tp_distance_pct', 0):.1f}%)")
        print(f"         → {signals['recommendation']}")
        print()
    
    # Luo DataFrame
    final_df = pd.DataFrame(watchlist_data)
    
    if final_df.empty:
        print(f"[WARN] Ei dataa exit watchlistille")
        empty_df = pd.DataFrame()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        empty_df.to_csv(args.output, index=False)
        print(f"[OK] Tyhjä exit watchlist tallennettu: {args.output}\n")
        return
    
    # Järjestä recommendation ja unrealized P&L mukaan
    final_df = final_df.sort_values(
        by=['Recommendation', 'UnrealizedPnL%'],
        ascending=[True, False]
    )
    
    # ==================== ARKISTOINTI ====================
    # Tallenna timestamped versio arkistoon
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join("seasonality_reports", "exit_watchlist_archive")
    os.makedirs(archive_dir, exist_ok=True)
    
    archive_path = os.path.join(archive_dir, f"exit_watchlist_{timestamp}.csv")
    final_df.to_csv(archive_path, index=False)
    print(f"[ARCHIVE] ✅ Saved: {archive_path}")
    # ====================================================
    
    # Tallenna current exit watchlist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"[OK] ✅ Saved: {args.output}")
    
    # Yhteenveto
    print("\n" + "="*80)
    print("📊 EXIT WATCHLIST YHTEENVETO")
    print("="*80)
    print(f"Positioita yhteensä:     {len(final_df)}")
    print(f"SELL suosituksia:        {len(final_df[final_df['Recommendation'] != 'HOLD'])}")
    print(f"HOLD suosituksia:        {len(final_df[final_df['Recommendation'] == 'HOLD'])}")
    print(f"\nKeskimääräinen P&L:      {final_df['UnrealizedPnL%'].mean():.2f}%")
    print(f"Paras P&L:               {final_df['UnrealizedPnL%'].max():.2f}%")
    print(f"Huonoin P&L:             {final_df['UnrealizedPnL%'].min():.2f}%")
    print(f"\nArkistoitu:              {archive_path}")
    print("="*80 + "\n")

def main():
    args = parse_args()
    
    try:
        build_exit_watchlist(args)
    except KeyboardInterrupt:
        print("\n[INFO] Keskeytetty käyttäjän toimesta")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Virhe: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()