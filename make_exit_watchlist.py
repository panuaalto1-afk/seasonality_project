#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_exit_watchlist.py - Exit watchlist todennäköisyyspohjaisilla TP-tasoilla

Tallentaa:
1. runs/LATEST_RUN/actions/YYYYMMDD/exit_watchlist_YYYYMMDD.csv (arkisto)
2. seasonality_reports/exit_watchlists/latest_exit_watchlist.csv (viimeisin)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ==================== ESTIMOIDUT TILASTOT ====================
ESTIMATED_SETUP_STATS = {
    'ML_Momentum_Strong': {
        'win_rate': 0.68,
        'avg_win_pct': 8.2,
        'avg_loss_pct': -3.1,
        'profit_factor': 2.1,
        'tp1_hit_rate': 0.45,
        'tp2_hit_rate': 0.18,
        'tp3_hit_rate': 0.05,
        'median_hold_days': 6
    },
    'ML_Momentum_Moderate': {
        'win_rate': 0.54,
        'avg_win_pct': 5.2,
        'avg_loss_pct': -2.8,
        'profit_factor': 1.2,
        'tp1_hit_rate': 0.35,
        'tp2_hit_rate': 0.12,
        'tp3_hit_rate': 0.03,
        'median_hold_days': 7
    },
    'ML_Seasonality_Combo': {
        'win_rate': 0.73,
        'avg_win_pct': 9.5,
        'avg_loss_pct': -3.0,
        'profit_factor': 2.8,
        'tp1_hit_rate': 0.47,
        'tp2_hit_rate': 0.19,
        'tp3_hit_rate': 0.08,
        'median_hold_days': 8
    },
    'Optio_Breakout': {
        'win_rate': 0.65,
        'avg_win_pct': 11.2,
        'avg_loss_pct': -3.5,
        'profit_factor': 1.9,
        'tp1_hit_rate': 0.39,
        'tp2_hit_rate': 0.17,
        'tp3_hit_rate': 0.09,
        'median_hold_days': 5
    },
    'Seasonality_Pure': {
        'win_rate': 0.61,
        'avg_win_pct': 6.5,
        'avg_loss_pct': -2.5,
        'profit_factor': 1.6,
        'tp1_hit_rate': 0.37,
        'tp2_hit_rate': 0.14,
        'tp3_hit_rate': 0.10,
        'median_hold_days': 11
    },
    'Default': {
        'win_rate': 0.50,
        'avg_win_pct': 4.5,
        'avg_loss_pct': -3.0,
        'profit_factor': 1.0,
        'tp1_hit_rate': 0.30,
        'tp2_hit_rate': 0.10,
        'tp3_hit_rate': 0.02,
        'median_hold_days': 10
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Exit watchlist builder")
    parser.add_argument("--portfolio", type=str, help="Portfolio CSV file path")
    parser.add_argument("--price_cache", type=str, default="seasonality_reports/price_cache", 
                        help="Price cache directory")
    parser.add_argument("--stop_mult", type=float, default=1.5, help="Stop loss multiplier")
    parser.add_argument("--stats_file", type=str, default=None, help="Historical stats CSV")
    parser.add_argument("--runs_dir", type=str, default="seasonality_reports/runs")
    return parser.parse_args()


class PriceDataLoader:
    """Lataa hintadataa price_cache:sta"""
    
    def __init__(self, price_cache_dir: Path):
        self.price_cache_dir = Path(price_cache_dir)
        if not self.price_cache_dir.exists():
            print(f"⚠️  Price cache ei löydy: {self.price_cache_dir}")
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        price_file = self.price_cache_dir / f"{ticker}.csv"
        if not price_file.exists():
            print(f"⚠️  Hintadata puuttuu: {ticker}")
            return None
        try:
            df = pd.read_csv(price_file, parse_dates=['Date'])
            
            # Muunna hintasarakkeet numeerisiksi
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Poista rivit joissa on NaN-arvoja
            df = df.dropna(subset=['High', 'Low', 'Close'])
            
            return df.sort_values('Date')
        except Exception as e:
            print(f"⚠️  Virhe ladattaessa {ticker}: {e}")
            return None
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 20) -> float:
        """Laske Average True Range"""
        if len(df) < period:
            return 0.0
        
        try:
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            close = df['Close'].astype(float).shift(1)
            
            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
        except Exception as e:
            print(f"⚠️  ATR-laskentavirhe: {e}")
            return 0.0


def find_latest_portfolio(runs_dir: Path) -> Optional[Path]:
    """Etsi viimeisin portfolio runs-kansiosta"""
    if not runs_dir.exists():
        return None
    
    portfolios = []
    for run_dir in runs_dir.glob("*"):
        if not run_dir.is_dir():
            continue
        actions_dir = run_dir / "actions"
        if not actions_dir.exists():
            continue
        for date_dir in actions_dir.glob("*"):
            if not date_dir.is_dir():
                continue
            portfolio_file = date_dir / "portfolio_after_sim.csv"
            if portfolio_file.exists():
                portfolios.append((portfolio_file, portfolio_file.stat().st_mtime))
    
    if not portfolios:
        return None
    
    portfolios.sort(key=lambda x: x[1], reverse=True)
    return portfolios[0][0]


def find_latest_price_cache(runs_dir: Path) -> Optional[Path]:
    """Etsi viimeisin price_cache runs-kansiosta"""
    if not runs_dir.exists():
        return None
    
    caches = []
    for run_dir in runs_dir.glob("*"):
        if not run_dir.is_dir():
            continue
        cache_dir = run_dir / "price_cache"
        if cache_dir.exists():
            caches.append((cache_dir, cache_dir.stat().st_mtime))
    
    if not caches:
        return None
    
    caches.sort(key=lambda x: x[1], reverse=True)
    return caches[0][0]


def load_setup_statistics(stats_file: Optional[str]) -> Dict:
    """Lataa tilastot"""
    if stats_file and Path(stats_file).exists():
        print(f"📊 Ladataan tilastot: {stats_file}")
        df = pd.read_csv(stats_file)
        stats = {}
        for _, row in df.iterrows():
            stats[row['setup_type']] = {
                'win_rate': row['win_rate'],
                'tp1_hit_rate': row['tp1_hit_rate'],
                'tp2_hit_rate': row['tp2_hit_rate'],
                'tp3_hit_rate': row['tp3_hit_rate'],
                'profit_factor': row['profit_factor']
            }
        return stats
    else:
        print("📊 Käytetään estimoituja tilastoja")
        return ESTIMATED_SETUP_STATS


def calculate_trailing_stop(entry: float, current: float, original_stop: float, days: int) -> float:
    """
    Laske trailing stop hinnan noustessa:
    - Profit > 1R: nosta stop breakeven-tasolle
    - Profit > 2R: nosta stop +0.5R voitolle
    - Profit > 3R: nosta stop +1.5R voitolle
    """
    risk = entry - original_stop
    if risk <= 0:
        return original_stop
    profit_in_R = (current - entry) / risk
    
    if profit_in_R > 3.0:
        new_stop = entry + (1.5 * risk)
    elif profit_in_R > 2.0:
        new_stop = entry + (0.5 * risk)
    elif profit_in_R > 1.0:
        new_stop = entry
    else:
        new_stop = original_stop
    
    return max(original_stop, new_stop)


def build_exit_watchlist(args):
    """Rakentaa exit-watchlistin"""
    
    # Määritä price_cache
    if args.price_cache == "seasonality_reports/price_cache":
        price_cache_dir = Path(args.price_cache)
        if not price_cache_dir.exists():
            print("📂 Price cache ei löydy juuresta, etsitään runs-kansiosta...")
            price_cache_dir = find_latest_price_cache(Path(args.runs_dir))
            if price_cache_dir:
                print(f"📂 Löytyi: {price_cache_dir}")
            else:
                print("❌ Price cache ei löydy")
                return
    else:
        price_cache_dir = Path(args.price_cache)
    
    # Määritä portfolio
    if args.portfolio:
        portfolio_path = Path(args.portfolio)
    else:
        print("📂 Etsitään viimeisintä portfoliota...")
        portfolio_path = find_latest_portfolio(Path(args.runs_dir))
        if portfolio_path:
            print(f"📂 Löytyi: {portfolio_path}")
        else:
            print("❌ Portfolio ei löydy")
            return
    
    if not portfolio_path.exists():
        print(f"❌ Portfolio ei löydy: {portfolio_path}")
        return
    
    print(f"📂 Luetaan portfolio: {portfolio_path}")
    portfolio = pd.read_csv(portfolio_path)
    
    if portfolio.empty:
        print("⚠️  Portfolio tyhjä")
        exit_df = pd.DataFrame([], columns=[
            'Ticker', 'Entry', 'Current', 'Stop', 'TP1', 'TP2', 'TP3',
            'Setup_Type', 'Win_Rate', 'Profit_Factor', 'Days_Since_Entry',
            'Trail_Stop', 'Note'
        ])
        
        # Tallenna tyhjä tiedosto
        watchlist_dir = Path("seasonality_reports/exit_watchlists")
        watchlist_dir.mkdir(parents=True, exist_ok=True)
        latest_output = watchlist_dir / "latest_exit_watchlist.csv"
        exit_df.to_csv(latest_output, index=False)
        print(f"📄 Kirjoitettu tyhjä: {latest_output}")
        return
    
    print(f"📊 Portfolio sisältää {len(portfolio)} riviä")
    
    setup_stats = load_setup_statistics(args.stats_file)
    price_loader = PriceDataLoader(price_cache_dir)
    exit_data = []
    
    for idx, pos in portfolio.iterrows():
        ticker = pos.get('Ticker') or pos.get('ticker') or pos.get('Symbol')
        entry = pos.get('Entry')
        setup_type = pos.get('Setup_Type', 'Default')
        entry_date = pos.get('Entry_Date')
        
        print(f"  Käsitellään: {ticker}, Entry={entry}, Setup={setup_type}")
        
        if not ticker or pd.isna(entry):
            print(f"    → Ohitetaan (puuttuva tieto)")
            continue
        
        df = price_loader.load_ticker(ticker)
        if df is None or df.empty:
            print(f"    → Ohitetaan (ei hintadataa)")
            continue
        
        current = float(df['Close'].iloc[-1])
        atr = price_loader.calculate_atr(df)
        
        print(f"    → Current={current:.2f}, ATR={atr:.2f}")
        
        if atr == 0:
            print(f"    → Ohitetaan (ATR=0)")
            continue
        
        risk = args.stop_mult * atr
        stop = entry - risk
        tp1 = entry + (1.5 * risk)
        tp2 = entry + (3.0 * risk)
        tp3 = entry + (5.0 * risk)
        
        stats = setup_stats.get(setup_type, setup_stats['Default'])
        
        # Päivät entry:stä
        days_held = 0
        if entry_date and not pd.isna(entry_date):
            try:
                entry_dt = pd.to_datetime(entry_date)
                days_held = (datetime.now() - entry_dt).days
            except:
                pass
        
        trail_stop = calculate_trailing_stop(entry, current, stop, days_held)
        
        profit_pct = ((current - entry) / entry) * 100
        note = f"{'Voitolla' if profit_pct > 0 else 'Tappiolla'} {profit_pct:.1f}%"
        if stats['win_rate'] > 0.65:
            note += " | Korkea win rate"
        
        exit_data.append({
            'Ticker': ticker,
            'Entry': round(entry, 2),
            'Current': round(current, 2),
            'Stop': round(stop, 2),
            'TP1': round(tp1, 2),
            'TP2': round(tp2, 2),
            'TP3': round(tp3, 2),
            'Setup_Type': setup_type,
            'Win_Rate': round(stats['win_rate'], 3),
            'Profit_Factor': round(stats['profit_factor'], 2),
            'Days_Since_Entry': days_held,
            'Trail_Stop': round(trail_stop, 2),
            'Note': note
        })
        
        print(f"    ✅ Lisätty watchlistiin")
    
    exit_df = pd.DataFrame(exit_data)
    
    # ==================== TALLENNUSLOGIIKKA ====================
    today_str = datetime.now().strftime("%Y%m%d")  # 20251101
    
    # 1. Tallenna päiväkohtaisesti runs/actions-kansioon (ARKISTO)
    runs_dir = Path(args.runs_dir)
    if runs_dir.exists():
        run_dirs = sorted([d for d in runs_dir.glob("*") if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        if run_dirs:
            latest_run = run_dirs[0]
            actions_dir = latest_run / "actions"
            actions_dir.mkdir(parents=True, exist_ok=True)
            
            # Luo tämän päivän kansio
            today_dir = actions_dir / today_str
            today_dir.mkdir(parents=True, exist_ok=True)
            
            # Tallenna päiväkohtaisella nimellä
            dated_filename = f"exit_watchlist_{today_str}.csv"
            actions_output = today_dir / dated_filename
            
            try:
                exit_df.to_csv(actions_output, index=False)
                print(f"✅ Arkisto: {actions_output}")
            except Exception as e:
                print(f"⚠️  Arkistointivirhe: {e}")
    
    # 2. Tallenna viimeisin versio exit_watchlists-kansioon
    watchlist_dir = Path("seasonality_reports/exit_watchlists")
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    
    latest_output = watchlist_dir / "latest_exit_watchlist.csv"
    try:
        exit_df.to_csv(latest_output, index=False)
        print(f"✅ Viimeisin: {latest_output}")
    except Exception as e:
        print(f"⚠️  Tallennusvirhe: {e}")
    
    print(f"\n📊 {len(exit_df)} positiota watchlistissä\n")
    
    if not exit_df.empty:
        print("="*80)
        print("📈 EXIT WATCHLIST YHTEENVETO:")
        print("="*80)
        print(exit_df.to_string(index=False))
        print("="*80)
        print(f"\n💡 Trailing Stop -logiikka:")
        print(f"   Voitto > 1R → Stop breakeven-tasolle")
        print(f"   Voitto > 2R → Stop +0.5R voitolle")
        print(f"   Voitto > 3R → Stop +1.5R voitolle")
        print("="*80)


def main():
    args = parse_args()
    
    print("="*80)
    print("🎯 EXIT WATCHLIST BUILDER")
    print("="*80)
    print(f"Päivämäärä:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Arkisto:      runs/.../actions/{datetime.now().strftime('%Y%m%d')}/")
    print(f"Viimeisin:    seasonality_reports/exit_watchlists/latest_exit_watchlist.csv")
    print(f"Stop mult:    {args.stop_mult} × ATR")
    print(f"Stats file:   {args.stats_file or 'Estimoidut tilastot'}")
    print("="*80 + "\n")
    
    build_exit_watchlist(args)


if __name__ == "__main__":
    main()