
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_exit_watchlist.py - Rakentaa exit-watchlistin todennäköisyyspohjaisilla TP-tasoilla

KÄYTTÖ:
    python make_exit_watchlist.py --portfolio portfolio_after_sim.csv
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
# HUOM: Nämä korvataan myöhemmin historical_exit_stats.csv -datalla

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, default="portfolio_after_sim.csv")
    parser.add_argument("--price_cache", type=str, default="seasonality_reports/price_cache")
    parser.add_argument("--output", type=str, default="exit_watchlist.csv")
    parser.add_argument("--stop_mult", type=float, default=1.5)
    parser.add_argument("--stats_file", type=str, default=None)
    return parser.parse_args()


class PriceDataLoader:
    def __init__(self, price_cache_dir: Path):
        self.price_cache_dir = Path(price_cache_dir)
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        price_file = self.price_cache_dir / f"{ticker}.csv"
        if not price_file.exists():
            return None
        try:
            df = pd.read_csv(price_file, parse_dates=['Date'])
            return df.sort_values('Date')
        except:
            return None
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 20) -> float:
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return atr if not pd.isna(atr) else 0.0


def load_setup_statistics(stats_file: Optional[str]) -> Dict:
    if stats_file and Path(stats_file).exists():
        print(f"📊 Ladataan: {stats_file}")
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
    portfolio_path = Path(args.portfolio)
    price_cache_dir = Path(args.price_cache)
    
    if not portfolio_path.exists():
        print(f"❌ Portfolio ei löydy: {portfolio_path}")
        return
    
    portfolio = pd.read_csv(portfolio_path)
    if portfolio.empty:
        print("⚠️ Portfolio tyhjä")
        return
    
    setup_stats = load_setup_statistics(args.stats_file)
    price_loader = PriceDataLoader(price_cache_dir)
    exit_data = []
    
    for _, pos in portfolio.iterrows():
        ticker = pos.get('Ticker') or pos.get('ticker')
        entry = pos.get('Entry')
        setup_type = pos.get('Setup_Type', 'Default')
        
        if not ticker or pd.isna(entry):
            continue
        
        df = price_loader.load_ticker(ticker)
        if df is None or df.empty:
            continue
        
        current = df['Close'].iloc[-1]
        atr = price_loader.calculate_atr(df)
        
        if atr == 0:
            continue
        
        risk = args.stop_mult * atr
        stop = entry - risk
        tp1 = entry + (1.5 * risk)
        tp2 = entry + (3.0 * risk)
        tp3 = entry + (5.0 * risk)
        
        stats = setup_stats.get(setup_type, setup_stats['Default'])
        trail_stop = calculate_trailing_stop(entry, current, stop, 0)
        
        profit_pct = ((current - entry) / entry) * 100
        note = f"{'Voitolla' if profit_pct > 0 else 'Tappiolla'} {profit_pct:.1f}%"
        
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
            'Trail_Stop': round(trail_stop, 2),
            'Note': note
        })
    
    exit_df = pd.DataFrame(exit_data)
    exit_df.to_csv(args.output, index=False)
    
    print(f"\n✅ Kirjoitettu: {args.output}")
    print(f"📊 {len(exit_df)} positiota\n")
    
    if not exit_df.empty:
        print("="*80)
        print(exit_df.to_string(index=False))
        print("="*80)


def main():
    args = parse_args()
    print("="*80)
    print("🎯 EXIT WATCHLIST (Estimoidut tilastot)")
    print("="*80)
    build_exit_watchlist(args)


if __name__ == "__main__":
    main()