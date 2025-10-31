#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_historical_exits.py - Analysoi historiallisia kauppoja ja laskee exit-tilastot
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Analysoi historiallisia exit-tilastoja")
    parser.add_argument("--runs_dir", type=str, default="seasonality_reports/runs")
    parser.add_argument("--price_cache", type=str, default="seasonality_reports/price_cache")
    parser.add_argument("--lookback", type=int, default=180)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--stop_mult", type=float, default=1.5)
    parser.add_argument("--max_hold_days", type=int, default=30)
    return parser.parse_args()

class TradeSimulator:
    def __init__(self, price_cache_dir: Path, stop_mult: float = 1.5, max_hold_days: int = 30):
        self.price_cache_dir = Path(price_cache_dir)
        self.stop_mult = stop_mult
        self.max_hold_days = max_hold_days
    
    def load_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        price_file = self.price_cache_dir / f"{ticker}.csv"
        if not price_file.exists():
            return None
        try:
            df = pd.read_csv(price_file, parse_dates=['Date'])
            return df.sort_values('Date')
        except Exception:
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
    
    def simulate_exit(self, ticker: str, entry_date: str, entry_price: float, stop_loss: Optional[float] = None, side: str = "long") -> Dict:
        df = self.load_price_data(ticker)
        if df is None:
            return self._empty_result("no_data")
        if isinstance(entry_date, str):
            entry_date = pd.to_datetime(entry_date)
        df_after_entry = df[df['Date'] >= entry_date].copy()
        if df_after_entry.empty:
            return self._empty_result("future_entry")
        if entry_price is None:
            entry_price = df_after_entry.iloc[0]['Close']
        df_before_entry = df[df['Date'] <= entry_date]
        if len(df_before_entry) < 20:
            return self._empty_result("insufficient_history")
        atr = self.calculate_atr(df_before_entry, period=20)
        if stop_loss is None:
            risk = self.stop_mult * atr
            stop_loss = entry_price - risk if side == "long" else entry_price + risk
        risk = abs(entry_price - stop_loss)
        tp1 = entry_price + (1.5 * risk) if side == "long" else entry_price - (1.5 * risk)
        tp2 = entry_price + (3.0 * risk) if side == "long" else entry_price - (3.0 * risk)
        tp3 = entry_price + (5.0 * risk) if side == "long" else entry_price - (5.0 * risk)
        max_favorable = entry_price
        max_adverse = entry_price
        for i, (idx, row) in enumerate(df_after_entry.iterrows()):
            days_held = i
            if days_held > self.max_hold_days:
                return {'exit_date': row['Date'], 'exit_price': row['Close'], 'exit_reason': 'time', 'return_pct': (row['Close'] - entry_price) / entry_price, 'return_R': (row['Close'] - entry_price) / risk, 'hold_days': days_held, 'max_favorable': max_favorable, 'max_adverse': max_adverse, 'hit_stop': False}
            if side == "long":
                max_favorable = max(max_favorable, row['High'])
                max_adverse = min(max_adverse, row['Low'])
                if row['Low'] <= stop_loss:
                    return {'exit_date': row['Date'], 'exit_price': stop_loss, 'exit_reason': 'stop', 'return_pct': (stop_loss - entry_price) / entry_price, 'return_R': -1.0, 'hold_days': days_held, 'max_favorable': max_favorable, 'max_adverse': max_adverse, 'hit_stop': True}
                if row['High'] >= tp3:
                    return self._create_exit_result(row, entry_price, risk, tp3, 'tp3', days_held, max_favorable, max_adverse)
                elif row['High'] >= tp2:
                    return self._create_exit_result(row, entry_price, risk, tp2, 'tp2', days_held, max_favorable, max_adverse)
                elif row['High'] >= tp1:
                    return self._create_exit_result(row, entry_price, risk, tp1, 'tp1', days_held, max_favorable, max_adverse)
        last_row = df_after_entry.iloc[-1]
        return {'exit_date': last_row['Date'], 'exit_price': last_row['Close'], 'exit_reason': 'incomplete', 'return_pct': (last_row['Close'] - entry_price) / entry_price, 'return_R': (last_row['Close'] - entry_price) / risk, 'hold_days': len(df_after_entry) - 1, 'max_favorable': max_favorable, 'max_adverse': max_adverse, 'hit_stop': False}
    
    def _create_exit_result(self, row, entry, risk, exit_price, reason, days, max_fav, max_adv, side='long'):
        mult = 1 if side == 'long' else -1
        return {'exit_date': row['Date'], 'exit_price': exit_price, 'exit_reason': reason, 'return_pct': mult * (exit_price - entry) / entry, 'return_R': mult * (exit_price - entry) / risk, 'hold_days': days, 'max_favorable': max_fav, 'max_adverse': max_adv, 'hit_stop': False}
    
    def _empty_result(self, reason: str) -> Dict:
        return {'exit_date': None, 'exit_price': None, 'exit_reason': reason, 'return_pct': 0.0, 'return_R': 0.0, 'hold_days': 0, 'max_favorable': 0.0, 'max_adverse': 0.0, 'hit_stop': False}

def find_trade_files(runs_dir: Path, lookback_days: int) -> List[Tuple[Path, datetime]]:
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    trade_files = []
    for run_dir in sorted(runs_dir.glob("2025-*")):
        try:
            date_str = run_dir.name.split("_")[0]
            run_date = datetime.strptime(date_str, "%Y-%m-%d")
            if run_date < cutoff_date:
                continue
            actions_dir = run_dir / "actions"
            if not actions_dir.exists():
                continue
            for date_subdir in actions_dir.iterdir():
                if not date_subdir.is_dir():
                    continue
                trade_file = date_subdir / "trade_candidates.csv"
                if trade_file.exists():
                    trade_files.append((trade_file, run_date))
        except Exception:
            continue
    return trade_files

def calculate_setup_statistics(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    trades_df['is_win'] = trades_df['return_R'] > 0
    stats_list = []
    for setup_type in trades_df['setup_type'].unique():
        subset = trades_df[trades_df['setup_type'] == setup_type]
        if len(subset) < 3:
            continue
        wins = subset[subset['is_win']]
        losses = subset[~subset['is_win']]
        win_rate = len(wins) / len(subset)
        avg_win = wins['return_pct'].mean() if len(wins) > 0 else 0.0
        avg_loss = losses['return_pct'].mean() if len(losses) > 0 else 0.0
        avg_win_R = wins['return_R'].mean() if len(wins) > 0 else 0.0
        avg_loss_R = losses['return_R'].mean() if len(losses) > 0 else 0.0
        if avg_loss != 0:
            profit_factor = (avg_win * win_rate) / (abs(avg_loss) * (1 - win_rate))
            R = avg_win / abs(avg_loss)
            kelly = win_rate - ((1 - win_rate) / R) if R > 0 else 0.0
        else:
            profit_factor = np.inf if avg_win > 0 else 0.0
            kelly = 0.0
        tp1_hits = len(subset[subset['exit_reason'] == 'tp1'])
        tp2_hits = len(subset[subset['exit_reason'] == 'tp2'])
        tp3_hits = len(subset[subset['exit_reason'] == 'tp3'])
        stop_hits = len(subset[subset['hit_stop']])
        stats_list.append({'setup_type': setup_type, 'total_trades': len(subset), 'win_rate': round(win_rate, 3), 'avg_win_pct': round(avg_win * 100, 2), 'avg_loss_pct': round(avg_loss * 100, 2), 'avg_win_R': round(avg_win_R, 2), 'avg_loss_R': round(avg_loss_R, 2), 'profit_factor': round(profit_factor, 2), 'kelly_criterion': round(kelly, 3), 'median_hold_days': int(subset['hold_days'].median()), 'tp1_hit_rate': round(tp1_hits / len(subset), 3), 'tp2_hit_rate': round(tp2_hits / len(subset), 3), 'tp3_hit_rate': round(tp3_hits / len(subset), 3), 'stop_hit_rate': round(stop_hits / len(subset), 3)})
    return pd.DataFrame(stats_list).sort_values('profit_factor', ascending=False)

def analyze_all_trades(args):
    runs_dir = Path(args.runs_dir)
    price_cache_dir = Path(args.price_cache)
    output_dir = Path(args.output_dir)
    if not runs_dir.exists():
        print(f"Runs-kansio ei loydy: {runs_dir}")
        return None, None
    if not price_cache_dir.exists():
        print(f"Price cache ei loydy: {price_cache_dir}")
        return None, None
    print(f"Etsitaan kauppoja viimeiselta {args.lookback} paivalta...")
    trade_files = find_trade_files(runs_dir, args.lookback)
    if not trade_files:
        print("Ei loytynyt trade_candidates.csv")
        return None, None
    print(f"Loytyi {len(trade_files)} ajoa")
    simulator = TradeSimulator(price_cache_dir, args.stop_mult, args.max_hold_days)
    all_trades = []
    for trade_file, run_date in trade_files:
        print(f"Analysoidaan: {trade_file.parent.parent.name}/{trade_file.parent.name}")
        try:
            trades_df = pd.read_csv(trade_file)
            if trades_df.empty:
                continue
            ticker_col = None
            for col in trades_df.columns:
                if col.lower() in ['ticker', 'symbol']:
                    ticker_col = col
                    break
            if ticker_col is None:
                continue
            for _, trade in trades_df.iterrows():
                result = simulator.simulate_exit(ticker=trade[ticker_col], entry_date=run_date, entry_price=trade.get('Entry'), stop_loss=trade.get('Stop'), side=trade.get('Side', 'long').lower())
                all_trades.append({'ticker': trade[ticker_col], 'entry_date': run_date, 'entry_price': trade.get('Entry'), 'side': trade.get('Side', 'long').lower(), 'setup_type': trade.get('Setup_Type', 'Unknown'), 'score': trade.get('Score'), **result})
        except Exception:
            continue
    if not all_trades:
        print("Ei onnistuneita simulaatioita")
        return None, None
    print(f"Analysoitu {len(all_trades)} kauppaa")
    trades_df = pd.DataFrame(all_trades)
    complete_trades = trades_df[trades_df['exit_reason'] != 'incomplete'].copy()
    detailed_file = output_dir / "historical_trades_detailed.csv"
    trades_df.to_csv(detailed_file, index=False)
    print(f"Kirjoitettu: {detailed_file}")
    stats = calculate_setup_statistics(complete_trades)
    if not stats.empty:
        stats_file = output_dir / "historical_exit_stats.csv"
        stats.to_csv(stats_file, index=False)
        print(f"Kirjoitettu: {stats_file}")
        print("TILASTOT PER SETUP-TYYPPI:")
        print(stats.to_string(index=False))
    return trades_df, stats

def main():
    args = parse_args()
    print("HISTORIALLINEN EXIT-ANALYYSI")
    analyze_all_trades(args)

if __name__ == "__main__":
    main()