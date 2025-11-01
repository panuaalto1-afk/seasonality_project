#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_decider.py - Automaattinen kauppapäätösten tekijä

Lukee:
- ML-signaalit (ml_unified_pipeline output)
- Hintadata (price_cache)
- Seasonality-data

Tuottaa:
- trade_candidates.csv (kaikki potentiaaliset kaupat)
- portfolio_after_sim.csv (simuloitu portfolio, top N signaaleja)

Tallentaa:
1. runs/LATEST_RUN/actions/YYYYMMDD/ (arkisto)
2. seasonality_reports/trade_decisions/ (viimeisimmät)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Auto Trade Decider")
    parser.add_argument("--runs_dir", type=str, default="seasonality_reports/runs",
                        help="Runs directory")
    parser.add_argument("--seasonality_dir", type=str, default="seasonality_reports",
                        help="Seasonality data directory")
    parser.add_argument("--price_cache_dir", type=str, default="seasonality_reports/runs/2025-10-04_0903/price_cache",
                        help="Price cache directory (overwrite version)")
    parser.add_argument("--top_n", type=int, default=15,
                        help="Top N signals to include in portfolio (default: 15)")
    parser.add_argument("--stop_mult", type=float, default=1.5,
                        help="Stop loss multiplier (default: 1.5)")
    parser.add_argument("--min_ml_score", type=float, default=0.75,
                        help="Minimum ML score to consider (default: 0.75)")
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
            return None
        try:
            df = pd.read_csv(price_file, parse_dates=['Date'])
            
            # Muunna hintasarakkeet numeerisiksi
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['High', 'Low', 'Close'])
            return df.sort_values('Date')
        except Exception as e:
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
            return 0.0


class SeasonalityDataLoader:
    """Lataa seasonality-dataa"""
    
    def __init__(self, seasonality_dir: Path):
        self.seasonality_dir = Path(seasonality_dir)
    
    def load_ticker_seasonality(self, ticker: str) -> Dict:
        """Lataa tickerin seasonality-data (week, segments_up, segments_down)"""
        result = {
            'week': None,
            'segments_up': None,
            'segments_down': None,
            'current_week_score': 0.0
        }
        
        # Viikkodata
        week_file = self.seasonality_dir / f"{ticker}_seasonality_week.csv"
        if week_file.exists():
            try:
                result['week'] = pd.read_csv(week_file)
            except:
                pass
        
        # Segments up
        segments_up_file = self.seasonality_dir / f"{ticker}_segments_up.csv"
        if segments_up_file.exists():
            try:
                result['segments_up'] = pd.read_csv(segments_up_file)
            except:
                pass
        
        # Segments down
        segments_down_file = self.seasonality_dir / f"{ticker}_segments_down.csv"
        if segments_down_file.exists():
            try:
                result['segments_down'] = pd.read_csv(segments_down_file)
            except:
                pass
        
        # Laske nykyisen viikon score
        current_week = datetime.now().isocalendar()[1]
        if result['week'] is not None:
            week_data = result['week'][result['week']['week_number'] == current_week]
            if not week_data.empty:
                result['current_week_score'] = float(week_data['AvgRet_WOY'].values[0])
        
        return result


def find_latest_run(runs_dir: Path) -> Optional[Path]:
    """Etsi viimeisin run-kansio"""
    if not runs_dir.exists():
        return None
    
    run_dirs = sorted([d for d in runs_dir.glob("*") if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
    
    return run_dirs[0] if run_dirs else None


def find_latest_ml_signals(reports_dir: Path) -> Optional[Path]:
    """Etsi viimeisin ML-signaali-tiedosto (GATED long candidates)"""
    if not reports_dir.exists():
        return None
    
    signal_files = list(reports_dir.glob("top_long_candidates_GATED_*.csv"))
    if not signal_files:
        return None
    
    # Järjestä päivämäärän mukaan (uusin ensin)
    signal_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return signal_files[0]


def determine_setup_type(ticker: str, ml_score: float, seasonality_score: float,
                        mom5: float, mom20: float) -> str:
    """
    Määritä setup-tyyppi ML-scoren ja seasonalityn perusteella
    
    Logiikka:
    - ML_Seasonality_Combo: Korkea ML + vahva seasonality
    - ML_Momentum_Strong: Korkea ML + korkea momentum
    - ML_Momentum_Moderate: Keskitaso ML
    - Seasonality_Pure: Vahva seasonality, matala ML
    """
    # ML_Seasonality_Combo: ML > 0.9 JA seasonality > 1.5%
    if ml_score > 0.9 and seasonality_score > 0.015:
        return 'ML_Seasonality_Combo'
    
    # ML_Momentum_Strong: ML > 0.9
    if ml_score > 0.9:
        return 'ML_Momentum_Strong'
    
    # ML_Seasonality_Combo: ML > 0.8 JA seasonality > 1.0%
    if ml_score > 0.8 and seasonality_score > 0.010:
        return 'ML_Seasonality_Combo'
    
    # ML_Momentum_Moderate: ML > 0.8
    if ml_score > 0.8:
        return 'ML_Momentum_Moderate'
    
    # Seasonality_Pure: Seasonality > 1.5%
    if seasonality_score > 0.015:
        return 'Seasonality_Pure'
    
    # Default
    return 'ML_Momentum_Moderate'


def calculate_entry_stop_tp(entry: float, atr: float, stop_mult: float = 1.5) -> Dict:
    """
    Laske entry/stop/TP-tasot ATR-pohjaisesti
    
    Logiikka:
    - Entry = current price
    - Stop = entry - (stop_mult × ATR)
    - TP1 = entry + (1.5 × risk)
    - TP2 = entry + (3.0 × risk)
    - TP3 = entry + (5.0 × risk)
    """
    risk = stop_mult * atr
    stop = entry - risk
    tp1 = entry + (1.5 * risk)
    tp2 = entry + (3.0 * risk)
    tp3 = entry + (5.0 * risk)
    
    return {
        'entry': round(entry, 2),
        'stop': round(stop, 2),
        'tp1': round(tp1, 2),
        'tp2': round(tp2, 2),
        'tp3': round(tp3, 2),
        'atr': round(atr, 4)
    }


def process_ml_signals(args):
    """Prosessoi ML-signaalit ja luo trade candidates + portfolio"""
    
    print("="*80)
    print("🤖 AUTO DECIDER - Kauppapäätösten automaatio")
    print("="*80)
    print(f"Päivämäärä:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Min ML Score:   {args.min_ml_score}")
    print(f"Portfolio size: Top {args.top_n}")
    print(f"Stop mult:      {args.stop_mult} × ATR")
    print("="*80 + "\n")
    
    # Etsi viimeisin run
    runs_dir = Path(args.runs_dir)
    latest_run = find_latest_run(runs_dir)
    
    if not latest_run:
        print("❌ Ei löytynyt run-kansioita")
        return
    
    print(f"📂 Viimeisin run: {latest_run.name}")
    
    # Etsi ML-signaalit
    reports_dir = latest_run / "reports"
    ml_signals_file = find_latest_ml_signals(reports_dir)
    
    if not ml_signals_file:
        print(f"❌ ML-signaaleja ei löydy: {reports_dir}")
        return
    
    print(f"📊 ML-signaalit: {ml_signals_file.name}")
    
    # Lue ML-signaalit
    try:
        ml_signals = pd.read_csv(ml_signals_file)
        print(f"📊 Luettu {len(ml_signals)} ML-signaalia\n")
    except Exception as e:
        print(f"❌ Virhe lukiessa ML-signaaleja: {e}")
        return
    
    # Suodata ML-scoren mukaan
    ml_signals = ml_signals[ml_signals['score_long'] >= args.min_ml_score]
    print(f"📊 Suodatettu: {len(ml_signals)} signaalia (ML score >= {args.min_ml_score})\n")
    
    if ml_signals.empty:
        print("⚠️  Ei signaaleja suodatuksen jälkeen")
        return
    
    # Lataa data
    price_loader = PriceDataLoader(Path(args.price_cache_dir))
    seasonality_loader = SeasonalityDataLoader(Path(args.seasonality_dir))
    
    # Prosessoi signaalit
    trade_candidates = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    print("🔄 Prosessoidaan signaalit...\n")
    
    for idx, row in ml_signals.iterrows():
        ticker = row['ticker']
        ml_score = row['score_long']
        mom5 = row.get('mom5', 0.0)
        mom20 = row.get('mom20', 0.0)
        vol20 = row.get('vol20', 0.0)
        
        # Lataa hintadata
        price_df = price_loader.load_ticker(ticker)
        if price_df is None or price_df.empty:
            print(f"  ⚠️  {ticker}: Ei hintadataa")
            continue
        
        current_price = float(price_df['Close'].iloc[-1])
        atr = price_loader.calculate_atr(price_df)
        
        if atr == 0:
            print(f"  ⚠️  {ticker}: ATR=0")
            continue
        
        # Lataa seasonality-data
        seasonality = seasonality_loader.load_ticker_seasonality(ticker)
        seasonality_score = seasonality['current_week_score']
        
        # Määritä setup-tyyppi
        setup_type = determine_setup_type(ticker, ml_score, seasonality_score, mom5, mom20)
        
        # Laske entry/stop/TP
        levels = calculate_entry_stop_tp(current_price, atr, args.stop_mult)
        
        # Lisää kandidaatti
        trade_candidates.append({
            'Ticker': ticker,
            'Entry': levels['entry'],
            'Stop': levels['stop'],
            'TP1': levels['tp1'],
            'TP2': levels['tp2'],
            'TP3': levels['tp3'],
            'Setup_Type': setup_type,
            'ML_Score': round(ml_score, 4),
            'Mom5': round(mom5, 4),
            'Mom20': round(mom20, 4),
            'Vol20': round(vol20, 4),
            'Seasonality_Score': round(seasonality_score, 4),
            'Entry_Date': today_str,
            'ATR': levels['atr']
        })
        
        print(f"  ✅ {ticker}: ML={ml_score:.3f}, Season={seasonality_score:.3f}, Setup={setup_type}")
    
    if not trade_candidates:
        print("\n⚠️  Ei kauppakandidaatteja")
        return
    
    candidates_df = pd.DataFrame(trade_candidates)
    
    print(f"\n📊 Yhteensä {len(candidates_df)} kauppakandidaattia")
    
    # Simuloi portfolio: Ota top N ML-scoren mukaan
    portfolio_df = candidates_df.nlargest(args.top_n, 'ML_Score').copy()
    
    # Lisää Side ja Comment
    portfolio_df['Side'] = 'LONG'
    portfolio_df['Comment'] = portfolio_df.apply(
        lambda x: f"ML={x['ML_Score']:.3f}, {x['Setup_Type']}", axis=1
    )
    
    print(f"📊 Portfolio: Top {len(portfolio_df)} signaalia\n")
    
    # ==================== TALLENNA ====================
    today_yyyymmdd = datetime.now().strftime("%Y%m%d")
    
    # 1. Tallenna actions-kansioon (arkisto)
    actions_dir = latest_run / "actions" / today_yyyymmdd
    actions_dir.mkdir(parents=True, exist_ok=True)
    
    candidates_output = actions_dir / f"trade_candidates_{today_yyyymmdd}.csv"
    portfolio_output = actions_dir / "portfolio_after_sim.csv"
    
    try:
        candidates_df.to_csv(candidates_output, index=False)
        print(f"✅ Arkisto (candidates): {candidates_output}")
    except Exception as e:
        print(f"⚠️  Tallennusvirhe: {e}")
    
    try:
        portfolio_df.to_csv(portfolio_output, index=False)
        print(f"✅ Arkisto (portfolio):  {portfolio_output}")
    except Exception as e:
        print(f"⚠️  Tallennusvirhe: {e}")
    
    # 2. Tallenna trade_decisions-kansioon (viimeisimmät)
    decisions_dir = Path("seasonality_reports/trade_decisions")
    decisions_dir.mkdir(parents=True, exist_ok=True)
    
    latest_candidates = decisions_dir / "latest_trade_candidates.csv"
    latest_portfolio = decisions_dir / "latest_portfolio.csv"
    
    try:
        candidates_df.to_csv(latest_candidates, index=False)
        print(f"✅ Viimeisin (candidates): {latest_candidates}")
    except Exception as e:
        print(f"⚠️  Tallennusvirhe: {e}")
    
    try:
        portfolio_df.to_csv(latest_portfolio, index=False)
        print(f"✅ Viimeisin (portfolio):  {latest_portfolio}")
    except Exception as e:
        print(f"⚠️  Tallennusvirhe: {e}")
    
    # Tulosta yhteenveto
    print("\n" + "="*80)
    print("📈 PORTFOLIO YHTEENVETO (TOP SIGNALS):")
    print("="*80)
    print(portfolio_df[['Ticker', 'Entry', 'Stop', 'TP1', 'Setup_Type', 'ML_Score', 'Seasonality_Score']].to_string(index=False))
    print("="*80)
    
    # Setup-tyyppi-jakauma
    print("\n📊 SETUP-TYYPIT:")
    setup_counts = candidates_df['Setup_Type'].value_counts()
    for setup, count in setup_counts.items():
        print(f"  {setup}: {count}")
    
    print("\n✅ Valmis!")


def main():
    args = parse_args()
    process_ml_signals(args)


if __name__ == "__main__":
    main()