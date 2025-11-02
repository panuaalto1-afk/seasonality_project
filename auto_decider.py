#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_decider.py - Regime-Adaptive Automated Trading Decision System

Muodostaa action-planin, trade- & sell-listat sekä portfolio_after_sim.csv:n
käyttäen regime-spesifisiä strategioita.

VERSION 2.0:
- Regime detection integration
- Regime-specific strategies
- Adaptive position sizing
- Dynamic filtering based on market regime
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# ==================== LISÄYKSET: Regime system ====================
from regime_detector import RegimeDetector
from regime_predictor import RegimePredictor
from regime_strategies import RegimeStrategy
# ==================================================================

# ------------------------------------------------------------
# Argparse + polkujen varmistus
# ------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Auto Decider – regime-adaptive trading decisions."
    )
    ap.add_argument("--project_root", required=True, help="Projektin juuripolku")
    ap.add_argument("--universe_csv", required=True, help="Universumi CSV")
    ap.add_argument("--run_root", required=True, help="Käytettävä RUN-kansio")
    ap.add_argument("--price_cache_dir", required=True, help="Käytettävä price_cache-kansio")
    ap.add_argument("--today", required=True, help="Päivä YYYY-MM-DD")
    ap.add_argument(
        "--portfolio_state",
        default=None,
        help="Salkun tilan JSON"
    )
    ap.add_argument("--commit", type=int, default=0, help="1 = kirjoita portfolio_state.json; 0 = dry-run")
    ap.add_argument("--config", default=None, help="(valinn.) polku asetustiedostolle")
    ap.add_argument("--no_new_positions", action="store_true", help="Estä uudet positiot")
    ap.add_argument("--vintage_cutoff", default=None, help="(valinn.) aikaraja datalle")
    
    # ==================== UUSI: Base position size ====================
    ap.add_argument("--base_position_size", type=float, default=1000.0, 
                   help="Base position size in $ (default: 1000)")
    # ==================================================================

    args = ap.parse_args()

    # Oletuspolku portfolio_state:lle
    if not args.portfolio_state:
        args.portfolio_state = os.path.join(
            os.path.abspath(args.project_root),
            "seasonality_reports",
            "portfolio_state.json",
        )

    # Peruspolut ja validoinnit
    args.project_root = os.path.abspath(args.project_root)
    args.run_root = os.path.abspath(args.run_root)
    args.price_cache_dir = os.path.abspath(args.price_cache_dir)
    args.universe_csv = os.path.abspath(args.universe_csv)
    args.portfolio_state = os.path.abspath(args.portfolio_state)

    if not os.path.isdir(args.run_root):
        raise FileNotFoundError(f"run_root puuttuu: {args.run_root}")
    if not os.path.isdir(args.price_cache_dir):
        raise FileNotFoundError(f"price_cache_dir puuttuu: {args.price_cache_dir}")
    if not os.path.isfile(args.universe_csv):
        raise FileNotFoundError(f"universe_csv puuttuu: {args.universe_csv}")

    # actions/YYYMMDD
    ymd = args.today.replace("-", "")
    args.actions_dir = os.path.join(args.run_root, "actions", ymd)
    os.makedirs(args.actions_dir, exist_ok=True)

    print(f"[INFO] Using RUN_ROOT : {args.run_root}")
    print(f"[INFO] Using PRICES  : {args.price_cache_dir}")
    print(f"[INFO] Using UNIVERSE: {args.universe_csv}")
    print(f"[INFO] PortfolioState: {args.portfolio_state}")
    print(f"[INFO] Actions dir   : {args.actions_dir}")

    return args


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _safe_read_csv(path: str) -> pd.DataFrame:
    """Lukee CSV:n yrittäen automaattisesti erotinta (,/;)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")

def _write_csv(df: pd.DataFrame, path: str):
    if df is None:
        return
    df.to_csv(path, index=False)

def _load_portfolio_state(path_json: str) -> dict:
    if os.path.isfile(path_json):
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"positions": [], "cash": 100000.0}

def _save_portfolio_state(state: dict, path_json: str):
    Path(os.path.dirname(path_json)).mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def _find_latest_signals_file(run_root: str, today: str) -> str:
    """
    Etsi viimeisin top_long_candidates_GATED_{date}.csv
    """
    reports_dir = os.path.join(run_root, "reports")
    if not os.path.isdir(reports_dir):
        return None
    
    # Etsi today:n tiedosto
    pattern = f"top_long_candidates_GATED_{today}.csv"
    path = os.path.join(reports_dir, pattern)
    
    if os.path.isfile(path):
        return path
    
    # Jos ei löydy, etsi viimeisin
    import glob
    files = glob.glob(os.path.join(reports_dir, "top_long_candidates_GATED_*.csv"))
    if files:
        return max(files, key=os.path.getmtime)
    
    return None

# ------------------------------------------------------------
# Trading logic with regime adaptation
# ------------------------------------------------------------

def build_trade_candidates(args, regime_strategy: RegimeStrategy, regime_data: dict) -> pd.DataFrame:
    """
    Rakenna kaupankäyntiehdotukset regime-strategian mukaan
    
    Args:
        args: Command line arguments
        regime_strategy: RegimeStrategy instance
        regime_data: Regime detection results
    
    Returns:
        DataFrame with columns: Ticker, Side, Score, Note, PositionSize
    """
    # Etsi signaalitiedosto
    signals_path = _find_latest_signals_file(args.run_root, args.today)
    
    if signals_path is None:
        print("[WARN] No signals file found!")
        return pd.DataFrame(columns=["Ticker", "Side", "Score", "Note", "PositionSize"])
    
    print(f"[INFO] Loading signals from: {signals_path}")
    signals_df = _safe_read_csv(signals_path)
    
    if signals_df.empty:
        print("[WARN] Signals file is empty!")
        return pd.DataFrame(columns=["Ticker", "Side", "Score", "Note", "PositionSize"])
    
    print(f"[INFO] Loaded {len(signals_df)} signals")
    
    # Varmista sarakkeet
    if 'ticker' not in signals_df.columns and 'Ticker' in signals_df.columns:
        signals_df['ticker'] = signals_df['Ticker']
    
    if 'mom20' not in signals_df.columns:
        signals_df['mom20'] = signals_df.get('mom20', 0.1)
    
    if 'vol20' not in signals_df.columns:
        signals_df['vol20'] = signals_df.get('vol20', 0.2)
    
    if 'ml_score' not in signals_df.columns:
        # Käytä score_long proxy:na
        signals_df['ml_score'] = signals_df.get('score_long', 0.75)
    
    # ==================== REGIME FILTERING ====================
    print(f"\n[INFO] Applying {regime_data['regime']} strategy filters...")
    print(f"  Signals before filtering: {len(signals_df)}")
    
    # Hae strategy parametrit
    strategy_params = regime_strategy.get_position_parameters(args.base_position_size)
    
    # Rankingoi ja filtteriä
    filtered_df = regime_strategy.rank_signals(
        signals_df,
        top_n=strategy_params['max_positions']
    )
    
    print(f"  Signals after filtering: {len(filtered_df)}")
    # ==========================================================
    
    if filtered_df.empty:
        print("[INFO] No signals passed regime filters")
        return pd.DataFrame(columns=["Ticker", "Side", "Score", "Note", "PositionSize"])
    
    # Muodosta trade candidates
    candidates = []
    
    for idx, row in filtered_df.iterrows():
        ticker = row.get('ticker', row.get('Ticker', ''))
        score = row.get('composite_score', row.get('score_long', 0.0))
        
        candidates.append({
            'Ticker': ticker,
            'Side': 'LONG',
            'Score': float(score),
            'Note': f"Regime: {regime_data['regime']}, Strategy: {regime_strategy.config['strategy_type']}",
            'PositionSize': strategy_params['position_size'],
            'StopMultiplier': strategy_params['stop_multiplier'],
            'TPMultiplier': strategy_params['tp_multiplier']
        })
    
    result_df = pd.DataFrame(candidates)
    
    print(f"\n[INFO] Generated {len(result_df)} trade candidates:")
    print(f"  Position size: ${strategy_params['position_size']:.0f}")
    print(f"  Stop mult: {strategy_params['stop_multiplier']:.2f}x")
    print(f"  TP mult: {strategy_params['tp_multiplier']:.2f}x")
    
    return result_df

def build_sell_candidates(args, portfolio_state: dict, regime_data: dict) -> pd.DataFrame:
    """
    Rakenna myyntiehdotukset salkun positioista
    
    CRISIS regime → Sulje kaikki
    Muut regimes → Normaali exit-logiikka
    """
    positions = portfolio_state.get('positions', [])
    
    if not positions:
        return pd.DataFrame(columns=["Ticker", "Reason"])
    
    sells = []
    
    # CRISIS: Sulje kaikki
    if regime_data['regime'] == 'CRISIS':
        print("[WARN] CRISIS regime detected - suggesting to close all positions")
        for pos in positions:
            sells.append({
                'Ticker': pos.get('ticker', ''),
                'Reason': 'CRISIS regime - capital preservation'
            })
    
    else:
        # Normaali exit-logiikka (placeholder)
        # Tässä voisi olla stop-loss / take-profit / trailing stop logiikka
        pass
    
    return pd.DataFrame(sells)

def simulate_portfolio_after_trades(portfolio_state: dict, buys_df: pd.DataFrame, sells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simuloi salkun tila kauppojen jälkeen
    """
    positions = portfolio_state.get('positions', []).copy()
    
    # Poista myydyt
    if not sells_df.empty:
        sold_tickers = set(sells_df['Ticker'].values)
        positions = [p for p in positions if p.get('ticker', '') not in sold_tickers]
    
    # Lisää ostetut
    if not buys_df.empty:
        for idx, row in buys_df.iterrows():
            positions.append({
                'ticker': row['Ticker'],
                'side': row['Side'],
                'size': row.get('PositionSize', 1000.0),
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
                'score': float(row.get('Score', 0.0))
            })
    
    # Muunna DataFrame:ksi
    if positions:
        result = pd.DataFrame(positions)
    else:
        result = pd.DataFrame(columns=["ticker", "side", "size", "entry_date", "score"])
    
    return result

def write_action_plan(args, buys_df: pd.DataFrame, sells_df: pd.DataFrame, regime_data: dict):
    """Kirjoita action plan tekstitiedostoon"""
    plan = []
    plan.append("="*80)
    plan.append("AUTOMATED TRADING DECISION PLAN")
    plan.append("="*80)
    plan.append(f"Date: {args.today}")
    plan.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    plan.append(f"Universe: {os.path.basename(args.universe_csv)}")
    plan.append("")
    
    # ==================== REGIME INFO ====================
    plan.append("="*80)
    plan.append("MARKET REGIME")
    plan.append("="*80)
    plan.append(f"Regime:     {regime_data['regime']}")
    plan.append(f"Score:      {regime_data['composite_score']:.4f}")
    plan.append(f"Confidence: {regime_data['confidence']:.2%}")
    
    if 'prediction_5d' in regime_data:
        plan.append(f"\n5-day forecast: {regime_data['prediction_5d']} ({regime_data.get('prediction_5d_prob', 0):.1%})")
        plan.append(f"Transition prob: {regime_data.get('transition_prob_5d', 0):.1%}")
    
    plan.append("")
    # =====================================================
    
    plan.append("="*80)
    plan.append("BUY CANDIDATES")
    plan.append("="*80)
    if buys_df is not None and len(buys_df):
        for idx, r in buys_df.iterrows():
            plan.append(f"{idx+1}. {r.get('Ticker','?')} - Score: {r.get('Score', 0):.3f} - Size: ${r.get('PositionSize', 0):.0f}")
            plan.append(f"   Note: {r.get('Note', '')}")
    else:
        plan.append("(none)")
    
    plan.append("")
    plan.append("="*80)
    plan.append("SELL CANDIDATES")
    plan.append("="*80)
    if sells_df is not None and len(sells_df):
        for idx, r in sells_df.iterrows():
            plan.append(f"{idx+1}. {r.get('Ticker','?')} - Reason: {r.get('Reason','')}")
    else:
        plan.append("(none)")
    
    plan.append("")
    plan.append("="*80)

    out = os.path.join(args.actions_dir, "action_plan.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(plan))
    print(f"[OK] Action Plan : {out}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("AUTO DECIDER - REGIME-ADAPTIVE TRADING DECISIONS")
    print("="*80)
    print(f"Date: {args.today}")
    print(f"Base position size: ${args.base_position_size:.0f}")
    print("="*80 + "\n")

    # ==================== REGIME DETECTION ====================
    print("\n" + "="*80)
    print("[STEP] REGIME DETECTION")
    print("="*80 + "\n")
    
    try:
        # Makro cache
        macro_cache_dir = os.path.join(
            args.project_root,
            "seasonality_reports",
            "price_cache"
        )
        
        detector = RegimeDetector(
            macro_price_cache_dir=macro_cache_dir,
            equity_price_cache_dir=args.price_cache_dir
        )
        
        regime_data = detector.detect_regime(date=args.today)
        
        print(f"[INFO] Regime: {regime_data['regime']}")
        print(f"[INFO] Score: {regime_data['composite_score']:.4f}")
        print(f"[INFO] Confidence: {regime_data['confidence']:.2%}")
        
    except Exception as e:
        print(f"[ERROR] Regime detection failed: {e}")
        regime_data = {
            'regime': 'NEUTRAL_BULLISH',
            'composite_score': 0.0,
            'confidence': 0.0
        }
    
    print("="*80 + "\n")
    # ==========================================================
    
    # ==================== REGIME PREDICTION ====================
    print("\n" + "="*80)
    print("[STEP] REGIME PREDICTION")
    print("="*80 + "\n")
    
    try:
        predictor = RegimePredictor()
        pred_5d = predictor.predict(args.today, horizon_days=5)
        
        print(f"[INFO] 5-day forecast: {pred_5d['most_likely']} ({pred_5d['predictions'][pred_5d['most_likely']]:.1%})")
        print(f"[INFO] Transition prob: {pred_5d['transition_probability']:.1%}")
        
        # Lisää ennusteet regime_data:an
        regime_data['prediction_5d'] = pred_5d['most_likely']
        regime_data['prediction_5d_prob'] = pred_5d['predictions'][pred_5d['most_likely']]
        regime_data['transition_prob_5d'] = pred_5d['transition_probability']
        
        if pred_5d['transition_probability'] > 0.70:
            print(f"[WARN] High transition probability - consider defensive positioning")
        
    except Exception as e:
        print(f"[WARN] Regime prediction failed: {e}")
    
    print("="*80 + "\n")
    # ===========================================================
    
    # ==================== REGIME STRATEGY ====================
    print("\n" + "="*80)
    print("[STEP] LOAD REGIME STRATEGY")
    print("="*80 + "\n")
    
    regime_strategy = RegimeStrategy(regime_data['regime'])
    regime_strategy.print_strategy()
    
    # Override no_new_positions if CRISIS
    if regime_data['regime'] == 'CRISIS':
        args.no_new_positions = True
        print("[WARN] CRISIS regime - no_new_positions enabled automatically")
    
    print("="*80 + "\n")
    # =========================================================

    # Lataa salkun tila
    state = _load_portfolio_state(args.portfolio_state)

    # Rakenna listat
    if args.no_new_positions:
        print("[INFO] No new positions mode - skipping trade candidate generation")
        buys_df = pd.DataFrame(columns=["Ticker", "Side", "Score", "Note", "PositionSize"])
    else:
        buys_df = build_trade_candidates(args, regime_strategy, regime_data)
    
    sells_df = build_sell_candidates(args, state, regime_data)
    port_df = simulate_portfolio_after_trades(state, buys_df, sells_df)

    # Kirjoitukset
    trade_path = os.path.join(args.actions_dir, "trade_candidates.csv")
    sell_path = os.path.join(args.actions_dir, "sell_candidates.csv")
    port_path = os.path.join(args.actions_dir, "portfolio_after_sim.csv")

    _write_csv(buys_df, trade_path)
    print(f"[OK] Candidates : {trade_path}")

    _write_csv(sells_df, sell_path)
    print(f"[OK] Sells      : {sell_path}")

    _write_csv(port_df, port_path)
    print(f"[OK] Portfolio  : {port_path}")

    # Action plan
    write_action_plan(args, buys_df, sells_df, regime_data)

    # Tallennetaanko portfolio_state.json?
    if int(args.commit) == 1:
        # Päivitä state
        state['positions'] = port_df.to_dict('records') if not port_df.empty else []
        state['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        state['regime'] = regime_data['regime']
        
        _save_portfolio_state(state, args.portfolio_state)
        print(f"[OK] Portfolio state COMMITTED: {args.portfolio_state}")
    else:
        print("[INFO] Dry-run: portfolio_state.json ei päivitetty (commit=0).")
    
    print("\n" + "="*80)
    print("AUTO DECIDER COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()