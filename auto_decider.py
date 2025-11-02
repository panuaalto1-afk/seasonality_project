#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ==================== LISÄYS: Regime-parametrit ====================
REGIME_PARAMS = {
    'BULL_STRONG': {
        'position_size_multiplier': 1.3,      # 30% suuremmat positiot
        'min_ml_score': 0.85,                 # Tiukemmat kriteerit
        'max_positions': 12,                  # Enemmän kauppoja
        'stop_loss_multiplier': 1.0,          # Normaali stop
        'take_profit_multiplier': 1.2,        # Korkeampi TP
        'allow_new_entries': True
    },
    'BULL_WEAK': {
        'position_size_multiplier': 1.0,
        'min_ml_score': 0.80,
        'max_positions': 10,
        'stop_loss_multiplier': 0.9,          # Tiukempi stop (korkea vol)
        'take_profit_multiplier': 1.0,
        'allow_new_entries': True
    },
    'NEUTRAL_BULLISH': {
        'position_size_multiplier': 0.9,
        'min_ml_score': 0.75,
        'max_positions': 8,
        'stop_loss_multiplier': 1.0,
        'take_profit_multiplier': 1.0,
        'allow_new_entries': True
    },
    'NEUTRAL_BEARISH': {
        'position_size_multiplier': 0.7,
        'min_ml_score': 0.70,
        'max_positions': 6,
        'stop_loss_multiplier': 0.95,
        'take_profit_multiplier': 0.9,
        'allow_new_entries': True
    },
    'BEAR_WEAK': {
        'position_size_multiplier': 0.5,
        'min_ml_score': 0.65,
        'max_positions': 4,
        'stop_loss_multiplier': 0.85,
        'take_profit_multiplier': 0.8,
        'allow_new_entries': True
    },
    'BEAR_STRONG': {
        'position_size_multiplier': 0.3,      # Pienet positiot
        'min_ml_score': 0.60,                 # Löysemmät kriteerit
        'max_positions': 2,                   # Vähän kauppoja
        'stop_loss_multiplier': 0.8,          # Tiukka stop
        'take_profit_multiplier': 0.7,
        'allow_new_entries': True
    },
    'CRISIS': {
        'position_size_multiplier': 0.0,
        'min_ml_score': 0.95,
        'max_positions': 0,
        'stop_loss_multiplier': 0.7,          # Hyvin tiukka
        'take_profit_multiplier': 0.5,
        'allow_new_entries': False            # EI UUSIA KAUPPOJA
    }
}
# ====================================================================

# ------------------------------------------------------------
# Argparse + polkujen varmistus
# ------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Auto Decider – muodostaa action-planin, trade- & sell-listat sekä portfolio_after_sim.csv:n."
    )
    ap.add_argument("--project_root", required=True, help="Projektin juuripolku (C:\\...\\seasonality_project)")
    ap.add_argument("--universe_csv", required=True, help="Universumi CSV (esim. seasonality_reports\\constituents_raw.csv)")
    ap.add_argument("--run_root", required=True, help="Käytettävä RUN-kansio (älä muuta jos käytät vakiota)")
    ap.add_argument("--price_cache_dir", required=True, help="Käytettävä price_cache-kansio")
    ap.add_argument("--today", required=True, help="Päivä YYYY-MM-DD")
    ap.add_argument(
        "--portfolio_state",
        default=None,
        help="Salkun tilan JSON (oletus: <project_root>\\seasonality_reports\\portfolio_state.json)"
    )
    ap.add_argument("--commit", type=int, default=0, help="1 = kirjoita portfolio_state.json; 0 = dry-run")
    ap.add_argument("--config", default=None, help="(valinn.) polku asetustiedostolle")
    ap.add_argument("--no_new_positions", action="store_true", help="Estä uudet positiot (vain myyntejä/kevennyksiä)")
    ap.add_argument("--vintage_cutoff", default=None, help="(valinn.) aikaraja datalle")

    args = ap.parse_args()

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


# ==================== LISÄYS: Regime-funktiot ====================

def get_latest_regime(project_root: str) -> dict:
    """Lue viimeisin regime regime_history.csv:stä"""
    try:
        regime_csv = os.path.join(project_root, "seasonality_reports", "regime_history.csv")
        if not os.path.exists(regime_csv):
            print("[WARN] regime_history.csv not found, using Neutral")
            return {'regime': 'NEUTRAL_BULLISH', 'composite_score': 0.0, 'confidence': 0.0}
        
        df = pd.read_csv(regime_csv)
        if df.empty:
            return {'regime': 'NEUTRAL_BULLISH', 'composite_score': 0.0, 'confidence': 0.0}
        
        # Viimeisin rivi
        latest = df.iloc[-1].to_dict()
        
        print(f"[REGIME] Current: {latest['regime']}")
        print(f"[REGIME] Score:   {latest['composite_score']:.4f}")
        print(f"[REGIME] Confidence: {latest['confidence']:.2%}")
        
        return latest
    except Exception as e:
        print(f"[WARN] Failed to read regime: {e}")
        return {'regime': 'NEUTRAL_BULLISH', 'composite_score': 0.0, 'confidence': 0.0}


def get_regime_params(regime: str) -> dict:
    """Hae regime-kohtaiset parametrit"""
    params = REGIME_PARAMS.get(regime, REGIME_PARAMS['NEUTRAL_BULLISH'])
    
    print(f"[REGIME] Using params:")
    print(f"  - Position size multiplier: {params['position_size_multiplier']:.1f}x")
    print(f"  - Min ML score: {params['min_ml_score']:.2f}")
    print(f"  - Max positions: {params['max_positions']}")
    print(f"  - Allow new entries: {params['allow_new_entries']}")
    
    return params

# =================================================================

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
    return {"positions": []}

def _save_portfolio_state(state: dict, path_json: str):
    Path(os.path.dirname(path_json)).mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ==================== PÄIVITETTY: build_trade_candidates ====================

def build_trade_candidates(args, regime_data: dict) -> pd.DataFrame:
    """
    Rakenna trade-kandidaatit regime-mukautetuilla parametreilla
    
    MUUTOS: Nyt käytetään regime_data:a filtteröintiin ja position sizeen
    """
    regime = regime_data['regime']
    params = get_regime_params(regime)
    
    # CRISIS: Ei uusia kauppoja
    if not params['allow_new_entries']:
        print(f"[REGIME] {regime}: NO NEW ENTRIES ALLOWED")
        return pd.DataFrame([], columns=["Ticker", "Side", "Score", "PositionSize", "Regime", "Note"])
    
    # Etsi viimeisin signaalitiedosto
    reports_dir = os.path.join(args.run_root, "reports")
    if not os.path.isdir(reports_dir):
        print(f"[WARN] Reports dir not found: {reports_dir}")
        return pd.DataFrame([], columns=["Ticker", "Side", "Score", "PositionSize", "Regime", "Note"])
    
    # Etsi viimeisin top_long_candidates_GATED_*.csv
    import glob
    gated_files = glob.glob(os.path.join(reports_dir, "top_long_candidates_GATED_*.csv"))
    if not gated_files:
        print(f"[WARN] No gated signal files found in {reports_dir}")
        return pd.DataFrame([], columns=["Ticker", "Side", "Score", "PositionSize", "Regime", "Note"])
    
    # Viimeisin tiedosto
    latest_file = max(gated_files, key=os.path.getmtime)
    print(f"[INFO] Using signals: {os.path.basename(latest_file)}")
    
    signals_df = pd.read_csv(latest_file)
    
    if signals_df.empty:
        print("[INFO] No signals in gated file")
        return pd.DataFrame([], columns=["Ticker", "Side", "Score", "PositionSize", "Regime", "Note"])
    
    # Filtteröi regime-parametrien mukaan
    if 'score_long' in signals_df.columns:
        filtered = signals_df[signals_df['score_long'] >= params['min_ml_score']].copy()
    else:
        filtered = signals_df.copy()
    
    # Rajoita määrää
    filtered = filtered.head(params['max_positions'])
    
    if filtered.empty:
        print(f"[INFO] No signals above threshold ({params['min_ml_score']:.2f})")
        return pd.DataFrame([], columns=["Ticker", "Side", "Score", "PositionSize", "Regime", "Note"])
    
    # Luo trade-kandidaatit
    candidates = []
    base_position_size = 1000  # Base: $1000 per position
    
    for _, row in filtered.iterrows():
        position_size = base_position_size * params['position_size_multiplier']
        
        candidate = {
            'Ticker': row['ticker'],
            'Side': 'LONG',
            'Score': row.get('score_long', 0.0),
            'PositionSize': round(position_size, 2),
            'Regime': regime,
            'RegimeScore': regime_data['composite_score'],
            'Note': f"mom20={row.get('mom20', 0.0):.2%}, vol={row.get('vol20', 0.0):.2%}"
        }
        candidates.append(candidate)
    
    result_df = pd.DataFrame(candidates)
    print(f"[INFO] Generated {len(result_df)} trade candidates for {regime}")
    
    return result_df

# ============================================================================


def build_sell_candidates(args, portfolio_state, regime_data: dict) -> pd.DataFrame:
    """
    Rakenna myyntikandidaatit (stop-loss / take-profit)
    
    MUUTOS: Regime vaikuttaa stop/TP -tasoihin
    """
    regime = regime_data['regime']
    params = get_regime_params(regime)
    
    positions = portfolio_state.get('positions', [])
    if not positions:
        return pd.DataFrame([], columns=["Ticker", "Reason"])
    
    sells = []
    
    for pos in positions:
        ticker = pos.get('ticker', '')
        entry_price = pos.get('entry_price', 0.0)
        current_price = pos.get('current_price', entry_price)  # Pitäisi päivittää oikeasta datasta
        
        # Laske stop/TP regime-mukaisesti
        stop_distance = 0.05 * params['stop_loss_multiplier']  # 5% base stop
        tp_distance = 0.10 * params['take_profit_multiplier']   # 10% base TP
        
        stop_level = entry_price * (1 - stop_distance)
        tp_level = entry_price * (1 + tp_distance)
        
        # Tarkista triggerit
        if current_price <= stop_level:
            sells.append({'Ticker': ticker, 'Reason': f'STOP_LOSS (regime: {regime})'})
        elif current_price >= tp_level:
            sells.append({'Ticker': ticker, 'Reason': f'TAKE_PROFIT (regime: {regime})'})
    
    return pd.DataFrame(sells, columns=["Ticker", "Reason"])


def simulate_portfolio_after_trades(portfolio_state, buys_df, sells_df) -> pd.DataFrame:
    """
    Muodosta taulukko, joka tallennetaan portfolio_after_sim.csv:ksi.
    """
    positions = portfolio_state.get('positions', [])
    
    # Poista myydyt
    if not sells_df.empty:
        sell_tickers = set(sells_df['Ticker'].tolist())
        positions = [p for p in positions if p.get('ticker', '') not in sell_tickers]
    
    # Lisää uudet
    if not buys_df.empty:
        for _, row in buys_df.iterrows():
            positions.append({
                'ticker': row['Ticker'],
                'side': row['Side'],
                'position_size': row['PositionSize'],
                'regime': row.get('Regime', 'Unknown'),
                'entry_date': datetime.now().strftime("%Y-%m-%d")
            })
    
    # Muodosta DataFrame
    if not positions:
        return pd.DataFrame([], columns=["Ticker", "Side", "PositionSize", "Regime", "Comment"])
    
    rows = []
    for pos in positions:
        rows.append({
            'Ticker': pos.get('ticker', ''),
            'Side': pos.get('side', 'LONG'),
            'PositionSize': pos.get('position_size', 0.0),
            'Regime': pos.get('regime', 'Unknown'),
            'Comment': f"Entry: {pos.get('entry_date', 'N/A')}"
        })
    
    return pd.DataFrame(rows)


def write_action_plan(args, buys_df, sells_df, regime_data: dict):
    """
    Kirjoita action plan TXT-tiedostoon
    
    MUUTOS: Sisältää regime-tiedot
    """
    regime = regime_data['regime']
    
    plan = []
    plan.append(f"Date: {args.today}")
    plan.append(f"Regime: {regime} (score: {regime_data['composite_score']:.4f}, confidence: {regime_data['confidence']:.2%})")
    plan.append(f"Universe: {os.path.basename(args.universe_csv)}")
    plan.append(f"Run root: {args.run_root}")
    plan.append("")
    plan.append("=== Buys ===")
    if buys_df is not None and len(buys_df):
        for _, r in buys_df.iterrows():
            plan.append(f"- {r.get('Ticker','?')} (Score={r.get('Score',''):.2f}, Size=${r.get('PositionSize',0):.0f})")
    else:
        plan.append("(none)")
    plan.append("")
    plan.append("=== Sells ===")
    if sells_df is not None and len(sells_df):
        for _, r in sells_df.iterrows():
            plan.append(f"- {r.get('Ticker','?')} ({r.get('Reason','')})")
    else:
        plan.append("(none)")

    out = os.path.join(args.actions_dir, "action_plan.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(plan))
    print(f"[OK] Action Plan : {out}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()
    
    # ==================== LISÄYS: Lue regime ====================
    print("\n" + "="*80)
    print("[STEP] REGIME DETECTION")
    print("="*80)
    
    regime_data = get_latest_regime(args.project_root)
    
    print("="*80 + "\n")
    # ============================================================

    # Lataa/valmistele salkun tila
    state = _load_portfolio_state(args.portfolio_state)

    # Rakenna listat (nyt regime-mukautetuilla parametreilla)
    buys_df  = build_trade_candidates(args, regime_data)
    sells_df = build_sell_candidates(args, state, regime_data)
    port_df  = simulate_portfolio_after_trades(state, buys_df, sells_df)

    # Kirjoitukset
    trade_path = os.path.join(args.actions_dir, "trade_candidates.csv")
    sell_path  = os.path.join(args.actions_dir, "sell_candidates.csv")
    port_path  = os.path.join(args.actions_dir, "portfolio_after_sim.csv")

    _write_csv(buys_df, trade_path)
    print(f"[OK] Candidates : {trade_path}")

    _write_csv(sells_df, sell_path)
    print(f"[OK] Sells      : {sell_path}")

    _write_csv(port_df, port_path)
    print(f"[OK] Portfolio  : {port_path}")

    # Action plan (nyt sisältää regime-tiedot)
    write_action_plan(args, buys_df, sells_df, regime_data)

    # Tallennetaanko portfolio_state.json?
    if int(args.commit) == 1:
        # Päivitä state uusilla positioilla
        state['positions'] = port_df.to_dict('records') if not port_df.empty else []
        state['last_update'] = args.today
        state['regime'] = regime_data['regime']
        state['regime_score'] = regime_data['composite_score']
        
        _save_portfolio_state(state, args.portfolio_state)
        print(f"[OK] Portfolio state COMMITTED: {args.portfolio_state}")
    else:
        print("[INFO] Dry-run: portfolio_state.json ei päivitetty (commit=0).")


if __name__ == "__main__":
    main()