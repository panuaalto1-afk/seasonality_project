#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_decider.py
===============
Automated trade decision maker for seasonality project.

Analyzes ML-generated signals, applies regime-specific strategies,
and generates actionable trade recommendations with Stop Loss & Take Profit levels.

Usage:
    python auto_decider.py --project_root "." --universe_csv "..." --run_root "..." --price_cache_dir "..." --today "2025-11-06" --commit 0

Features:
- Regime-aware strategy selection
- Sector rotation filtering
- Position sizing based on regime
- ATR-based Stop Loss & Take Profit calculation
- Automatic email notifications
"""

import argparse
import os
import sys
import json
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

# Regime system imports
try:
    from regime_detector import RegimeDetector
    from regime_strategies import RegimeStrategy
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    print("[WARN] Regime modules not available. Running in basic mode.")

# ======================== ARGUMENT PARSING ========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Automated trade decision maker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--project_root", type=str, required=True,
                   help="Project root directory")
    p.add_argument("--universe_csv", type=str, required=True,
                   help="Path to universe CSV (constituents)")
    p.add_argument("--run_root", type=str, required=True,
                   help="Run root directory (contains reports/)")
    p.add_argument("--price_cache_dir", type=str, required=True,
                   help="Price cache directory")
    p.add_argument("--today", type=str, default=None,
                   help="Date YYYY-MM-DD (defaults to today)")
    p.add_argument("--commit", type=int, default=0, choices=[0, 1],
                   help="0=dry-run, 1=commit changes to portfolio_state.json")
    p.add_argument("--max_positions", type=int, default=8,
                   help="Maximum number of open positions")
    p.add_argument("--position_size", type=float, default=1000.0,
                   help="Base position size in dollars")
    p.add_argument("--no_new_positions", action="store_true",
                   help="Prevent opening new positions (exit-only mode)")
    return p.parse_args()

# ======================== HELPER FUNCTIONS ========================

def _as_date(s: Optional[str]) -> date:
    if s is None:
        return date.today()
    return datetime.strptime(s, "%Y-%m-%d").date()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _find_latest_gated_csv(run_root: str, d: date) -> Optional[str]:
    """Find latest top_long_candidates_GATED_*.csv"""
    tag = d.strftime("%Y-%m-%d")
    reports_dir = os.path.join(run_root, "reports")
    
    pattern = f"top_long_candidates_GATED_{tag}.csv"
    path = os.path.join(reports_dir, pattern)
    
    if os.path.isfile(path):
        return path
    
    # Fallback: try without GATED
    pattern2 = f"top_long_candidates_RAW_{tag}.csv"
    path2 = os.path.join(reports_dir, pattern2)
    
    if os.path.isfile(path2):
        print(f"[WARN] Using RAW candidates (GATED not found)")
        return path2
    
    return None

def load_portfolio_state(project_root: str) -> Dict:
    """Load current portfolio state"""
    path = os.path.join(project_root, "seasonality_reports", "portfolio_state.json")
    
    if not os.path.exists(path):
        print(f"[INFO] No existing portfolio_state.json, starting fresh")
        return {"positions": {}, "cash": 100000.0, "last_updated": None}
    
    try:
        with open(path, 'r') as f:
            state = json.load(f)
            
            # Varmista että positions on dict
            if not isinstance(state.get('positions'), dict):
                print(f"[WARN] portfolio_state.json positions was not a dict, resetting to empty dict")
                state['positions'] = {}
            
            return state
    except Exception as e:
        print(f"[WARN] Failed to load portfolio_state.json: {e}")
        return {"positions": {}, "cash": 100000.0, "last_updated": None}

def save_portfolio_state(project_root: str, state: Dict):
    """Save portfolio state"""
    path = os.path.join(project_root, "seasonality_reports", "portfolio_state.json")
    
    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"[OK] Saved portfolio_state.json")
    except Exception as e:
        print(f"[ERROR] Failed to save portfolio_state.json: {e}")

def read_candidates(csv_path: str) -> pd.DataFrame:
    """Read candidate signals from CSV"""
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure required columns
        required = ["ticker"]
        for col in required:
            if col not in df.columns:
                print(f"[ERROR] Missing required column: {col}")
                return pd.DataFrame()
        
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_path}: {e}")
        return pd.DataFrame()

# ======================== PRICE & ATR FUNCTIONS ========================

def read_price_data(ticker: str, price_cache_dir: str) -> Optional[pd.DataFrame]:
    """Read price data for a ticker from price_cache"""
    ticker = ticker.upper().strip()
    price_file = os.path.join(price_cache_dir, f"{ticker}.csv")
    
    if not os.path.exists(price_file):
        return None
    
    try:
        df = pd.read_csv(price_file)
        df.columns = [c.lower().strip() for c in df.columns]
        if 'adj_close' in df.columns:
            df['close'] = df['adj_close']
        return df
    except Exception as e:
        return None

def calculate_atr(price_df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Calculate Average True Range (ATR)"""
    if price_df is None or len(price_df) < period + 1:
        return None
    
    try:
        high = price_df['high'].values
        low = price_df['low'].values
        close = price_df['close'].values
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:])
        
        return float(atr)
    except Exception as e:
        return None

def enrich_with_stop_tp(df: pd.DataFrame, price_cache_dir: str, is_buy: bool = True) -> pd.DataFrame:
    """Add Stop/TP levels to dataframe"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Add columns
    if is_buy:
        df['EntryPrice'] = np.nan
    else:
        df['CurrentPrice'] = np.nan
    
    df['ATR'] = np.nan
    df['StopLoss'] = np.nan
    df['TakeProfit'] = np.nan
    
    print(f"\n[STOP/TP] Calculating for {len(df)} {'buy' if is_buy else 'sell'} candidates...")
    
    for idx, row in df.iterrows():
        ticker = str(row.get('ticker', row.get('Ticker', ''))).strip().upper()
        if not ticker or ticker == 'NAN':
            continue
        
        price_df = read_price_data(ticker, price_cache_dir)
        if price_df is None or price_df.empty:
            print(f"  ⚠️  {ticker}: No price data")
            continue
        
        current_price = float(price_df['close'].iloc[-1])
        atr = calculate_atr(price_df, period=14)
        
        if atr is None or atr == 0:
            print(f"  ⚠️  {ticker}: ATR calculation failed")
            continue
        
        # For BUY: use current price as entry
        # For SELL: use entry_price from row (or current as fallback)
        if is_buy:
            entry_price = current_price
            df.at[idx, 'EntryPrice'] = round(entry_price, 2)
        else:
            entry_price = row.get('EntryPrice', current_price)
            df.at[idx, 'CurrentPrice'] = round(current_price, 2)
        
        stop_loss = entry_price - (1.0 * atr)
        take_profit = entry_price + (3.0 * atr)
        
        df.at[idx, 'ATR'] = round(atr, 2)
        df.at[idx, 'StopLoss'] = round(stop_loss, 2)
        df.at[idx, 'TakeProfit'] = round(take_profit, 2)
        
        price_str = f"Entry=${entry_price:>7.2f}" if is_buy else f"Current=${current_price:>7.2f}"
        print(f"  ✅ {ticker:6} {price_str}  ATR=${atr:>5.2f}  SL=${stop_loss:>7.2f}  TP=${take_profit:>7.2f}")
    
    return df

# ======================== REGIME INTEGRATION ========================

def detect_current_regime(project_root: str, today: date) -> Dict:
    """Detect current market regime"""
    
    if not REGIME_AVAILABLE:
        return {
            'regime': 'NEUTRAL_BULLISH',
            'composite_score': 0.0,
            'confidence': 0.0
        }
    
    try:
        detector = RegimeDetector(
            macro_price_cache_dir=os.path.join(project_root, "seasonality_reports", "price_cache")
        )
        
        regime_data = detector.detect_regime(date=today.strftime("%Y-%m-%d"))
        
        print(f"\n[REGIME] Current: {regime_data['regime']}")
        print(f"[REGIME] Score: {regime_data['composite_score']:.4f}")
        print(f"[REGIME] Confidence: {regime_data['confidence']:.1%}")
        
        return regime_data
        
    except Exception as e:
        print(f"[WARN] Regime detection failed: {e}")
        return {
            'regime': 'NEUTRAL_BULLISH',
            'composite_score': 0.0,
            'confidence': 0.0
        }

def apply_regime_strategy(candidates_df: pd.DataFrame, regime_data: Dict, max_positions: int, position_size: float) -> pd.DataFrame:
    """Apply regime-specific strategy to filter and rank candidates"""
    
    if candidates_df.empty or not REGIME_AVAILABLE:
        return candidates_df
    
    regime = regime_data.get('regime', 'NEUTRAL_BULLISH')
    
    try:
        strategy = RegimeStrategy(regime)
        
        print(f"\n[STRATEGY] Applying {regime} strategy ({strategy.config['strategy_type']})")
        print(f"[STRATEGY] Max positions: {strategy.config['max_positions']}")
        
        # Position sizing can be either a dict or a float
        position_sizing = strategy.config.get('position_sizing', 1.0)
        if isinstance(position_sizing, dict):
            position_multiplier = position_sizing.get('base_multiplier', 1.0)
        else:
            position_multiplier = float(position_sizing)
        
        print(f"[STRATEGY] Position size multiplier: {position_multiplier:.1f}x")
        
        # Kandidaatit on jo filtteröity ml_unified_pipeline:ssa
        filtered = candidates_df.copy()
        
        print(f"[STRATEGY] Using {len(filtered)} pre-filtered candidates")
        
        # Adjust position size based on regime
        adjusted_size = position_size * position_multiplier
        filtered['adjusted_position_size'] = adjusted_size
        
        # Use regime's max positions if lower than user's setting
        regime_max = strategy.config['max_positions']
        if regime_max < max_positions:
            print(f"[STRATEGY] Reducing max positions: {max_positions} → {regime_max} (regime constraint)")
            filtered = filtered.head(regime_max)
        else:
            filtered = filtered.head(max_positions)
        
        return filtered
        
    except Exception as e:
        print(f"[WARN] Regime strategy application failed: {e}")
        import traceback
        traceback.print_exc()
        return candidates_df.head(max_positions)

# ======================== DECISION LOGIC ========================

def decide_trades(
    candidates_df: pd.DataFrame,
    portfolio_state: Dict,
    regime_data: Dict,
    max_positions: int,
    position_size: float,
    no_new_positions: bool
) -> Dict:
    """
    Main decision logic: determine which trades to make
    
    Returns:
        {
            'buy': [...],
            'sell': [...],
            'hold': [...],
            'reason': {...}
        }
    """
    
    decisions = {
        'buy': [],
        'sell': [],
        'hold': [],
        'reason': {}
    }
    
    current_positions = portfolio_state.get('positions', {})
    if not isinstance(current_positions, dict):
        print(f"[WARN] portfolio_state['positions'] was not a dict, treating as empty")
        current_positions = {}
    
    current_tickers = set(current_positions.keys())
    
    # If CRISIS or no_new_positions → exit all
    regime = regime_data.get('regime', 'NEUTRAL_BULLISH')
    if regime == 'CRISIS' or no_new_positions:
        print(f"\n[DECISION] {'CRISIS MODE' if regime == 'CRISIS' else 'NO NEW POSITIONS MODE'}")
        print(f"[DECISION] Exiting all {len(current_tickers)} positions")
        
        decisions['sell'] = list(current_tickers)
        for ticker in current_tickers:
            decisions['reason'][ticker] = 'CRISIS' if regime == 'CRISIS' else 'NO_NEW_POSITIONS'
        
        return decisions
    
    # Get candidate tickers
    if candidates_df.empty:
        print(f"\n[DECISION] No candidates available")
        decisions['hold'] = list(current_tickers)
        return decisions
    
    candidate_tickers = set(candidates_df['ticker'].tolist()[:max_positions])
    
    # Decide sells
    to_sell = current_tickers - candidate_tickers
    decisions['sell'] = list(to_sell)
    
    for ticker in to_sell:
        decisions['reason'][ticker] = 'NOT_IN_TOP_CANDIDATES'
    
    # Decide holds
    to_hold = current_tickers & candidate_tickers
    decisions['hold'] = list(to_hold)
    
    # Decide buys
    open_slots = max_positions - len(to_hold)
    potential_buys = candidate_tickers - current_tickers
    to_buy = list(potential_buys)[:open_slots]
    
    decisions['buy'] = to_buy
    
    for ticker in to_buy:
        decisions['reason'][ticker] = 'NEW_CANDIDATE'
    
    print(f"\n[DECISION] Summary:")
    print(f"  Buy:  {len(decisions['buy'])} positions")
    print(f"  Sell: {len(decisions['sell'])} positions")
    print(f"  Hold: {len(decisions['hold'])} positions")
    
    return decisions

# ======================== OUTPUT GENERATION ========================

def generate_outputs(
    decisions: Dict,
    candidates_df: pd.DataFrame,
    portfolio_state: Dict,
    regime_data: Dict,
    run_root: str,
    today: date,
    position_size: float,
    price_cache_dir: str
):
    """Generate output files: trade_candidates.csv, sell_candidates.csv, action_plan.txt"""
    
    tag = today.strftime("%Y%m%d")
    actions_dir = os.path.join(run_root, "actions", tag)
    _ensure_dir(actions_dir)
    
    # 1. Trade candidates (BUY)
    buy_tickers = decisions['buy']
    
    if buy_tickers and not candidates_df.empty:
        buy_df = candidates_df[candidates_df['ticker'].isin(buy_tickers)].copy()
        
        # Enrich with Stop/TP
        buy_df = enrich_with_stop_tp(buy_df, price_cache_dir, is_buy=True)
        
        # Add metadata
        buy_df['Side'] = 'LONG'
        buy_df['Note'] = buy_df['ticker'].map(decisions['reason'])
        
        # Position size
        if 'adjusted_position_size' in buy_df.columns:
            buy_df['PositionSize'] = buy_df['adjusted_position_size']
        else:
            buy_df['PositionSize'] = position_size
        
        # Regime info
        buy_df['Regime'] = regime_data.get('regime', 'Unknown')
        buy_df['RegimeScore'] = regime_data.get('composite_score', 0.0)
        
        # Save
        buy_path = os.path.join(actions_dir, "trade_candidates.csv")
        buy_df.to_csv(buy_path, index=False)
        print(f"[OK] Saved: {buy_path}")
    else:
        # Empty file
        pd.DataFrame(columns=['Ticker', 'Side', 'Note']).to_csv(
            os.path.join(actions_dir, "trade_candidates.csv"), index=False
        )
        print(f"[OK] No buy candidates")
    
    # 2. Sell candidates
    sell_tickers = decisions['sell']
    
    if sell_tickers:
        sell_data = []
        current_positions = portfolio_state.get('positions', {})
        if not isinstance(current_positions, dict):
            current_positions = {}
        
        for ticker in sell_tickers:
            pos = current_positions.get(ticker, {})
            sell_data.append({
                'Ticker': ticker,
                'Side': 'SELL',
                'EntryPrice': pos.get('entry_price', 0.0),
                'Quantity': pos.get('quantity', 0),
                'Reason': decisions['reason'].get(ticker, 'Unknown')
            })
        
        sell_df = pd.DataFrame(sell_data)
        
        # Enrich with Stop/TP
        sell_df = enrich_with_stop_tp(sell_df, price_cache_dir, is_buy=False)
        
        sell_path = os.path.join(actions_dir, "sell_candidates.csv")
        sell_df.to_csv(sell_path, index=False)
        print(f"[OK] Saved: {sell_path}")
    else:
        # Empty file
        pd.DataFrame(columns=['Ticker', 'Side', 'Reason']).to_csv(
            os.path.join(actions_dir, "sell_candidates.csv"), index=False
        )
        print(f"[OK] No sell candidates")
    
    # 3. Action plan (text summary)
    action_plan_path = os.path.join(actions_dir, "action_plan.txt")
    
    with open(action_plan_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"TRADE ACTION PLAN - {today.strftime('%Y-%m-%d')}\n")
        f.write("="*80 + "\n\n")
        
        # Regime info
        f.write("MARKET REGIME:\n")
        f.write(f"  Current: {regime_data.get('regime', 'Unknown')}\n")
        f.write(f"  Score: {regime_data.get('composite_score', 0.0):+.4f}\n")
        f.write(f"  Confidence: {regime_data.get('confidence', 0.0):.1%}\n\n")
        
        # Multi-timeframe (if available)
        if 'multi_timeframe' in regime_data:
            mtf = regime_data['multi_timeframe']
            f.write("MULTI-TIMEFRAME:\n")
            f.write(f"  Daily: {mtf.get('daily', 'Unknown')}\n")
            f.write(f"  Weekly: {mtf.get('weekly', 'Unknown')}\n")
            f.write(f"  Monthly: {mtf.get('monthly', 'Unknown')}\n")
            f.write(f"  Alignment: {mtf.get('alignment', 'Unknown')}\n")
            f.write(f"  Bias: {mtf.get('bias', 'Unknown').upper()} ({mtf.get('bias_strength', 0.0):.1%})\n\n")
        
        # Actions
        f.write("ACTIONS:\n")
        f.write(f"  BUY:  {len(decisions['buy'])} positions\n")
        f.write(f"  SELL: {len(decisions['sell'])} positions\n")
        f.write(f"  HOLD: {len(decisions['hold'])} positions\n\n")
        
        # Details
        if decisions['buy']:
            f.write("BUY ORDERS:\n")
            for ticker in decisions['buy']:
                f.write(f"  - {ticker} (${position_size:.0f})\n")
            f.write("\n")
        
        if decisions['sell']:
            f.write("SELL ORDERS:\n")
            for ticker in decisions['sell']:
                reason = decisions['reason'].get(ticker, 'Unknown')
                f.write(f"  - {ticker} ({reason})\n")
            f.write("\n")
        
        if decisions['hold']:
            f.write("HOLD POSITIONS:\n")
            for ticker in decisions['hold']:
                f.write(f"  - {ticker}\n")
            f.write("\n")
        
        # Portfolio summary
        f.write("="*80 + "\n")
        f.write("PORTFOLIO AFTER ACTIONS:\n")
        f.write("="*80 + "\n")
        
        total_positions = len(decisions['hold']) + len(decisions['buy'])
        f.write(f"Total Positions: {total_positions}\n")
        f.write(f"Cash: ${portfolio_state.get('cash', 0.0):.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"[OK] Saved: {action_plan_path}")
    
    return actions_dir

# ======================== MAIN ========================

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("AUTO DECIDER - Regime-Aware Trade Decision System")
    print("="*80 + "\n")
    
    # Parse date
    today = _as_date(args.today)
    print(f"[INFO] Today: {today.strftime('%Y-%m-%d')}")
    print(f"[INFO] Commit mode: {'LIVE' if args.commit else 'DRY-RUN'}")
    
    # Load portfolio
    portfolio_state = load_portfolio_state(args.project_root)
    print(f"[INFO] Current positions: {len(portfolio_state['positions'])}")
    print(f"[INFO] Cash: ${portfolio_state.get('cash', 0.0):.2f}")
    
    # Detect regime
    regime_data = detect_current_regime(args.project_root, today)
    
    # Check for CRISIS or no_new_positions
    if regime_data.get('regime') == 'CRISIS':
        print(f"\n⚠️  WARNING: CRISIS REGIME DETECTED")
        print(f"⚠️  Enabling capital preservation mode (exit all positions)")
        args.no_new_positions = True
    
    if args.no_new_positions:
        print(f"\n[INFO] NO NEW POSITIONS mode enabled")
    
    # Find latest candidates
    gated_csv = _find_latest_gated_csv(args.run_root, today)
    
    if gated_csv is None:
        print(f"\n[ERROR] No candidates CSV found for {today.strftime('%Y-%m-%d')}")
        print(f"[ERROR] Expected: {args.run_root}/reports/top_long_candidates_GATED_{today.strftime('%Y-%m-%d')}.csv")
        sys.exit(1)
    
    print(f"\n[INFO] Loading candidates: {gated_csv}")
    candidates_df = read_candidates(gated_csv)
    
    if candidates_df.empty:
        print(f"[WARN] No candidates found")
    else:
        print(f"[INFO] Loaded {len(candidates_df)} candidates")
    
    # Apply regime strategy
    filtered_candidates = apply_regime_strategy(
        candidates_df,
        regime_data,
        args.max_positions,
        args.position_size
    )
    
    # Make decisions
    decisions = decide_trades(
        filtered_candidates,
        portfolio_state,
        regime_data,
        args.max_positions,
        args.position_size,
        args.no_new_positions
    )
    
    # Generate outputs
    actions_dir = generate_outputs(
        decisions,
        filtered_candidates,
        portfolio_state,
        regime_data,
        args.run_root,
        today,
        args.position_size,
        args.price_cache_dir
    )
    
    # Simulate portfolio update
    print(f"\n[INFO] Simulating portfolio after actions...")
    
    simulated_portfolio = portfolio_state.copy()
    
    if not isinstance(simulated_portfolio.get('positions'), dict):
        simulated_portfolio['positions'] = {}
    
    # Remove sells
    for ticker in decisions['sell']:
        if ticker in simulated_portfolio['positions']:
            del simulated_portfolio['positions'][ticker]
    
    # Add buys (simulated)
    for ticker in decisions['buy']:
        simulated_portfolio['positions'][ticker] = {
            'entry_date': today.strftime('%Y-%m-%d'),
            'entry_price': 100.0,  # Placeholder
            'quantity': int(args.position_size / 100.0),
            'regime_at_entry': regime_data.get('regime', 'Unknown')
        }
    
    # Save simulated portfolio
    sim_path = os.path.join(actions_dir, "portfolio_after_sim.csv")
    sim_df = pd.DataFrame([
        {'Ticker': ticker, **data}
        for ticker, data in simulated_portfolio['positions'].items()
    ])
    sim_df.to_csv(sim_path, index=False)
    print(f"[OK] Saved simulated portfolio: {sim_path}")
    
    # Commit changes?
    if args.commit:
        print(f"\n[INFO] Committing changes to portfolio_state.json...")
        simulated_portfolio['last_updated'] = today.strftime('%Y-%m-%d')
        save_portfolio_state(args.project_root, simulated_portfolio)
    else:
        print(f"\n[INFO] Dry-run: portfolio_state.json ei päivitetty (commit=0).")
    
    print("\n" + "="*80)
    print("AUTO DECIDER COMPLETE")
    print("="*80 + "\n")
    
    # Email integration
    try:
        print("[INFO] Lähetetään trade candidates emaililla...")
        import send_trades_email
        send_trades_email.main()
    except Exception as e:
        print(f"[WARN] Email-lähetys epäonnistui: {e}")
        print("[INFO] Voit lähettää manuaalisesti: python send_trades_email.py")

if __name__ == "__main__":
    main()