#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_decider.py v4.3.2
====================
Automated trade decision maker for seasonality project.

Features:
- Regime-aware strategy selection
- Position sizing based on regime
- Portfolio status with REAL PRICES, SL/TP, ATR from price_cache
- Inverse ETF system (SH, PSQ, DOG, RWM)
- CRISIS mode: 80% inverse ETF allocation
- AUTOMATIC EMAIL NOTIFICATIONS (integrated)

FIX v4.3.2 (2025-11-07):
- Fixed price fetch to use previous day's close (T-1) for T morning decisions
- Column normalization handles "Adj Close" with space properly
- ATR calculation uses 14-day rolling average (industry standard)
- No placeholders - uses REAL prices from static price_cache
- Price cache updates via overwrite (single directory with 20y history)
"""

import argparse
import os
import sys
import json
import glob
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from pathlib import Path

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

# Load .env for email credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ======================== INVERSE ETF SYSTEM ========================

def get_inverse_etf_universe(regime: str) -> List[str]:
    """
    Palauttaa inverse ETF:t regimen mukaan
    
    CRISIS: 80% allokaatio -> SH, PSQ, DOG, RWM
    BEAR_STRONG: 60% allokaatio -> SH, PSQ
    BEAR_WEAK: 40% allokaatio -> SH
    NEUTRAL_BEARISH: 20% allokaatio -> SH
    """
    inverse_etfs = {
        'CRISIS': ['SH', 'PSQ', 'DOG', 'RWM'],
        'BEAR_STRONG': ['SH', 'PSQ'],
        'BEAR_WEAK': ['SH'],
        'NEUTRAL_BEARISH': ['SH'],
    }
    
    return inverse_etfs.get(regime, [])

def get_inverse_allocation_pct(regime: str) -> float:
    """Palauttaa inverse ETF allokaatio-% regimen mukaan"""
    allocation_pct = {
        'CRISIS': 0.80,
        'BEAR_STRONG': 0.60,
        'BEAR_WEAK': 0.40,
        'NEUTRAL_BEARISH': 0.20,
    }
    return allocation_pct.get(regime, 0.0)

def calculate_inverse_allocation(regime: str, total_portfolio_value: float) -> Dict[str, float]:
    """
    Laske inverse ETF allokaatio regimen mukaan
    
    Returns:
        dict: {ticker: dollar_amount}
    """
    pct = get_inverse_allocation_pct(regime)
    if pct == 0:
        return {}
    
    inverse_etfs = get_inverse_etf_universe(regime)
    if not inverse_etfs:
        return {}
    
    total_inverse = total_portfolio_value * pct
    amount_per_etf = total_inverse / len(inverse_etfs)
    
    return {ticker: amount_per_etf for ticker in inverse_etfs}

def calculate_portfolio_value(portfolio_state: Dict) -> float:
    """Calculate total portfolio value (positions + cash)"""
    cash = portfolio_state.get('cash', 0.0)
    
    positions = portfolio_state.get('positions', {})
    if not isinstance(positions, dict):
        positions = {}
    
    position_value = 0.0
    for ticker, pos in positions.items():
        entry_price = pos.get('entry_price', 0.0)
        qty = pos.get('quantity', 0)
        position_value += entry_price * qty
    
    return cash + position_value

# ======================== PRICE & SL/TP CALCULATION (FIXED v4.3.2) ========================

def _fetch_price_from_cache(ticker: str, price_cache_dir: str, lookback: int = 60) -> tuple:
    """
    Fetch last close price and ATR14 from price cache.
    
    Uses T-1 close for T morning decisions (previous day's close).
    Price cache contains ~20 years of history, updated daily via overwrite.
    
    Args:
        ticker: Stock symbol
        price_cache_dir: Static price cache directory (e.g., .../2025-10-04_0903/price_cache)
        lookback: Number of days to load for ATR calculation
    
    Returns:
        (last_close, atr14) or (None, None) if not found
    """
    ticker_upper = ticker.upper()
    
    # Price cache contains files like: DDOG.csv, AAPL.csv, etc.
    csv_path = os.path.join(price_cache_dir, f"{ticker_upper}.csv")
    
    if not os.path.exists(csv_path):
        return (None, None)
    
    try:
        # Read CSV with full history
        df = pd.read_csv(csv_path)
        
        # Normalize column names: "Adj Close" -> "adj_close"
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Find close column
        close_col = None
        for col_name in ['adj_close', 'adjclose', 'close']:
            if col_name in df.columns:
                close_col = col_name
                break
        
        if close_col is None:
            return (None, None)
        
        # Get last N rows for ATR calculation
        df_recent = df.tail(lookback).copy()
        
        if df_recent.empty or df_recent[close_col].dropna().empty:
            return (None, None)
        
        # Last available close (T-1 for T morning decision)
        last_close = float(df_recent[close_col].dropna().iloc[-1])
        
        # Calculate ATR(14) - industry standard
        atr = 0.0
        if all(col in df_recent.columns for col in ['high', 'low']) and len(df_recent) >= 14:
            prev_close = df_recent[close_col].shift(1)
            
            # True Range = max(H-L, H-C_prev, C_prev-L)
            tr1 = df_recent['high'] - df_recent['low']
            tr2 = (df_recent['high'] - prev_close).abs()
            tr3 = (df_recent['low'] - prev_close).abs()
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR(14) = 14-day rolling average of True Range
            atr_series = true_range.rolling(window=14, min_periods=14).mean()
            
            if not atr_series.dropna().empty:
                atr = float(atr_series.iloc[-1])
        
        return (last_close, atr)
        
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_path}: {e}")
        return (None, None)


def load_current_prices(price_cache_dir: str, tickers: List[str], today: date) -> Dict[str, float]:
    """
    Lataa tämän päivän (T) aamun hinnat price_cache:sta.
    
    Käyttää T-1 closing prices (edellisen päivän päätöskursseja),
    koska päätökset tehdään ennen T:n markkinoiden aukeamista.
    
    Args:
        price_cache_dir: Static price cache directory
        tickers: List of tickers
        today: Date (T) - not used directly, but context for T-1 logic
    
    Returns:
        dict: {ticker: close_price}
    """
    prices = {}
    
    for ticker in tickers:
        last_close, _ = _fetch_price_from_cache(ticker, price_cache_dir)
        
        if last_close is not None:
            prices[ticker] = last_close
        else:
            print(f"[WARN] No price data found for {ticker} in price_cache")
    
    return prices


def calculate_atr(price_cache_dir: str, ticker: str, period: int = 14) -> float:
    """
    Laske ATR(14) - Average True Range.
    
    Industry standard: 14-day rolling average of True Range.
    Used for stop-loss and position sizing.
    
    Args:
        price_cache_dir: Static price cache directory
        ticker: Stock symbol
        period: ATR period (default 14)
    
    Returns:
        float: ATR value
    """
    _, atr = _fetch_price_from_cache(ticker, price_cache_dir, lookback=period + 20)
    
    if atr is not None and atr > 0:
        return atr
    
    # Fallback: return 0 if ATR calculation fails
    return 0.0


def calculate_sl_tp(entry_price: float, atr: float, sl_multiplier: float = 2.0, tp_multiplier: float = 3.0) -> Dict[str, float]:
    """
    Laske Stop Loss ja Take Profit tasot ATR:n perusteella.
    
    Args:
        entry_price: Entry price
        atr: Average True Range
        sl_multiplier: Stop loss multiplier (default 2.0x ATR)
        tp_multiplier: Take profit multiplier (default 3.0x ATR)
    
    Returns:
        dict: {'sl': stop_loss, 'tp': take_profit}
    """
    if atr > 0:
        sl = entry_price - (sl_multiplier * atr)
        tp = entry_price + (tp_multiplier * atr)
    else:
        # Fallback: percentage-based if ATR not available
        sl = entry_price * 0.95
        tp = entry_price * 1.10
    
    return {
        'sl': max(sl, 0.01),  # Ensure SL is positive
        'tp': tp
    }

def enrich_portfolio_with_prices(
    portfolio_state: Dict,
    price_cache_dir: str,
    today: date
) -> pd.DataFrame:
    """
    Rikastuta portfolio hinnoilla, SL/TP tasoilla ja P/L%:lla.
    
    Uses T-1 closing prices for current valuation.
    """
    positions = portfolio_state.get('positions', {})
    if not isinstance(positions, dict):
        positions = {}
    
    if not positions:
        return pd.DataFrame()
    
    tickers = list(positions.keys())
    current_prices = load_current_prices(price_cache_dir, tickers, today)
    
    rows = []
    for ticker, pos in positions.items():
        entry_price = pos.get('entry_price', 0.0)
        qty = pos.get('quantity', 0)
        entry_date_str = pos.get('entry_date', '?')
        regime = pos.get('regime_at_entry', '?')
        
        current_price = current_prices.get(ticker, entry_price)
        atr = calculate_atr(price_cache_dir, ticker)
        sl_tp = calculate_sl_tp(current_price if current_price > 0 else entry_price, atr)
        
        if entry_price > 0:
            pl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pl_pct = 0.0
        
        try:
            entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d').date()
            days_held = (today - entry_date).days
        except:
            days_held = 0
        
        current_value = current_price * qty
        
        rows.append({
            'Ticker': ticker,
            'Entry_Date': entry_date_str,
            'Entry_Price': entry_price,
            'Current_Price': current_price,
            'Quantity': qty,
            'Value': current_value,
            'ATR': atr,
            'SL': sl_tp['sl'],
            'TP': sl_tp['tp'],
            'PL_Pct': pl_pct,
            'Regime': regime,
            'Days_Held': days_held
        })
    
    return pd.DataFrame(rows)

# ======================== EMAIL SYSTEM (INTEGRATED) ========================

def send_email_notification(actions_dir: str, action_plan_path: str) -> bool:
    """
    Send email with trade recommendations
    
    Reads EMAIL_USER, EMAIL_APP_PASSWORD, EMAIL_RECIPIENT from environment (.env)
    
    Returns:
        True if successful, False otherwise
    """
    
    # Get credentials from environment
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_APP_PASSWORD")
    recipient = os.getenv("EMAIL_RECIPIENT", sender)
    
    if not sender or not password:
        print("[WARN] Email credentials not found in .env - skipping email")
        print("[INFO] To enable email: create .env with EMAIL_USER and EMAIL_APP_PASSWORD")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = f"📈 Trade Recommendations - {date.today().strftime('%Y-%m-%d')}"
        
        # Read action plan for body
        try:
            with open(action_plan_path, 'r', encoding='utf-8') as f:
                action_plan_text = f.read()
        except:
            action_plan_text = "(Failed to read action plan)"
        
        # Plain text body
        text_body = f"""
Daily Trade Recommendations - {date.today().strftime('%Y-%m-%d')}
{'='*80}

Attached files:
  - trade_candidates.csv (BUY orders with Entry, SL, TP)
  - sell_candidates.csv (SELL orders with P/L%)
  - action_plan.txt (Full portfolio summary)
  - portfolio_after_sim.csv (Expected portfolio)

Action Plan Preview:
{'-'*80}
{action_plan_text[:1500]}
{'...(see attachment for full plan)' if len(action_plan_text) > 1500 else ''}

{'='*80}
Automated by seasonality_project/auto_decider.py v4.3.2
"""
        
        # HTML body
        html_body = f"""
<html>
<head>
<style>
    body {{ font-family: 'Courier New', monospace; background: #f5f5f5; }}
    .container {{ max-width: 900px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; }}
    .header {{ background: #2c3e50; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    .preview {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-size: 11px; overflow-x: auto; white-space: pre-wrap; }}
    .footer {{ margin-top: 20px; color: #7f8c8d; font-size: 11px; border-top: 1px solid #ecf0f1; padding-top: 10px; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h2>📈 Trade Recommendations</h2>
        <p>{date.today().strftime('%A, %B %d, %Y')}</p>
    </div>
    
    <h3>📎 Attached Files</h3>
    <ul>
        <li>trade_candidates.csv (BUY orders)</li>
        <li>sell_candidates.csv (SELL orders)</li>
        <li>action_plan.txt (Full summary)</li>
        <li>portfolio_after_sim.csv (After trades)</li>
    </ul>
    
    <h3>📋 Action Plan Preview</h3>
    <div class="preview">{action_plan_text[:1500]}</div>
    {'<p><i>...(see attachment for full plan)</i></p>' if len(action_plan_text) > 1500 else ''}
    
    <div class="footer">
        <p><b>Automated by seasonality_project/auto_decider.py v4.3.2</b></p>
        <p>GitHub: <a href="https://github.com/panuaalto1-afk/seasonality_project">panuaalto1-afk/seasonality_project</a></p>
    </div>
</div>
</body>
</html>
"""
        
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach files
        files_to_attach = [
            "trade_candidates.csv",
            "sell_candidates.csv",
            "action_plan.txt",
            "portfolio_after_sim.csv"
        ]
        
        for filename in files_to_attach:
            filepath = Path(actions_dir) / filename
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename={filename}')
                        msg.attach(part)
                except Exception as e:
                    print(f"[WARN] Failed to attach {filename}: {e}")
        
        # Send via Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        
        print(f"[OK] Email sent to {recipient}")
        return True
    
    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Email authentication failed - check EMAIL_APP_PASSWORD in .env")
        return False
    
    except Exception as e:
        print(f"[WARN] Email sending failed: {e}")
        return False

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
                   help="Price cache directory (static, ~20y history)")
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
    p.add_argument("--no_email", action="store_true",
                   help="Skip email notification")
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
        
        required = ["ticker"]
        for col in required:
            if col not in df.columns:
                print(f"[ERROR] Missing required column: {col}")
                return pd.DataFrame()
        
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_path}: {e}")
        return pd.DataFrame()

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
        
        position_sizing = strategy.config.get('position_sizing', 1.0)
        if isinstance(position_sizing, dict):
            position_multiplier = position_sizing.get('base_multiplier', 1.0)
        else:
            position_multiplier = float(position_sizing)
        
        print(f"[STRATEGY] Position size multiplier: {position_multiplier:.1f}x")
        
        filtered = candidates_df.copy()
        
        print(f"[STRATEGY] Using {len(filtered)} pre-filtered candidates")
        
        adjusted_size = position_size * position_multiplier
        filtered['adjusted_position_size'] = adjusted_size
        
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
    Main decision logic with inverse ETF support
    """
    
    decisions = {
        'buy': [],
        'sell': [],
        'hold': [],
        'reason': {},
        'inverse_etfs': {}
    }
    
    current_positions = portfolio_state.get('positions', {})
    if not isinstance(current_positions, dict):
        print(f"[WARN] portfolio_state['positions'] was not a dict, treating as empty")
        current_positions = {}
    
    current_tickers = set(current_positions.keys())
    regime = regime_data.get('regime', 'NEUTRAL_BULLISH')
    
    # Inverse ETF logic
    inverse_allocation_pct = get_inverse_allocation_pct(regime)
    
    if inverse_allocation_pct > 0:
        print(f"\n[INVERSE ETF] {regime} requires {inverse_allocation_pct:.0%} inverse allocation")
        
        portfolio_value = calculate_portfolio_value(portfolio_state)
        inverse_allocations = calculate_inverse_allocation(regime, portfolio_value)
        decisions['inverse_etfs'] = inverse_allocations
        
        print(f"[INVERSE ETF] Portfolio value: ${portfolio_value:,.2f}")
        for ticker, amount in inverse_allocations.items():
            print(f"[INVERSE ETF]   {ticker}: ${amount:,.2f}")
    
    # CRISIS mode
    if regime == 'CRISIS':
        print(f"\n⚠️  CRISIS MODE ACTIVATED")
        print(f"⚠️  Exiting all LONG positions")
        print(f"⚠️  Allocating {inverse_allocation_pct:.0%} to inverse ETFs")
        
        long_positions = [t for t in current_tickers if t not in ['SH', 'PSQ', 'DOG', 'RWM']]
        decisions['sell'] = long_positions
        
        for ticker in long_positions:
            decisions['reason'][ticker] = 'CRISIS_EXIT_LONGS'
        
        for ticker in decisions['inverse_etfs'].keys():
            if ticker not in current_tickers:
                decisions['buy'].append(ticker)
                decisions['reason'][ticker] = 'CRISIS_INVERSE_ETF'
            else:
                decisions['hold'].append(ticker)
        
        return decisions
    
    # Bearish regimes
    if regime in ['BEAR_STRONG', 'BEAR_WEAK', 'NEUTRAL_BEARISH']:
        print(f"\n[BEARISH] {regime} - Adding inverse ETF hedge")
        
        long_positions = [t for t in current_tickers if t not in ['SH', 'PSQ', 'DOG', 'RWM']]
        max_longs = max_positions - len(decisions['inverse_etfs'])
        
        if len(long_positions) > max_longs:
            to_sell = long_positions[max_longs:]
            decisions['sell'].extend(to_sell)
            for ticker in to_sell:
                decisions['reason'][ticker] = f'{regime}_REDUCE_LONGS'
        
        for ticker in decisions['inverse_etfs'].keys():
            if ticker not in current_tickers:
                decisions['buy'].append(ticker)
                decisions['reason'][ticker] = f'{regime}_INVERSE_HEDGE'
            else:
                decisions['hold'].append(ticker)
    
    # No new positions mode
    if no_new_positions and regime != 'CRISIS':
        print(f"\n[DECISION] NO NEW POSITIONS MODE")
        print(f"[DECISION] Exiting all {len(current_tickers)} positions")
        
        decisions['sell'] = list(current_tickers)
        for ticker in current_tickers:
            decisions['reason'][ticker] = 'NO_NEW_POSITIONS'
        
        return decisions
    
    # Normal mode
    if regime not in ['CRISIS', 'BEAR_STRONG', 'BEAR_WEAK', 'NEUTRAL_BEARISH']:
        if candidates_df.empty:
            print(f"\n[DECISION] No candidates available")
            decisions['hold'] = list(current_tickers)
            return decisions
        
        candidate_tickers = set(candidates_df['ticker'].tolist()[:max_positions])
        
        to_sell = current_tickers - candidate_tickers
        decisions['sell'] = list(to_sell)
        
        for ticker in to_sell:
            decisions['reason'][ticker] = 'NOT_IN_TOP_CANDIDATES'
        
        to_hold = current_tickers & candidate_tickers
        decisions['hold'] = list(to_hold)
        
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
    if decisions['inverse_etfs']:
        print(f"  Inverse ETFs: {len(decisions['inverse_etfs'])} ({inverse_allocation_pct:.0%} allocation)")
    
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
    
    enriched_portfolio = enrich_portfolio_with_prices(
        portfolio_state,
        price_cache_dir,
        today
    )
    
    # 1. Trade candidates (BUY)
    buy_tickers = decisions['buy']
    inverse_etfs = decisions.get('inverse_etfs', {})
    
    if buy_tickers:
        buy_data = []
        current_prices = load_current_prices(price_cache_dir, buy_tickers, today)
        
        for ticker in buy_tickers:
            entry_price = current_prices.get(ticker)
            
            if entry_price is None:
                print(f"[ERROR] No price found for {ticker}, skipping")
                continue
            
            atr = calculate_atr(price_cache_dir, ticker)
            sl_tp = calculate_sl_tp(entry_price, atr)
            
            if ticker in inverse_etfs:
                pos_size = inverse_etfs[ticker]
                qty = int(pos_size / entry_price) if entry_price > 0 else 0
                
                buy_data.append({
                    'Ticker': ticker,
                    'Side': 'LONG',
                    'EntryPrice': entry_price,
                    'Quantity': qty,
                    'PositionSize': pos_size,
                    'ATR': atr,
                    'StopLoss': sl_tp['sl'],
                    'TakeProfit': sl_tp['tp'],
                    'Type': 'INVERSE_ETF',
                    'Note': decisions['reason'].get(ticker, 'Unknown'),
                    'Regime': regime_data.get('regime', 'Unknown')
                })
            else:
                if 'adjusted_position_size' in candidates_df.columns and ticker in candidates_df['ticker'].values:
                    row = candidates_df[candidates_df['ticker'] == ticker].iloc[0]
                    pos_size = row.get('adjusted_position_size', position_size)
                else:
                    pos_size = position_size
                
                qty = int(pos_size / entry_price) if entry_price > 0 else 0
                
                buy_data.append({
                    'Ticker': ticker,
                    'Side': 'LONG',
                    'EntryPrice': entry_price,
                    'Quantity': qty,
                    'PositionSize': pos_size,
                    'ATR': atr,
                    'StopLoss': sl_tp['sl'],
                    'TakeProfit': sl_tp['tp'],
                    'Type': 'REGULAR',
                    'Note': decisions['reason'].get(ticker, 'Unknown'),
                    'Regime': regime_data.get('regime', 'Unknown')
                })
        
        if buy_data:
            buy_df = pd.DataFrame(buy_data)
            buy_path = os.path.join(actions_dir, "trade_candidates.csv")
            buy_df.to_csv(buy_path, index=False)
            print(f"[OK] Saved: {buy_path}")
        else:
            pd.DataFrame(columns=['Ticker', 'Side', 'EntryPrice', 'StopLoss', 'TakeProfit']).to_csv(
                os.path.join(actions_dir, "trade_candidates.csv"), index=False
            )
            print(f"[WARN] No valid buy candidates with prices")
    else:
        pd.DataFrame(columns=['Ticker', 'Side', 'EntryPrice', 'StopLoss', 'TakeProfit']).to_csv(
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
        
        current_prices = load_current_prices(price_cache_dir, sell_tickers, today)
        
        for ticker in sell_tickers:
            pos = current_positions.get(ticker, {})
            entry_price = pos.get('entry_price', 0.0)
            qty = pos.get('quantity', 0)
            
            current_price = current_prices.get(ticker, entry_price)
            
            atr = calculate_atr(price_cache_dir, ticker)
            sl_tp = calculate_sl_tp(entry_price if entry_price > 0 else current_price, atr)
            
            if entry_price > 0:
                pl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pl_pct = 0.0
            
            sell_data.append({
                'Ticker': ticker,
                'Side': 'SELL',
                'EntryPrice': entry_price,
                'CurrentPrice': current_price,
                'Quantity': qty,
                'PL_Pct': pl_pct,
                'ATR': atr,
                'StopLoss': sl_tp['sl'],
                'TakeProfit': sl_tp['tp'],
                'Reason': decisions['reason'].get(ticker, 'Unknown')
            })
        
        sell_df = pd.DataFrame(sell_data)
        sell_path = os.path.join(actions_dir, "sell_candidates.csv")
        sell_df.to_csv(sell_path, index=False)
        print(f"[OK] Saved: {sell_path}")
    else:
        pd.DataFrame(columns=['Ticker', 'Side', 'CurrentPrice', 'StopLoss', 'Reason']).to_csv(
            os.path.join(actions_dir, "sell_candidates.csv"), index=False
        )
        print(f"[OK] No sell candidates")
    
    # 3. Action plan
    action_plan_path = os.path.join(actions_dir, "action_plan.txt")
    
    with open(action_plan_path, 'w', encoding='utf-8') as f:
        f.write("="*130 + "\n")
        f.write(f"TRADE ACTION PLAN - {today.strftime('%Y-%m-%d')}\n")
        f.write("="*130 + "\n\n")
        
        # Current portfolio
        f.write("CURRENT PORTFOLIO:\n")
        f.write("-"*130 + "\n")
        
        if not enriched_portfolio.empty:
            f.write(f"Total Positions: {len(enriched_portfolio)}\n")
            f.write(f"Cash: ${portfolio_state.get('cash', 0.0):,.2f}\n")
            
            total_value = portfolio_state.get('cash', 0.0) + enriched_portfolio['Value'].sum()
            entry_value = (enriched_portfolio['Entry_Price'] * enriched_portfolio['Quantity']).sum()
            total_pl = enriched_portfolio['Value'].sum() - entry_value
            total_pl_pct = (total_pl / entry_value) * 100 if entry_value > 0 else 0.0
            
            f.write(f"Total Portfolio Value: ${total_value:,.2f}\n")
            f.write(f"Total P/L: ${total_pl:,.2f} ({total_pl_pct:+.2f}%)\n\n")
            
            f.write(f"{'Ticker':<8} | {'Entry':<10} | {'Days':<4} | {'Entry $':<9} | {'Current $':<9} | {'Qty':<4} | ")
            f.write(f"{'Value $':<10} | {'P/L%':<8} | {'ATR $':<7} | {'SL $':<9} | {'TP $':<9} | {'Regime':<16}\n")
            f.write("-"*130 + "\n")
            
            inverse_etf_tickers = ['SH', 'PSQ', 'DOG', 'RWM']
            longs = enriched_portfolio[~enriched_portfolio['Ticker'].isin(inverse_etf_tickers)]
            inverses = enriched_portfolio[enriched_portfolio['Ticker'].isin(inverse_etf_tickers)]
            
            if not longs.empty:
                f.write("LONG POSITIONS:\n")
                for _, row in longs.iterrows():
                    pl_symbol = "+" if row['PL_Pct'] >= 0 else ""
                    f.write(f"{row['Ticker']:<8} | {row['Entry_Date']:<10} | {row['Days_Held']:<4} | ")
                    f.write(f"${row['Entry_Price']:>8.2f} | ${row['Current_Price']:>8.2f} | {row['Quantity']:<4} | ")
                    f.write(f"${row['Value']:>9.2f} | {pl_symbol}{row['PL_Pct']:>6.2f}% | ")
                    f.write(f"${row['ATR']:>6.2f} | ${row['SL']:>8.2f} | ${row['TP']:>8.2f} | {row['Regime']:<16}\n")
            
            if not inverses.empty:
                f.write("\nINVERSE ETF POSITIONS (HEDGES):\n")
                for _, row in inverses.iterrows():
                    pl_symbol = "+" if row['PL_Pct'] >= 0 else ""
                    f.write(f"{row['Ticker']:<8} | {row['Entry_Date']:<10} | {row['Days_Held']:<4} | ")
                    f.write(f"${row['Entry_Price']:>8.2f} | ${row['Current_Price']:>8.2f} | {row['Quantity']:<4} | ")
                    f.write(f"${row['Value']:>9.2f} | {pl_symbol}{row['PL_Pct']:>6.2f}% | ")
                    f.write(f"${row['ATR']:>6.2f} | ${row['SL']:>8.2f} | ${row['TP']:>8.2f} | {row['Regime']:<16}\n")
        else:
            f.write("(empty - no positions)\n")
        
        f.write("\n" + "="*130 + "\n\n")
        
        # Regime info
        f.write("MARKET REGIME:\n")
        f.write(f"  Current: {regime_data.get('regime', 'Unknown')}\n")
        f.write(f"  Score: {regime_data.get('composite_score', 0.0):+.4f}\n")
        f.write(f"  Confidence: {regime_data.get('confidence', 0.0):.1%}\n")
        
        inverse_allocation_pct = get_inverse_allocation_pct(regime_data.get('regime', ''))
        if inverse_allocation_pct > 0:
            f.write(f"  Inverse ETF Allocation: {inverse_allocation_pct:.0%}\n")
        
        f.write("\n")
        
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
        f.write(f"  HOLD: {len(decisions['hold'])} positions\n")
        if decisions.get('inverse_etfs'):
            f.write(f"  Inverse ETFs: {len(decisions['inverse_etfs'])} positions\n")
        f.write("\n")
        
        # BUY Details
        if decisions['buy'] and buy_data:
            f.write("BUY ORDERS:\n")
            f.write(f"{'Ticker':<8} | {'Entry $':<9} | {'Qty':<5} | {'Size $':<9} | {'ATR $':<7} | {'SL $':<9} | {'TP $':<9} | {'Reason':<30}\n")
            f.write("-"*130 + "\n")
            for item in buy_data:
                f.write(f"{item['Ticker']:<8} | ${item['EntryPrice']:>8.2f} | {item['Quantity']:<5} | ${item['PositionSize']:>8.2f} | ")
                f.write(f"${item['ATR']:>6.2f} | ${item['StopLoss']:>8.2f} | ${item['TakeProfit']:>8.2f} | {item['Note']:<30}\n")
            f.write("\n")
        
        # SELL Details
        if decisions['sell'] and sell_data:
            f.write("SELL ORDERS:\n")
            f.write(f"{'Ticker':<8} | {'Entry $':<9} | {'Current $':<9} | {'P/L%':<8} | {'Reason':<30}\n")
            f.write("-"*130 + "\n")
            for item in sell_data:
                pl_symbol = "+" if item['PL_Pct'] >= 0 else ""
                f.write(f"{item['Ticker']:<8} | ${item['EntryPrice']:>8.2f} | ${item['CurrentPrice']:>8.2f} | {pl_symbol}{item['PL_Pct']:>6.2f}% | {item['Reason']:<30}\n")
            f.write("\n")
        
        if decisions['hold']:
            f.write("HOLD POSITIONS:\n")
            for ticker in decisions['hold']:
                f.write(f"  - {ticker}\n")
            f.write("\n")
        
        # Portfolio after actions
        f.write("="*130 + "\n")
        f.write("PORTFOLIO AFTER ACTIONS:\n")
        f.write("="*130 + "\n")
        
        total_positions = len(decisions['hold']) + len(decisions['buy'])
        f.write(f"Total Positions: {total_positions}\n")
        f.write(f"Cash: ${portfolio_state.get('cash', 0.0):.2f}\n")
        
        f.write("\n" + "="*130 + "\n")
    
    print(f"[OK] Saved: {action_plan_path}")
    
    return actions_dir, action_plan_path

# ======================== MAIN ========================

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("AUTO DECIDER - Regime-Aware Trade Decision System v4.3.2")
    print("With Inverse ETF Support + REAL Price Enrichment + Email")
    print("="*80 + "\n")
    
    today = _as_date(args.today)
    print(f"[INFO] Today: {today.strftime('%Y-%m-%d')}")
    print(f"[INFO] Commit mode: {'LIVE' if args.commit else 'DRY-RUN'}")
    
    portfolio_state = load_portfolio_state(args.project_root)
    print(f"[INFO] Current positions: {len(portfolio_state['positions'])}")
    print(f"[INFO] Cash: ${portfolio_state.get('cash', 0.0):.2f}")
    
    regime_data = detect_current_regime(args.project_root, today)
    
    if regime_data.get('regime') == 'CRISIS':
        print(f"\n⚠️  WARNING: CRISIS REGIME DETECTED")
        print(f"⚠️  Enabling capital preservation mode (exit longs, buy inverse ETFs)")
    
    if args.no_new_positions:
        print(f"\n[INFO] NO NEW POSITIONS mode enabled")
    
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
    
    filtered_candidates = apply_regime_strategy(
        candidates_df,
        regime_data,
        args.max_positions,
        args.position_size
    )
    
    decisions = decide_trades(
        filtered_candidates,
        portfolio_state,
        regime_data,
        args.max_positions,
        args.position_size,
        args.no_new_positions
    )
    
    actions_dir, action_plan_path = generate_outputs(
        decisions,
        filtered_candidates,
        portfolio_state,
        regime_data,
        args.run_root,
        today,
        args.position_size,
        args.price_cache_dir
    )
    
    print(f"\n[INFO] Simulating portfolio after actions...")
    
    simulated_portfolio = portfolio_state.copy()
    
    if not isinstance(simulated_portfolio.get('positions'), dict):
        simulated_portfolio['positions'] = {}
    
    buy_tickers = decisions['buy']
    if buy_tickers:
        current_prices = load_current_prices(args.price_cache_dir, buy_tickers, today)
    else:
        current_prices = {}
    
    for ticker in decisions['sell']:
        if ticker in simulated_portfolio['positions']:
            del simulated_portfolio['positions'][ticker]
    
    for ticker in decisions['buy']:
        price = current_prices.get(ticker)
        
        if price is None:
            print(f"[WARN] Skipping {ticker} in portfolio sim (no price)")
            continue
        
        if ticker in decisions.get('inverse_etfs', {}):
            amount = decisions['inverse_etfs'][ticker]
            qty = int(amount / price) if price > 0 else 10
        else:
            qty = int(args.position_size / price) if price > 0 else 10
        
        simulated_portfolio['positions'][ticker] = {
            'entry_date': today.strftime('%Y-%m-%d'),
            'entry_price': price,
            'quantity': qty,
            'regime_at_entry': regime_data.get('regime', 'Unknown')
        }
    
    sim_path = os.path.join(actions_dir, "portfolio_after_sim.csv")
    
    sim_enriched = enrich_portfolio_with_prices(
        simulated_portfolio,
        args.price_cache_dir,
        today
    )
    
    if not sim_enriched.empty:
        sim_enriched['Status'] = sim_enriched['Ticker'].apply(
            lambda t: 'HOLD' if t in decisions['hold'] else 'NEW'
        )
        
        sim_enriched['Type'] = sim_enriched['Ticker'].apply(
            lambda t: 'INVERSE_ETF' if t in ['SH', 'PSQ', 'DOG', 'RWM'] else 'REGULAR'
        )
        
        sim_enriched.to_csv(sim_path, index=False)
    else:
        pd.DataFrame(columns=['Ticker', 'Entry_Date', 'Entry_Price', 'Current_Price', 'Quantity', 'Value', 'ATR', 'SL', 'TP', 'Type', 'Status']).to_csv(sim_path, index=False)
    
    print(f"[OK] Saved simulated portfolio: {sim_path}")
    
    if args.commit:
        print(f"\n[INFO] Committing changes to portfolio_state.json...")
        simulated_portfolio['last_updated'] = today.strftime('%Y-%m-%d')
        save_portfolio_state(args.project_root, simulated_portfolio)
    else:
        print(f"\n[INFO] Dry-run: portfolio_state.json ei päivitetty (commit=0).")
    
    print("\n" + "="*80)
    print("AUTO DECIDER COMPLETE")
    print("="*80 + "\n")
    
    # ==================== EMAIL NOTIFICATION (INTEGRATED) ====================
    if not args.no_email:
        print("[INFO] Sending email notification...")
        email_sent = send_email_notification(actions_dir, action_plan_path)
        
        if email_sent:
            print("[OK] Email notification sent successfully")
        else:
            print("[INFO] Email not sent (credentials missing or error occurred)")
    else:
        print("[INFO] Email notification skipped (--no_email flag)")
    # ========================================================================

if __name__ == "__main__":
    main()