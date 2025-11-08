#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_exit_watchlist.py v2.1

Rakentaa exit-watchlistin päivän salkun ja hintacachen perusteella.

- Lukee ENSISIJAISESTI portfolio_state.json (oikeat entry prices)
- Vaihtoehtoisesti portfolio_after_sim.csv
- Hakee kunkin tickerin viimeisimmän hinnan ja ATR:n price_cache -kansiosta.
- Laskee Entry_Price, Current_Price, P/L%, Stop_Loss, Take_Profit, Exit_Signal.
- Kirjoittaa exit_watchlist.csv actions-kansioon.

v2.1 KORJAUKSET:
- FIXED: Lukee portfolio_state.json joka sisältää oikeat entry_prices
- Fallback: portfolio_after_sim.csv jos JSON ei löydy
- Output: Ticker, Entry_Date, Entry_Price, Current_Price, PL_Pct, ATR, Stop_Loss, Take_Profit, Exit_Signal
"""

from __future__ import annotations

import os
import sys
import json
import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ---- Konsolin enkoodaus: ei kaadu erikoismerkkeihin ----
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# ---- Joustava CSV-luku ----
_SEPARATORS = [",", ";", "\t", "|"]
_ENCODINGS = ["utf-8", "cp1252", "latin1"]


def _read_csv_flexible(path: Path, expect_columns: Optional[list[str]] = None) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in _ENCODINGS:
        for sep in _SEPARATORS:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                if df.shape[1] == 1 and (expect_columns is None or df.shape[1] < len(expect_columns)):
                    raise ValueError(f"Separation guess failed (sep='{sep}', enc={enc}) -> only 1 column")
                return df
            except Exception as e:
                last_err = e
                continue
    raise ValueError(f"CSV-luku epäonnistui: {path} (last_err={last_err})")


def _load_portfolio_from_json(json_path: Path) -> Optional[dict]:
    """
    Load portfolio from portfolio_state.json.
    
    Returns dict: {ticker: {'entry_price': float, 'entry_date': str, 'quantity': int}}
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        positions = data.get('positions', {})
        if not isinstance(positions, dict):
            return None
        
        portfolio_dict = {}
        for ticker, pos in positions.items():
            portfolio_dict[ticker.upper()] = {
                'entry_price': float(pos.get('entry_price', 0.0)),
                'entry_date': str(pos.get('entry_date', 'Unknown')),
                'quantity': int(pos.get('quantity', 0))
            }
        
        return portfolio_dict
        
    except Exception as e:
        print(f"[WARN] Failed to load portfolio_state.json: {e}")
        return None


def _find_portfolio_csv(actions_dir: Path) -> Optional[Path]:
    p = actions_dir / "portfolio_after_sim.csv"
    if p.exists():
        return p
    candidates = sorted(actions_dir.glob("portfolio_after_sim*.csv"))
    return candidates[0] if candidates else None


def _find_portfolio_json(project_root: Path) -> Optional[Path]:
    """Find portfolio_state.json in seasonality_reports directory"""
    candidates = [
        project_root / "seasonality_reports" / "portfolio_state.json",
        project_root / "portfolio_state.json",
    ]
    
    for c in candidates:
        if c.exists():
            return c
    
    return None


def _ticker_column(df: pd.DataFrame) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for key in ("ticker", "symbol"):
        if key in lowered:
            return lowered[key]
    return None


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_price_file(price_cache_dir: Path, ticker: str) -> Optional[pd.DataFrame]:
    """
    Read price CSV from cache with CSV parsing fixes (v4.3.3).
    Handles ticker symbols in first row and converts strings to numeric.
    """
    t = ticker.upper()
    candidates = [
        price_cache_dir / f"{t}.csv",
        price_cache_dir / f"{t}",
    ]
    if not any(c.exists() for c in candidates):
        g = sorted(glob.glob(str(price_cache_dir / f"{t}*")))
        candidates.extend([Path(x) for x in g])
    
    for c in candidates:
        if c.exists() and c.is_file():
            try:
                df = pd.read_csv(c)
                
                # CRITICAL FIX: Skip first row if it contains ticker symbols
                if not df.empty and 'Date' in df.columns:
                    first_date = str(df.iloc[0]['Date']).strip()
                    if first_date == '' or first_date == t or not first_date[0].isdigit():
                        df = df.iloc[1:].reset_index(drop=True)
                
                # Normalize column names
                df.columns = [str(x).strip().lower().replace(' ', '_') for x in df.columns]
                
                # CRITICAL FIX: Convert string columns to numeric
                numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'adjclose', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            except Exception as e:
                print(f"[WARN] Failed to read {c}: {e}")
                continue
    return None


def _compute_atr(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    """
    Calculate ATR(14) - Average True Range.
    Industry standard: 14-day rolling average of True Range.
    """
    cols = df.columns
    close_col = "adj_close" if "adj_close" in cols else ("close" if "close" in cols else None)
    if close_col is None or "high" not in cols or "low" not in cols:
        return None
    
    d = df[["high", "low", close_col]].dropna()
    if len(d) < n + 1:
        return None
    
    prev_close = d[close_col].shift(1)
    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - prev_close).abs()
    tr3 = (d["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=n, min_periods=n).mean().iloc[-1]
    
    return float(atr) if pd.notna(atr) else None


def _discover_price_cache_dir(price_cache_dir: Path, actions_dir: Path) -> Path:
    """
    Jos annettu price_cache_dir ei löydy, etsi runs/*/price_cache -kansiot ja
    valitse se, jossa on eniten .csv -tiedostoja. Palauta tämä polku.
    Jos mitään ei löydy, palauta alkuperäinen polku (skripti ei kaadu).
    """
    if price_cache_dir.exists():
        return price_cache_dir

    # Etsi 'runs' -juuri actions_dirin yläpuolelta
    runs_root: Optional[Path] = None
    for p in actions_dir.parents:
        if p.name.lower() == "runs":
            runs_root = p
            break

    candidates: list[Path] = []
    if runs_root and runs_root.exists():
        for p in runs_root.glob("*"):
            pc = p / "price_cache"
            if pc.is_dir():
                candidates.append(pc)

    if candidates:
        # valitse se, jossa eniten csv-tiedostoja (tyypillisesti oikea cache)
        best = max(candidates, key=lambda d: len(list(d.glob("*.csv"))))
        print(f"[WARN] price_cache_dir ei löytynyt → käytetään löydettyä: {best}")
        return best

    print(f"[WARN] price_cache_dir '{price_cache_dir}' ei löytynyt eikä korvaavaa löydetty; jatketaan silti.")
    return price_cache_dir


def build_exit_watchlist(
    price_cache_dir: Path,
    actions_dir: Path,
    portfolio_csv: Optional[Path] = None,
    portfolio_json: Optional[Path] = None,
    atr_n: int = 14,
    lookback: int = 60,
    stop_mult: float = 2.0,
    take_mult: float = 3.0,
) -> Path:
    """
    Build exit watchlist with full portfolio context.
    
    Priority:
    1. portfolio_json (portfolio_state.json) - contains REAL entry prices
    2. portfolio_csv (portfolio_after_sim.csv) - fallback
    
    Returns CSV with columns:
    - Ticker, Entry_Date, Entry_Price, Current_Price, PL_Pct, ATR, Stop_Loss, Take_Profit, Exit_Signal
    """
    actions_dir.mkdir(parents=True, exist_ok=True)
    out = actions_dir / "exit_watchlist.csv"

    # -- varmista price_cache_dir (auto-discovery) --
    price_cache_dir = _discover_price_cache_dir(price_cache_dir, actions_dir)

    # 1) Try to load from portfolio_state.json FIRST (contains real entry prices)
    portfolio_dict = None
    
    if portfolio_json:
        print(f"[INFO] Trying to load portfolio from JSON: {portfolio_json}")
        portfolio_dict = _load_portfolio_from_json(portfolio_json)
        if portfolio_dict:
            print(f"[OK] Loaded {len(portfolio_dict)} positions from portfolio_state.json")
    
    # 2) Fallback to portfolio_after_sim.csv if JSON failed
    if portfolio_dict is None:
        print(f"[INFO] Falling back to portfolio CSV")
        
        if portfolio_csv is None:
            portfolio_csv = _find_portfolio_csv(actions_dir)

        if portfolio_csv is None or not portfolio_csv.exists():
            pd.DataFrame(columns=["Ticker", "Entry_Date", "Entry_Price", "Current_Price", "PL_Pct", "ATR", "Stop_Loss", "Take_Profit", "Exit_Signal"]).to_csv(out, index=False)
            print(f"[WARN] No portfolio found → empty exit_watchlist: {out}")
            return out

        try:
            port = _read_csv_flexible(portfolio_csv)
        except Exception as e:
            pd.DataFrame(columns=["Ticker", "Entry_Date", "Entry_Price", "Current_Price", "PL_Pct", "ATR", "Stop_Loss", "Take_Profit", "Exit_Signal"]).to_csv(out, index=False)
            print(f"[WARN] Portfolio read failed ({e}) → empty exit_watchlist: {out}")
            return out

        tcol = _ticker_column(port)
        if tcol is None:
            pd.DataFrame(columns=["Ticker", "Entry_Date", "Entry_Price", "Current_Price", "PL_Pct", "ATR", "Stop_Loss", "Take_Profit", "Exit_Signal"]).to_csv(out, index=False)
            print(f"[WARN] Ticker column not found → empty exit_watchlist: {out}")
            return out

        # Build portfolio dict from CSV
        portfolio_dict = {}
        col_lower_map = {str(col).strip().lower().replace('_', '').replace(' ', ''): col for col in port.columns}
        
        entry_price_col = col_lower_map.get('entryprice') or col_lower_map.get('price')
        entry_date_col = col_lower_map.get('entrydate') or col_lower_map.get('date')
        
        for idx, row in port.iterrows():
            ticker_val = str(row[tcol]).strip().upper()
            if ticker_val and ticker_val != 'NAN':
                entry_price_val = 0.0
                if entry_price_col:
                    try:
                        entry_price_val = float(row[entry_price_col])
                    except (ValueError, TypeError):
                        entry_price_val = 0.0
                
                entry_date_val = 'Unknown'
                if entry_date_col and pd.notna(row[entry_date_col]):
                    entry_date_val = str(row[entry_date_col])
                
                portfolio_dict[ticker_val] = {
                    'entry_price': entry_price_val,
                    'entry_date': entry_date_val,
                    'quantity': 0
                }

    if not portfolio_dict:
        pd.DataFrame(columns=["Ticker", "Entry_Date", "Entry_Price", "Current_Price", "PL_Pct", "ATR", "Stop_Loss", "Take_Profit", "Exit_Signal"]).to_csv(out, index=False)
        print(f"[WARN] Empty portfolio → empty exit_watchlist: {out}")
        return out

    tickers = list(portfolio_dict.keys())

    # 3) Calculate watchlist data with full context
    rows = []
    for t in tickers:
        # Get portfolio data
        entry_price = portfolio_dict[t]['entry_price']
        entry_date = portfolio_dict[t]['entry_date']
        
        # Get price data
        px = _read_price_file(price_cache_dir, t)
        if px is None or px.empty:
            rows.append({
                "Ticker": t,
                "Entry_Date": entry_date,
                "Entry_Price": entry_price,
                "Current_Price": np.nan,
                "PL_Pct": np.nan,
                "ATR": np.nan,
                "Stop_Loss": np.nan,
                "Take_Profit": np.nan,
                "Exit_Signal": "N/A"
            })
            continue

        px = px.tail(max(lookback, 20)).copy()
        close_col = "adj_close" if "adj_close" in px.columns else ("close" if "close" in px.columns else None)
        current_price = float(px[close_col].dropna().iloc[-1]) if close_col and not px[close_col].dropna().empty else np.nan
        atr = _compute_atr(px, n=atr_n)
        
        # Calculate derived values
        stop_loss = np.nan
        take_profit = np.nan
        pl_pct = np.nan
        exit_signal = "N/A"
        
        if not np.isnan(current_price) and atr is not None:
            stop_loss = current_price - stop_mult * atr
            take_profit = current_price + (take_mult * atr)
            
            # Calculate P/L%
            if entry_price > 0:
                pl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Exit signal if current price below stop loss
            exit_signal = "YES" if current_price < stop_loss else "NO"
        
        rows.append({
            "Ticker": t,
            "Entry_Date": entry_date,
            "Entry_Price": entry_price,
            "Current_Price": current_price,
            "PL_Pct": round(pl_pct, 2) if not np.isnan(pl_pct) else np.nan,
            "ATR": round(atr, 2) if atr is not None else np.nan,
            "Stop_Loss": round(stop_loss, 2) if not np.isnan(stop_loss) else np.nan,
            "Take_Profit": round(take_profit, 2) if not np.isnan(take_profit) else np.nan,
            "Exit_Signal": exit_signal
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)
    
    # Print summary
    print(f"[OK] Exit watchlist kirjoitettu: {out}  (n={len(out_df)})")
    
    # Count exit signals
    exit_count = len([r for r in rows if r["Exit_Signal"] == "YES"])
    if exit_count > 0:
        print(f"[WARN] Exit signaaleja: {exit_count}")
        for r in rows:
            if r["Exit_Signal"] == "YES":
                print(f"  - {r['Ticker']}: ${r['Current_Price']:.2f} < ${r['Stop_Loss']:.2f} (P/L: {r['PL_Pct']:.2f}%)")
    else:
        print(f"[OK] Ei exit signaaleja - kaikki positiot OK")
    
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build exit watchlist CSV from portfolio and price cache.")
    parser.add_argument("--price_cache_dir", required=True, help="Path to price_cache directory.")
    parser.add_argument("--actions_dir", required=True, help="Path to actions/YYYYMMDD directory for output & portfolio scan.")
    parser.add_argument("--portfolio_csv", default=None, help="Optional explicit path to portfolio_after_sim.csv.")
    parser.add_argument("--portfolio_json", default=None, help="Optional explicit path to portfolio_state.json (contains real entry prices).")
    parser.add_argument("--atr_n", type=int, default=14, help="ATR window (default 14).")
    parser.add_argument("--lookback", type=int, default=60, help="Price lookback rows to compute ATR (default 60).")
    parser.add_argument("--stop_mult", type=float, default=2.0, help="Stop multiplier, stop = close - stop_mult*ATR (default 2.0).")
    parser.add_argument("--take_mult", type=float, default=3.0, help="Take profit multiplier, tp = close + take_mult*ATR (default 3.0).")

    args = parser.parse_args()

    price_cache_dir = Path(args.price_cache_dir).expanduser()
    actions_dir = Path(args.actions_dir).expanduser()
    portfolio_csv = Path(args.portfolio_csv).expanduser() if args.portfolio_csv else None
    portfolio_json = Path(args.portfolio_json).expanduser() if args.portfolio_json else None
    
    # Auto-discover portfolio_state.json if not specified
    if portfolio_json is None:
        for parent in actions_dir.parents:
            if parent.name == "runs":
                project_root = parent.parent.parent
                portfolio_json = _find_portfolio_json(project_root)
                break

    out_path = build_exit_watchlist(
        price_cache_dir=price_cache_dir,
        actions_dir=actions_dir,
        portfolio_csv=portfolio_csv,
        portfolio_json=portfolio_json,
        atr_n=args.atr_n,
        lookback=args.lookback,
        stop_mult=args.stop_mult,
        take_mult=args.take_mult,
    )
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()