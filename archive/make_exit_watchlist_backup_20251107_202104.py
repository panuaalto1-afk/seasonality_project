#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_exit_watchlist.py

Rakentaa exit-watchlistin päivän salkun ja hintacachen perusteella.

- Lukee portfolio_after_sim.csv (tai --portfolio_csv).
- Hakee kunkin tickerin viimeisimmän hinnan ja ATR:n price_cache -kansiosta.
- Laskee ehdotetun stop-tason: stop = close - stop_mult * ATR.
- Kirjoittaa exit_watchlist.csv actions-kansioon.

Kestävät ominaisuudet:
- Konsolin merkistö ei kaada ajon (stdout UTF-8 / errors='replace').
- CSV-luku joustavasti: erotin (',',';','\\t','|'), enkoodaus ('utf-8','cp1252','latin1').
- Ticker-sarake löytyy kirjainkoosta riippumatta (ticker / symbol).
- Tyhjän salkun tapaus: kirjoitetaan tyhjä exit_watchlist.csv otsikoilla.
- Jos annettu price_cache_dir ei löydy, löydetään se automaattisesti runs/*/price_cache -haulla.
"""

from __future__ import annotations

import os
import sys
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


def _find_portfolio_csv(actions_dir: Path) -> Optional[Path]:
    p = actions_dir / "portfolio_after_sim.csv"
    if p.exists():
        return p
    candidates = sorted(actions_dir.glob("portfolio_after_sim*.csv"))
    return candidates[0] if candidates else None


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
                df = _read_csv_flexible(c)
                df.columns = [str(x).strip().lower() for x in df.columns]
                df = _coerce_numeric(df, ["open", "high", "low", "close", "adj_close"])
                return df
            except Exception:
                continue
    return None


def _compute_atr(df: pd.DataFrame, n: int = 14) -> Optional[float]:
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
    atr_n: int = 14,
    lookback: int = 60,
    stop_mult: float = 2.0,
) -> Path:
    actions_dir.mkdir(parents=True, exist_ok=True)
    out = actions_dir / "exit_watchlist.csv"

    # -- varmista price_cache_dir (auto-discovery) --
    price_cache_dir = _discover_price_cache_dir(price_cache_dir, actions_dir)

    # 1) Portfolion luku
    if portfolio_csv is None:
        portfolio_csv = _find_portfolio_csv(actions_dir)

    if portfolio_csv is None or not portfolio_csv.exists():
        pd.DataFrame(columns=["ticker", "last_close", "atr", "stop_suggestion"]).to_csv(out, index=False)
        print(f"[OK] Portfolio puuttuu → kirjoitettu tyhjä exit_watchlist: {out}")
        return out

    try:
        port = _read_csv_flexible(portfolio_csv)
    except Exception as e:
        pd.DataFrame(columns=["ticker", "last_close", "atr", "stop_suggestion"]).to_csv(out, index=False)
        print(f"[WARN] Portfolion luku epäonnistui ({e}) → kirjoitettu tyhjä exit_watchlist: {out}")
        return out

    tcol = _ticker_column(port)
    if tcol is None:
        pd.DataFrame(columns=["ticker", "last_close", "atr", "stop_suggestion"]).to_csv(out, index=False)
        print(f"[OK] Ticker-saraketta ei löytynyt → kirjoitettu tyhjä exit_watchlist: {out}")
        return out

    tickers = (
        port[tcol].astype(str).str.strip().str.upper().replace({np.nan: None}).tolist()
    )
    tickers = [t for t in tickers if t and t != "NAN"]

    if not tickers:
        pd.DataFrame(columns=["ticker", "last_close", "atr", "stop_suggestion"]).to_csv(out, index=False)
        print(f"[OK] Portfolio tyhjä → kirjoitettu tyhjä exit_watchlist: {out}")
        return out

    # 2) Lasketaan viimeisin close + ATR
    rows = []
    for t in tickers:
        px = _read_price_file(price_cache_dir, t)
        if px is None or px.empty:
            rows.append({"ticker": t, "last_close": np.nan, "atr": np.nan, "stop_suggestion": np.nan})
            continue

        px = px.tail(max(lookback, 20)).copy()
        close_col = "adj_close" if "adj_close" in px.columns else ("close" if "close" in px.columns else None)
        last_close = float(px[close_col].dropna().iloc[-1]) if close_col and not px[close_col].dropna().empty else np.nan
        atr = _compute_atr(px, n=atr_n)
        stop = np.nan
        if (not np.isnan(last_close)) and (atr is not None):
            stop = last_close - stop_mult * atr

        rows.append({"ticker": t, "last_close": last_close, "atr": atr if atr is not None else np.nan, "stop_suggestion": stop})

    out_df = pd.DataFrame(rows, columns=["ticker", "last_close", "atr", "stop_suggestion"])
    out_df.to_csv(out, index=False)
    print(f"[OK] Exit watchlist kirjoitettu: {out}  (n={len(out_df)})")
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build exit watchlist CSV from portfolio and price cache.")
    parser.add_argument("--price_cache_dir", required=True, help="Path to price_cache directory.")
    parser.add_argument("--actions_dir", required=True, help="Path to actions/YYYYMMDD directory for output & portfolio scan.")
    parser.add_argument("--portfolio_csv", default=None, help="Optional explicit path to portfolio_after_sim.csv.")
    parser.add_argument("--atr_n", type=int, default=14, help="ATR window (default 14).")
    parser.add_argument("--lookback", type=int, default=60, help="Price lookback rows to compute ATR (default 60).")
    parser.add_argument("--stop_mult", type=float, default=2.0, help="Stop multiplier, stop = close - stop_mult*ATR (default 2.0).")

    args = parser.parse_args()

    price_cache_dir = Path(args.price_cache_dir).expanduser()
    actions_dir = Path(args.actions_dir).expanduser()
    portfolio_csv = Path(args.portfolio_csv).expanduser() if args.portfolio_csv else None

    out_path = build_exit_watchlist(
        price_cache_dir=price_cache_dir,
        actions_dir=actions_dir,
        portfolio_csv=portfolio_csv,
        atr_n=args.atr_n,
        lookback=args.lookback,
        stop_mult=args.stop_mult,
    )
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
