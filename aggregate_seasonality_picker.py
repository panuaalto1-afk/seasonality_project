#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
aggregate_seasonality_picker.py
- Korjaa aiemman args.use / "use-latest-run" -bugin (käytä: args.use_latest_run).
- Rakentaa varman tikkeripoolin suoraan constituents_raw.csv:stä:
    seasonality_reports/aggregates/segments/tickers_pool.csv  (sarake: ticker)

Voit antaa halutessasi:
  --universe_csv <polku>.csv
  --use-latest-run  (ei pakollinen; pidetään yhteensopivuutta varten)
"""

import os, sys, argparse, csv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT_ROOT  = os.path.join(PROJECT_ROOT, "seasonality_reports")
RUNS_DIR     = os.path.join(REPORT_ROOT, "runs")
AGG_DIR      = os.path.join(REPORT_ROOT, "aggregates")
SEG_DIR      = os.path.join(AGG_DIR, "segments")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def latest_run_dir(base):
    if not os.path.isdir(base):
        return None
    dirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    return max(dirs, key=os.path.getmtime) if dirs else None

def read_universe(universe_csv):
    if not os.path.isfile(universe_csv):
        print(f"[WARN] Universe CSV not found: {universe_csv}")
        return []
    import pandas as pd
    df = pd.read_csv(universe_csv)
    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker") or cols.get("symbol") or list(df.columns)[0]
    tickers = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )
    return [t for t in tickers if t and t != "NAN"]

def write_pool(tickers, out_csv):
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker"])
        for t in tickers:
            w.writerow([t])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_csv", default=os.path.join(REPORT_ROOT, "constituents_raw.csv"))
    ap.add_argument("--run_root", default="")
    ap.add_argument("--use-latest-run", dest="use_latest_run", action="store_true")
    args = ap.parse_args()

    # Fallback run_root jos pyydetty (ei pakollinen tähän poolin rakentamiseen)
    run_root = args.run_root
    if not run_root and getattr(args, "use_latest_run", False):
        rr = latest_run_dir(RUNS_DIR)
        if rr:
            run_root = rr

    tickers = read_universe(args.universe_csv)
    out_csv = os.path.join(SEG_DIR, "tickers_pool.csv")
    write_pool(tickers, out_csv)

    print(f"[OK] Built tickers_pool: {out_csv}  ({len(tickers)} tickers)")
    return 0

if __name__ == "__main__":
    sys.exit(main())

