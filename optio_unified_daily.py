#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optio_unified_daily.py
v2025-11-11 (FIXED regime_sector_momentum.csv)

- Lukee optioiden enriched-datan (optio_price_enriched_all.csv)
- Jos tämän päivän CSV puuttuu, käyttää uusinta saatavilla olevaa (fallback)
- Kirjoittaa _with_regime -tiedoston (pass-through)
- Luo OIKEAN regime_sector_momentum.csv tiedoston (ei enää placeholder)
"""

import os
import sys
import argparse
import datetime as dt
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", required=True, help="Projektin juuressa oleva polku")
    p.add_argument("--today", default=None, help="YYYY-MM-DD; jos ei annettu, käytetään tämän päivän päivää")
    p.add_argument("--collect_mode", choices=["today", "existing"], default="existing",
                   help="today = vaadi tämän päivän CSV, existing = käytä uusinta saatavilla olevaa")
    p.add_argument("--universe", default="file")
    p.add_argument("--universe_csv", default=None)
    p.add_argument("--price_cache_dir", default=None)
    p.add_argument("--top_n", type=int, default=60)
    p.add_argument("--optio_csv", default=None)
    return p.parse_args()

def _to_date(s):
    return dt.date.fromisoformat(s)

def _fmt(d):
    return d.strftime("%Y-%m-%d")

def find_latest_optio_csv(base_dir, not_after_date):
    """Palauttaa (path, datestr, used_fallback)"""
    expected = os.path.join(base_dir, _fmt(not_after_date), "optio_price_enriched_all.csv")
    if os.path.isfile(expected):
        return expected, _fmt(not_after_date), False

    candidates = []
    if not os.path.isdir(base_dir):
        return None, None, False

    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            d = _to_date(name)
        except Exception:
            continue
        if d <= not_after_date:
            csv_path = os.path.join(full, "optio_price_enriched_all.csv")
            if os.path.isfile(csv_path):
                candidates.append((d, csv_path))

    if not candidates:
        return None, None, False

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_date, latest_path = candidates[0]
    return latest_path, _fmt(latest_date), True

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    args = parse_args()

    today = _to_date(args.today) if args.today else dt.date.today()
    print(f"[INFO] today = {today}")

    base_dir = os.path.join(args.project_root, "seasonality_reports", "aggregates", "optio_signals_enriched")

    if args.optio_csv:
        optio_csv = args.optio_csv
        used_fallback = False
        datestr = os.path.basename(os.path.dirname(optio_csv))
        print(f"[INFO] Using explicit optio_csv: {optio_csv}")
        if not os.path.isfile(optio_csv):
            print(f"[WARN] {optio_csv} ei löydy. Jatketaan ilman optiota (placeholderit).")
            optio_csv = None
    else:
        if args.collect_mode == "today":
            optio_csv = os.path.join(base_dir, _fmt(today), "optio_price_enriched_all.csv")
            datestr = _fmt(today)
            used_fallback = False
            if not os.path.isfile(optio_csv):
                raise FileNotFoundError(f"Optio CSV puuttuu (collect_mode=today): {optio_csv}")
        else:
            optio_csv, datestr, used_fallback = find_latest_optio_csv(base_dir, today)
            if optio_csv is None:
                print(f"[WARN] Optio CSV:tä ei löytynyt polusta: {base_dir}. Jatketaan ilman optiota (placeholderit).")

    out_dir = os.path.join(base_dir, datestr if datestr else _fmt(today))
    ensure_dir(out_dir)

    if optio_csv:
        if used_fallback:
            print(f"[WARN] Päivän optio CSV puuttui. Käytetään uusinta saatavilla olevaa: {optio_csv}")
        else:
            print(f"[INFO] Käytetään optio CSV:tä: {optio_csv}")

        df = pd.read_csv(optio_csv)

        out_csv = os.path.join(out_dir, "optio_price_enriched_all_with_regime.csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] Optio-enrich done -> {out_csv}")
    else:
        out_csv = os.path.join(out_dir, "optio_price_enriched_all_with_regime.csv")
        pd.DataFrame(columns=["ticker"]).to_csv(out_csv, index=False)
        print(f"[WARN] Optio CSV:tä ei löydy. Kirjoitettiin placeholder -> {out_csv}")

    # FIXED: Luo OIKEA regime_sector_momentum.csv
    sector_mom = os.path.join(out_dir, "regime_sector_momentum.csv")
    try:
        regime_history_path = os.path.join(args.project_root, "seasonality_reports", "regime", "regime_history.csv")
        
        if os.path.isfile(regime_history_path):
            regime_df = pd.read_csv(regime_history_path)
            
            if not regime_df.empty:
                latest = regime_df.iloc[-1]
                
                sector_data = []
                sector_data.append({'factor': 'regime', 'value': str(latest.get('regime', 'UNKNOWN'))})
                sector_data.append({'factor': 'composite_score', 'value': float(latest.get('composite_score', 0.0))})
                sector_data.append({'factor': 'confidence', 'value': float(latest.get('confidence', 0.0))})
                
                component_cols = ['equity_signal', 'volatility_signal', 'credit_signal', 'safe_haven_signal', 'breadth_signal']
                for col in component_cols:
                    if col in latest:
                        sector_data.append({'factor': col, 'value': float(latest[col])})
                
                pd.DataFrame(sector_data).to_csv(sector_mom, index=False)
                print(f"[OK] Regime sector momentum saved: {sector_mom} ({len(sector_data)} factors)")
            else:
                pd.DataFrame(columns=["factor","value"]).to_csv(sector_mom, index=False)
                print(f"[WARN] Regime history empty -> placeholder: {sector_mom}")
        else:
            pd.DataFrame(columns=["factor","value"]).to_csv(sector_mom, index=False)
            print(f"[WARN] Regime history not found: {regime_history_path}")
            print(f"[INFO] Saved placeholder: {sector_mom}")

    except Exception as e:
        print(f"[ERROR] Failed to create regime data: {e}")
        pd.DataFrame(columns=["factor","value"]).to_csv(sector_mom, index=False)

    print("[DONE] optio_unified_daily complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
