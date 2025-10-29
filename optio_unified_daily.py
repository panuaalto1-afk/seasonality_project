#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optio_unified_daily.py
v2025-10-18a

- Lukee optioiden enriched-datan (optio_price_enriched_all.csv)
- Jos tämän päivän CSV puuttuu, käyttää uusinta saatavilla olevaa (fallback)
- Kirjoittaa varmistukseksi _with_regime -tiedoston (tässä pass-through, jotta putki ei katkea)
- Luo/varmistaa myös placeholderin "regime_sector_momentum.csv" jos sitä ei ole
"""

import os
import sys
import argparse
import datetime as dt
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", required=True, help="Projektin juuressa oleva polku (esim. C:\\Users\\...\\seasonality_project)")
    p.add_argument("--today", default=None, help="YYYY-MM-DD; jos ei annettu, käytetään tämän päivän päivää")
    p.add_argument("--collect_mode", choices=["today", "existing"], default="existing",
                   help="today = vaadi tämän päivän CSV, existing = käytä uusinta saatavilla olevaa")
    # yhteensopivuuden vuoksi (daily_one_click välittää nämä)
    p.add_argument("--universe", default="file")
    p.add_argument("--universe_csv", default=None)
    p.add_argument("--price_cache_dir", default=None)
    p.add_argument("--top_n", type=int, default=60)
    # mahdollinen pakotettu polku suoraan CSV:lle
    p.add_argument("--optio_csv", default=None)
    return p.parse_args()

def _to_date(s):
    return dt.date.fromisoformat(s)

def _fmt(d):
    return d.strftime("%Y-%m-%d")

def find_latest_optio_csv(base_dir, not_after_date):
    """
    Palauttaa (path, datestr, used_fallback)
    """
    # 1) Kokeile ensin not_after_date
    expected = os.path.join(base_dir, _fmt(not_after_date), "optio_price_enriched_all.csv")
    if os.path.isfile(expected):
        return expected, _fmt(not_after_date), False

    # 2) Etsi uusin olemassa oleva <= not_after_date
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

    # Mistä etsitään optio CSV?
    base_dir = os.path.join(args.project_root, "seasonality_reports", "aggregates", "optio_signals_enriched")

    # Jos käyttäjä antoi suoran polun, käytä sitä
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

    # Output-kansio: käytä sitä kansiota, josta optio CSV löytyi (tai tämän päivän kansiota jos ei löytynyt mitään)
    out_dir = os.path.join(base_dir, datestr if datestr else _fmt(today))
    ensure_dir(out_dir)

    if optio_csv:
        if used_fallback:
            print(f"[WARN] Päivän optio CSV puuttui. Käytetään uusinta saatavilla olevaa: {optio_csv}")
        else:
            print(f"[INFO] Käytetään optio CSV:tä: {optio_csv}")

        df = pd.read_csv(optio_csv)

        # Tässä vaiheessa voisimme liittää "regime overlayn". Jotta putki ei kaadu
        # silloinkaan kun overlay-funktiot muuttuvat, kirjoitetaan pass-through -versio.
        out_csv = os.path.join(out_dir, "optio_price_enriched_all_with_regime.csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] Optio-enrich done -> {out_csv}")
    else:
        # Ei yhtään optiota -> luodaan tyhjä placeholder
        out_csv = os.path.join(out_dir, "optio_price_enriched_all_with_regime.csv")
        pd.DataFrame(columns=["ticker"]).to_csv(out_csv, index=False)
        print(f"[WARN] Optio CSV:tä ei löydy. Kirjoitettiin placeholder -> {out_csv}")

    # Varmista, että löytyy myös sektorimomentin tiedosto (jos joku muu skripti katsoo tätä)
    sector_mom = os.path.join(out_dir, "regime_sector_momentum.csv")
    if not os.path.isfile(sector_mom):
        pd.DataFrame(columns=["factor","value"]).to_csv(sector_mom, index=False)
        print(f"[OK] Regime files -> {sector_mom} (placeholder)")

    print("[DONE] optio_unified_daily complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
