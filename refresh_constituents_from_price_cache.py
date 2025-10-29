# refresh_constituents_from_price_cache.py
# Lukee kaikki <price_cache>\*.csv tiedostonimet ja kirjoittaa
# seasonality_reports\Constituents_raw.csv (sarake: ticker).

import argparse, csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--project_root", required=True)
ap.add_argument("--price_cache_dir", required=True)
args = ap.parse_args()

PROJECT = Path(args.project_root)
PRICE_DIR = Path(args.price_cache_dir)
OUT = PROJECT / "seasonality_reports" / "Constituents_raw.csv"

tickers = []
for p in PRICE_DIR.glob("*.csv"):
    name = p.stem.upper()
    if name.startswith("_"):
        continue  # ohita apu-cachet tms.
    if name in ("UNIVERSE", "CONSTITUENTS_RAW"):
        continue
    tickers.append(name)

tickers = sorted(set(tickers))
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["ticker"])
    for t in tickers:
        w.writerow([t])

print(f"[OK] Wrote {len(tickers)} tickers -> {OUT}")
