import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\panua\seasonality_project")
REPORTS = ROOT / "seasonality_reports"
PRICE_CACHE = REPORTS / "runs" / "2025-10-04_0903" / "price_cache"  # käytössä oleva cache
UNIVERSE_CSV = REPORTS / "Constituents_raw.csv"

# 1) Tickerit cachesta
tickers = []
for f in PRICE_CACHE.glob("*.csv"):
    name = f.stem.upper()
    if name in {"_UNIVERSE_CACHE"}:  # jätä apu/indeksi-tiedostot pois
        continue
    tickers.append(name)
df_t = pd.DataFrame({"ticker": sorted(set(tickers))})

# 2) Metadata lähteistä (valitaan 1. olemassa oleva)
meta_sources = [
    REPORTS / "aggregates" / "universe_with_funda.csv",
    REPORTS / "aggregates" / "universe_filtered.csv",
]
meta = None
for src in meta_sources:
    if src.exists():
        meta = pd.read_csv(src)
        break
if meta is None:
    meta = pd.DataFrame(columns=[
        "ticker","index","name","sector","industry","exchange","country","market_cap","has_options"
    ])

# normalisoi ja yhdistä
if "ticker" not in meta.columns:
    meta["ticker"] = []
meta["ticker"] = meta["ticker"].astype(str).str.upper()

out = df_t.merge(meta.drop_duplicates("ticker"), on="ticker", how="left")

# valinnainen: täytä puuttuva index selkeästi
if "index" not in out.columns:
    out["index"] = "NA"
else:
    out["index"] = out["index"].fillna("NA")

out.to_csv(UNIVERSE_CSV, index=False)
print(f"[OK] Wrote {len(out)} rows -> {UNIVERSE_CSV}")
