# regime_cross_assets.py  — Phase 0 feature builder for cross-asset proxies
import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np

CROSS = ["SHY","IEF","TLT","HYG","LQD","UUP","USO","GLD","CPER","TIP","RINF"]  # keep in sync with build_prices_from_indexes
ALT_COPPER = "HG=F"  # optional future symbol if CPER missing

def _read_close(pc_dir: Path, symbol: str) -> pd.Series:
    """Robust reader for price_cache CSV with 'date' and 'close' (case-insensitive)."""
    f = pc_dir / f"{symbol}.csv"
    if not f.exists() and symbol == "CPER":
        # try copper future fallback
        f = pc_dir / f"{ALT_COPPER}.csv"
    if not f.exists():
        return pd.Series(dtype="float64", name=symbol)
    df = pd.read_csv(f)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("timestamp") or list(df.columns)[0]
    ccol = cols.get("close") or "Close"
    s = pd.Series(df[ccol].values, index=pd.to_datetime(df[dcol]), name=symbol)
    s = s[~s.index.duplicated()].sort_index()
    return s

def _pct_rank(s: pd.Series, window: int) -> pd.Series:
    """Rolling percent rank of the LAST element within window (0..1)."""
    def _pr(x):
        r = pd.Series(x).rank(pct=True).iloc[-1]
        return float(r)
    return s.rolling(window, min_periods=window).apply(_pr, raw=False)

def build_features(price_cache_dir: str, outdir: str):
    pc_dir = Path(price_cache_dir)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    # Load closes into wide table
    closes = []
    for sym in CROSS:
        s = _read_close(pc_dir, sym)
        if s.empty:
            print(f"[WARN] Missing series in price_cache: {sym}")
        closes.append(s)
    wide = pd.concat(closes, axis=1).sort_index()

    # Copper fallback rename (HG=F -> CPER logical name)
    if "CPER" in wide.columns and wide["CPER"].isna().all() and "HG=F" in wide.columns:
        wide["CPER"] = wide["HG=F"]

    # Save long prices
    long_prices = wide.stack(dropna=False).rename("close").reset_index()
    long_prices.columns = ["date", "symbol", "close"]
    long_prices.to_csv(out / "cross_asset_prices.csv", index=False)

    # --- Derived series
    def roc(s, n): return s.pct_change(n)
    feat = pd.DataFrame(index=wide.index)

    # Rates
    for sym in ["SHY","IEF","TLT"]:
        if sym in wide: feat[f"roc21_{sym.lower()}"] = roc(wide[sym], 21)

    # Slope proxies
    if "IEF" in wide and "SHY" in wide:
        slope = wide["IEF"] / wide["SHY"]
        feat["slope_ief_shy"] = slope
        feat["slope_ief_shy_delta5"] = slope.diff(5)

    # Credit
    if "HYG" in wide and "LQD" in wide:
        feat["credit_ret21"] = roc(wide["HYG"], 21) - roc(wide["LQD"], 21)
        feat["credit_ratio"] = wide["HYG"] / wide["LQD"]
        feat["credit_z63"] = (feat["credit_ret21"] - feat["credit_ret21"].rolling(63).mean()) / feat["credit_ret21"].rolling(63).std()

    # USD
    if "UUP" in wide:
        feat["roc21_uup"] = roc(wide["UUP"], 21)
        feat["uup_pct_rank_252"] = _pct_rank(wide["UUP"], 252)

    # Commodities
    for sym in ["USO","GLD","CPER"]:
        if sym in wide: feat[f"roc21_{sym.lower()}"] = roc(wide[sym], 21)

    if "CPER" in wide and "GLD" in wide:
        cogr = wide["CPER"] / wide["GLD"]
        feat["cogr"] = cogr
        feat["cogr_pct_rank_252"] = _pct_rank(cogr, 252)
        feat["cogr_delta5"] = cogr.diff(5)

    # Breakeven / real
    if "TIP" in wide and "IEF" in wide:
        tipief = wide["TIP"] / wide["IEF"]
        feat["tip_ief_ratio"] = tipief
        feat["tipief_delta5"] = tipief.diff(5)
    if "RINF" in wide:
        feat["roc21_rinf"] = roc(wide["RINF"], 21)

    # Finalize
    feat = feat.reset_index().rename(columns={"index":"date"})
    feat.to_csv(out / "cross_asset_features.csv", index=False)
    print(f"[OK] Wrote {out/'cross_asset_prices.csv'} and {out/'cross_asset_features.csv'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help="Latest run root (…\\seasonality_reports\\runs\\YYYY-mm-dd_hhmm)")
    ap.add_argument("--reports_root", required=True, help="…\\seasonality_reports")
    ap.add_argument("--today", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    price_cache = Path(args.run_root) / "price_cache"
    outdir = Path(args.reports_root) / "aggregates" / "regime_research" / args.today
    build_features(str(price_cache), str(outdir))

if __name__ == "__main__":
    main()
