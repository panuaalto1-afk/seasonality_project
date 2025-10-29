#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# learn_drift_monitor.py  (v2025-10-18a)
# Datan/featureiden drift-raportti price_cache:sta (PSI per feature)

import os, sys, argparse, json, math, glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

FEATURES = [
    "ret_1d",      # close/close[-1]-1
    "ret_5d",      # close/close[-5]-1
    "roc_21",      # close/close[-21]-1
    "rv_21",       # realized vol 21d (std of daily returns)
    "rsi_14",      # RSI(14) (0..100)
]

def rsi(series, n=14):
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_features_from_close(close: pd.Series):
    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    roc21 = close.pct_change(21)
    rv21 = close.pct_change(1).rolling(21).std()
    rsi14 = rsi(close, 14)
    df = pd.DataFrame({
        "ret_1d": ret1,
        "ret_5d": ret5,
        "roc_21": roc21,
        "rv_21": rv21,
        "rsi_14": rsi14
    })
    return df

def psi(base: np.ndarray, cur: np.ndarray, nbins=10):
    # Drop NaNs
    base = base[np.isfinite(base)]
    cur  = cur[np.isfinite(cur)]
    if len(base) < 200 or len(cur) < 100:
        return np.nan
    # Binning by base quantiles
    qs = np.linspace(0, 1, nbins+1)
    edges = np.quantile(base, qs)
    edges[0]  = -np.inf
    edges[-1] = np.inf
    base_counts, _ = np.histogram(base, bins=edges)
    cur_counts,  _ = np.histogram(cur,  bins=edges)
    # Convert to proportions (avoid zero with small epsilon)
    eps = 1e-8
    base_prop = np.maximum(base_counts / base_counts.sum(), eps)
    cur_prop  = np.maximum(cur_counts  / cur_counts.sum(),  eps)
    # PSI per bin and sum
    return float(np.sum((cur_prop - base_prop) * np.log(cur_prop / base_prop)))

def main():
    ap = argparse.ArgumentParser("Drift monitor (PSI) from price_cache")
    ap.add_argument("--price_cache_dir", required=True)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--baseline_days", type=int, default=90)
    ap.add_argument("--recent_days", type=int, default=20)
    ap.add_argument("--min_tickers", type=int, default=50)
    args = ap.parse_args()

    price_cache = Path(args.price_cache_dir)
    run_root = Path(args.run_root)
    out_dir = run_root / "learn"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect features across all tickers
    base_vals = {k: [] for k in FEATURES}
    cur_vals  = {k: [] for k in FEATURES}
    n_files = 0

    for csvp in price_cache.glob("*.csv"):
        try:
            df = pd.read_csv(csvp)
            # flexible column naming
            cols = {c.lower(): c for c in df.columns}
            if "close" not in cols:
                if "adj close" in cols:
                    df["close"] = pd.to_numeric(df[cols["adj close"]], errors="coerce")
                else:
                    continue
            else:
                df["close"] = pd.to_numeric(df[cols["close"]], errors="coerce")

            df = df.dropna(subset=["close"])
            if len(df) < (args.baseline_days + args.recent_days + 40):
                continue

            feats = compute_features_from_close(df["close"])
            # windows
            recent = feats.tail(args.recent_days)
            baseline = feats.iloc[-(args.recent_days + args.baseline_days) : -args.recent_days]

            if len(recent) < args.recent_days or len(baseline) < args.baseline_days:
                continue

            for k in FEATURES:
                base_vals[k].extend(baseline[k].values.tolist())
                cur_vals[k].extend(recent[k].values.tolist())

            n_files += 1
        except Exception:
            continue

    if n_files < args.min_tickers:
        print(f"[DRIFT][WARN] liian vähän tikkeriä (got={n_files}, need>={args.min_tickers}). Raportti kirjoitetaan silti.")

    # Compute PSI per feature
    rows = []
    for k in FEATURES:
        b = np.array(base_vals[k], dtype=float)
        c = np.array(cur_vals[k], dtype=float)
        val = psi(b, c, nbins=10)
        level = ("high" if (isinstance(val, float) and val>0.25) else
                 "moderate" if (isinstance(val, float) and val>0.10) else
                 "stable")
        rows.append({"feature": k, "psi": round(val, 6) if isinstance(val, float) else None, "level": level,
                     "base_n": int(np.isfinite(b).sum()), "recent_n": int(np.isfinite(c).sum())})

    summary = pd.DataFrame(rows).sort_values(["level","psi"], ascending=[True, False])
    csv_path = out_dir / "drift_summary.csv"
    summary.to_csv(csv_path, index=False)

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "price_cache_dir": str(price_cache),
        "baseline_days": args.baseline_days,
        "recent_days": args.recent_days,
        "tickers_processed": n_files,
        "psi_thresholds": {"stable":"<=0.10", "moderate":"(0.10,0.25]", "high":">0.25"},
        "features": rows,
    }
    json_path = out_dir / "drift_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print(f"[DRIFT][OK] {csv_path}")
    print(f"[DRIFT][OK] {json_path}")

if __name__ == "__main__":
    main()
