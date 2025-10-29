# apply_regime_to_outputs.py
# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
from datetime import datetime, timedelta

from regime_overlay import load_mapping, build_regime_snapshot, attach_regime_overlay

def last_weekday(d: datetime) -> datetime:
    if d.weekday()==5: return d - timedelta(days=1)
    if d.weekday()==6: return d - timedelta(days=2)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--price_cache_dir", required=True)
    ap.add_argument("--universe_csv", required=True)
    ap.add_argument("--outputs_dir", required=True, help="Kansio, jossa ML-putken CSV:t")
    ap.add_argument("--today", default="")
    args = ap.parse_args()

    today = last_weekday(pd.to_datetime(args.today).to_pydatetime() if args.today else datetime.now())
    mapping = load_mapping(args.universe_csv)
    snapshot, _, _ = build_regime_snapshot(args.price_cache_dir, mapping, pd.to_datetime(today))

    targets = [
        "top_breakout_long.csv",
        "top_breakout_short.csv",
        "all_candidates.csv",
        "predictions_all.csv",
    ]
    for name in targets:
        p = os.path.join(args.outputs_dir, name)
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        if "ticker" not in df.columns:
            cand = [c for c in df.columns if c.lower() in ("symbol","ticker")]
            if cand: df = df.rename(columns={cand[0]:"ticker"})
            else:
                print("[WARN] skip", name, "(ticker-sarake puuttuu)")
                continue
        out = attach_regime_overlay(df, mapping, snapshot)
        out.to_csv(p, index=False)
        print("[OK] overlaid ->", p)

if __name__ == "__main__":
    main()
