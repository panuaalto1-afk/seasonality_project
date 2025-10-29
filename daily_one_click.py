#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, subprocess, sys, os
from pathlib import Path
from datetime import datetime

def run(cmd, cwd=None):
    print(">>>", " ".join([str(c) for c in cmd]))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    parser = argparse.ArgumentParser("Daily One Click (reuse-first)")
    parser.add_argument("--source_cache", required=True,
                        help="Polku valmiiseen price_cache-kansioon (ei ladata uusia hintoja).")
    parser.add_argument("--no_download", type=int, default=1)
    parser.add_argument("--topoff", type=int, default=0)
    parser.add_argument("--feature_mode", default="split")
    parser.add_argument("--gate_alpha", type=float, default=0.03)
    parser.add_argument("--train_years", type=int, default=7)
    parser.add_argument("--top_n", type=int, default=60)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent
    REPORTS = PROJECT_ROOT / "seasonality_reports"
    UNIVERSE_CSV = REPORTS / "constituents_raw.csv"

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    yyyymmdd = now.strftime("%Y%m%d")
    run_root = REPORTS / "runs" / now.strftime("%Y-%m-%d_%H%M")
    price_cache_src = Path(args.source_cache)  # VANHA toimiva cache

    ensure_dir(run_root)
    ensure_dir(run_root / "reports")
    actions_dir = ensure_dir(run_root / "actions" / yyyymmdd)

    print("\n=== [1/4] ML unified pipeline ===")
    run([
        sys.executable, str(PROJECT_ROOT / "ml_unified_pipeline.py"),
        "--run_root", str(run_root),
        "--today", today_str,
        "--universe_csv", str(UNIVERSE_CSV),
        "--feature_mode", args.feature_mode,
        "--gate_alpha", str(args.gate_alpha),
        "--train_years", str(args.train_years),
    ], cwd=PROJECT_ROOT)

    print("\n=== [2/4] Optio unified daily ===")
    run([
        sys.executable, str(PROJECT_ROOT / "optio_unified_daily.py"),
        "--project_root", str(PROJECT_ROOT),
        "--today", today_str,
        "--collect_mode", "existing",
        "--universe", "file",
        "--universe_csv", str(UNIVERSE_CSV),
        "--price_cache_dir", str(price_cache_src),
        "--top_n", str(args.top_n),
    ], cwd=PROJECT_ROOT)

    print("\n=== [3/4] Auto Decider (commit=1) ===")
    # HUOM: EI --top_n koska auto_decider ei tunne sitä
    run([
        sys.executable, str(PROJECT_ROOT / "auto_decider.py"),
        "--project_root", str(PROJECT_ROOT),
        "--universe_csv", str(UNIVERSE_CSV),
        "--run_root", str(run_root),
        "--price_cache_dir", str(price_cache_src),
        "--today", today_str,
        "--commit", "1",
    ], cwd=PROJECT_ROOT)

    print("\n=== [4/4] Exit watchlist ===")
    run([
        sys.executable, str(PROJECT_ROOT / "make_exit_watchlist.py"),
        "--price_cache_dir", str(price_cache_src),
        "--actions_dir", str(actions_dir),
    ], cwd=PROJECT_ROOT)

    # Valinnainen: päivitä tickers_pool (huom. oikea lippu --use-latest-run)
    try:
        print("\n=== [opt] aggregate_seasonality_picker (use latest run) ===")
        run([
            sys.executable, str(PROJECT_ROOT / "aggregate_seasonality_picker.py"),
            "--universe_csv", str(UNIVERSE_CSV),
            "--run_root", str(run_root),
            "--use-latest-run"
        ], cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        print("[WARN] aggregate_seasonality_picker epäonnistui (ei kriittinen päivittäiselle ajolle).")

    print("\n=== Daily One Click FINISHED ===")
    print("[Run ]", run_root)
    print("[Prices] reuse:", price_cache_src)
    print("[Actions]", actions_dir)
    print("[Reports]", run_root / "reports")

if __name__ == "__main__":
    main()
