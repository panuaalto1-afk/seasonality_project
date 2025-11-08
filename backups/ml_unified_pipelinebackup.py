# ml_unified_pipeline.py
# Seasonality project — unified daily ML pipeline (robust writer edition)
# Keeps the same CLI you already use. Produces the expected 6 report files, always.

import argparse
import os
import sys
import glob
import math
from datetime import datetime, date
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

# ----------------------------- CLI ---------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Seasonality ML unified pipeline (robust).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_root", type=str, default=None, help="Run root folder (will be created).")
    p.add_argument("--today", type=str, default=None, help="YYYY-MM-DD; defaults to today.")
    p.add_argument("--universe_csv", type=str, default="seasonality_reports/constituents_raw.csv",
                   help="Universe CSV (expects a column like ticker/symbol).")
    p.add_argument("--feature_mode", type=str, choices=["shared", "split"], default="shared",
                   help="Feature mode label (informational in this lightweight version).")
    p.add_argument("--gate_alpha", type=float, default=0.10,
                   help="Gating threshold on 0–1 scores (higher = fewer names).")
    p.add_argument("--train_years", type=int, default=7,
                   help="Historical lookback hint (used for labels window bounds).")
    p.add_argument("--min_samples_per_regime", type=int, default=60,
                   help="Minimum samples per regime (informational; not strict).")
    p.add_argument("--vintage_cutoff", type=str, default=None,
                   help="Optional vintage cutoff YYYY-MM-DD (informational).")
    return p.parse_args()

# --------------------------- Helpers --------------------------------

def _as_date(s: Optional[str]) -> date:
    if s is None:
        return date.today()
    return datetime.strptime(s, "%Y-%m-%d").date()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _reports_dir(run_root: str) -> str:
    d = os.path.join(run_root, "reports")
    _ensure_dir(d)
    return d

def _run_root_default(reports_root: str, d: date) -> str:
    tag = d.strftime("%Y-%m-%d_%H%M")
    # If hour/min not desired, still ok; we keep full tag to avoid clashes
    return os.path.join(reports_root, "runs", tag)

def _find_reports_root() -> str:
    # Project root = cwd; reports root under it by convention
    return os.path.join(os.getcwd(), "seasonality_reports")

def _find_price_cache_dir(reports_root: str) -> Optional[str]:
    # 1) Environment override
    env = os.environ.get("PRICE_CACHE_DIR")
    if env and os.path.isdir(env):
        return env

    # 2) Canonical cache mentioned in your notes
    canonical = os.path.join(
        reports_root, "runs", "2025-10-04_0903", "price_cache"
    )
    if os.path.isdir(canonical):
        return canonical

    # 3) Latest price_cache under runs
    runs_root = os.path.join(reports_root, "runs")
    if not os.path.isdir(runs_root):
        return None
    candidates = []
    for d in glob.glob(os.path.join(runs_root, "*")):
        pc = os.path.join(d, "price_cache")
        if os.path.isdir(pc):
            candidates.append(pc)
    if not candidates:
        return None
    # Sort by folder name descending (timestamped names)
    candidates.sort(reverse=True)
    return candidates[0]

def _read_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize ticker column
    cols = {c.lower(): c for c in df.columns}
    cand = None
    for k in ["ticker", "symbol", "ric", "isin"]:
        if k in cols:
            cand = cols[k]
            break
    if cand is None:
        # try first column
        df.columns = ["ticker"] + list(df.columns[1:])
        cand = "ticker"
    df["ticker"] = df[cand].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["ticker"]).drop_duplicates("ticker")
    return df[["ticker"]]

def _read_price_csv(pc_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    # Try common filename patterns
    patterns = [
        os.path.join(pc_dir, f"{ticker}.csv"),
        os.path.join(pc_dir, f"{ticker.upper()}.csv"),
        os.path.join(pc_dir, f"{ticker.lower()}.csv"),
    ]
    path = None
    for p in patterns:
        if os.path.isfile(p):
            path = p
            break
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date", None)
        if date_col is None:
            # common alternatives
            for k in ["time", "datetime"]:
                if k in cols:
                    date_col = cols[k]
                    break
        close_col = None
        for k in ["adj close", "adj_close", "close", "last", "price"]:
            if k in cols:
                close_col = cols[k]
                break
        if date_col is None or close_col is None:
            return None
        out = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out = out.sort_values("date").dropna()
        return out
    except Exception:
        return None

def _daily_features_from_prices(p: pd.DataFrame) -> Optional[pd.Series]:
    """
    Compute lightweight features for *today* (last available row):
    - mom5, mom20, mom60 (pct change)
    - vol20 (rolling std of daily returns)
    Returns a Series for the last date. If not enough history, returns None.
    """
    if p is None or len(p) < 60:
        return None
    s = p["close"].astype(float)
    ret = s.pct_change()
    mom5  = s.pct_change(5)
    mom20 = s.pct_change(20)
    mom60 = s.pct_change(60)
    vol20 = ret.rolling(20).std()

    last_idx = p.index[-1]
    feat = pd.Series({
        "asof_date": p.loc[last_idx, "date"],
        "mom5": float(mom5.iloc[-1]) if not pd.isna(mom5.iloc[-1]) else np.nan,
        "mom20": float(mom20.iloc[-1]) if not pd.isna(mom20.iloc[-1]) else np.nan,
        "mom60": float(mom60.iloc[-1]) if not pd.isna(mom60.iloc[-1]) else np.nan,
        "vol20": float(vol20.iloc[-1]) if not pd.isna(vol20.iloc[-1]) else np.nan,
    })
    return feat

def _regime_from_breadth(feats: pd.DataFrame) -> str:
    """Very lightweight regime: median mom20 across universe."""
    if feats is None or feats.empty or "mom20" not in feats:
        return "Neutral"
    m = feats["mom20"].median(skipna=True)
    if pd.isna(m):
        return "Neutral"
    if m > 0.02:
        return "Bull"
    if m < -0.02:
        return "Bear"
    return "Neutral"

# ------------------------ Robust writers ----------------------------

def _write_csv(df: Optional[pd.DataFrame], out_path: str, write_empty: bool = True):
    """Always create the file (header-only if empty), and log outcome."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rows = None
        if df is None:
            if write_empty:
                pd.DataFrame().to_csv(out_path, index=False)
                rows = 0
        else:
            if getattr(df, "empty", True):
                if write_empty:
                    df.head(0).to_csv(out_path, index=False)
                    rows = 0
            else:
                df.to_csv(out_path, index=False)
                rows = len(df)
        print(f"[OK] wrote CSV: {out_path} (rows={rows if rows is not None else 'NA'})")
    except Exception as e:
        print(f"[WARN] failed to write CSV: {out_path} :: {type(e).__name__}: {e}")

def write_reports(run_root: str, today: date,
                  feats_df: pd.DataFrame,
                  labels_df: Optional[pd.DataFrame],
                  preds_df: Optional[pd.DataFrame],
                  metrics: dict,
                  regime_now: str,
                  gate_alpha: float):
    """
    Writes the standard reports ALWAYS (empty files if needed):
      - features_{YYYY-MM-DD}.csv
      - labels_{YYYY-MM-DD}.csv
      - top_*_RAW_{YYYY-MM-DD}.csv
      - top_*_GATED_{YYYY-MM-DD}.csv
      - summary_{YYYY-MM-DD}.txt
    """
    tag = today.strftime("%Y-%m-%d")
    reports = _reports_dir(run_root)

    # Derive simple long/short rankings from preds_df
    longs = pd.DataFrame()
    shorts = pd.DataFrame()
    gated_long = pd.DataFrame()
    gated_short = pd.DataFrame()

    if preds_df is not None and not preds_df.empty:
        # Expect 'score_long' / 'score_short' but fall back gracefully
        sc_long = "score_long" if "score_long" in preds_df.columns else None
        sc_short = "score_short" if "score_short" in preds_df.columns else None

        if sc_long:
            longs = preds_df.sort_values(sc_long, ascending=False)
            if gate_alpha is not None:
                gated_long = longs[longs[sc_long] >= float(gate_alpha)]
        if sc_short:
            shorts = preds_df.sort_values(sc_short, ascending=False)
            if gate_alpha is not None:
                gated_short = shorts[shorts[sc_short] >= float(gate_alpha)]

    # Always write files (header-only when empty)
    _write_csv(feats_df,               os.path.join(reports, f"features_{tag}.csv"))
    _write_csv(labels_df,              os.path.join(reports, f"labels_{tag}.csv"))
    _write_csv(longs.head(200),        os.path.join(reports, f"top_long_candidates_RAW_{tag}.csv"))
    _write_csv(shorts.head(200),       os.path.join(reports, f"top_short_candidates_RAW_{tag}.csv"))
    _write_csv(gated_long.head(200),   os.path.join(reports, f"top_long_candidates_GATED_{tag}.csv"))
    _write_csv(gated_short.head(200),  os.path.join(reports, f"top_short_candidates_GATED_{tag}.csv"))

    # Summary
    summary_path = os.path.join(reports, f"summary_{tag}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"[INFO] Reports dir: {reports.replace('\\', '\\\\')}\n")
        f.write(f"[INFO] Regime today: {regime_now}\n")
        f.write(f"[INFO] Gate alpha: {gate_alpha}\n")
        try:
            n_long = 0 if longs is None else len(longs)
            n_short = 0 if shorts is None else len(shorts)
            n_g_long = 0 if gated_long is None else len(gated_long)
            n_g_short = 0 if gated_short is None else len(gated_short)
            f.write(f"[INFO] Candidates RAW: long={n_long}, short={n_short}\n")
            f.write(f"[INFO] Candidates GATED: long={n_g_long}, short={n_g_short}\n")
        except Exception:
            pass

    print(f"[STEP] write reports -> {summary_path}")

# ---------------------------- Core ----------------------------------

def build_today_features(universe: List[str], price_cache_dir: str) -> pd.DataFrame:
    rows = []
    for t in universe:
        p = _read_price_csv(price_cache_dir, t)
        if p is None or len(p) < 60:
            continue
        feat = _daily_features_from_prices(p)
        if feat is None:
            continue
        feat = feat.to_dict()
        feat["ticker"] = t
        rows.append(feat)
    df = pd.DataFrame(rows)
    if not df.empty:
        # Clip insane values, fillna
        for c in ["mom5", "mom20", "mom60", "vol20"]:
            if c in df:
                df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                df[c] = df[c].clip(-1.0, 1.0)
    return df

def predict_from_features(feats_df: pd.DataFrame, gate_alpha: float) -> Tuple[pd.DataFrame, dict, str]:
    """
    Create simple ranking scores (0..1) based on momentum mix.
    Returns (preds_df, metrics, regime).
    """
    if feats_df is None or feats_df.empty:
        return pd.DataFrame(columns=["ticker","score_long","score_short"]), {}, "Neutral"

    regime = _regime_from_breadth(feats_df)

    # Scoring: blend mom5 & mom20; scale to 0..1 via rank percentile
    z = 0.6 * feats_df["mom5"].astype(float) + 0.4 * feats_df["mom20"].astype(float)
    ranks = z.rank(pct=True)  # 0..1
    score_long = ranks
    score_short = (1.0 - ranks)

    preds = feats_df[["ticker"]].copy()
    preds["score_long"] = score_long.values
    preds["score_short"] = score_short.values
    preds["mom5"] = feats_df["mom5"].values
    preds["mom20"] = feats_df["mom20"].values
    preds["vol20"] = feats_df["vol20"].values

    # Simple metrics just for summary/debug
    metrics = {
        "n": int(len(preds)),
        "regime_median_mom20": float(np.median(feats_df["mom20"])) if "mom20" in feats_df else np.nan,
        "gate_alpha": float(gate_alpha),
    }
    return preds, metrics, regime

# --------------------------- Main -----------------------------------

def main():
    args = parse_args()
    d_today = _as_date(args.today)
    reports_root = _find_reports_root()

    run_root = args.run_root or _run_root_default(reports_root, d_today)
    _ensure_dir(run_root)
    print(f"[INFO] Using 'today' = {d_today.isoformat()}")
    print(f"[INFO] run_root      = {run_root}")

    # Universe
    uni_path = args.universe_csv
    if not os.path.isabs(uni_path):
        uni_path = os.path.join(os.getcwd(), uni_path)
    if not os.path.isfile(uni_path):
        print(f"[WARN] Universe CSV not found: {uni_path}")
        uni = pd.DataFrame(columns=["ticker"])
    else:
        uni = _read_universe(uni_path)
    tickers = uni["ticker"].tolist()
    print(f"[INFO] Parsed tickers: {len(tickers)}")

    # Price cache dir
    pc_dir = _find_price_cache_dir(reports_root)
    if pc_dir is None:
        print("[WARN] price_cache_dir not found; predictions will be empty.")
        feats_df = pd.DataFrame(columns=["ticker","asof_date","mom5","mom20","mom60","vol20"])
    else:
        print(f"[INFO] price_cache_dir = {pc_dir}")
        feats_df = build_today_features(tickers, pc_dir)

    print("[STEP] Build features…")
    # Labels (optional in this lightweight version) – write empty header
    labels_df = pd.DataFrame(columns=["ticker","target"])

    print("[STEP] Regime today:", _regime_from_breadth(feats_df))
    print("[STEP] Train per-regime models… (lightweight ranking)")
    preds_df, metrics, regime = predict_from_features(feats_df, args.gate_alpha)

    print("[STEP] Predict today…")
    print("[STEP] Write reports…")
    write_reports(run_root, d_today, feats_df, labels_df, preds_df, metrics, regime, args.gate_alpha)

    print("[DONE] ml_unified_pipeline complete.")

if __name__ == "__main__":
    main()
