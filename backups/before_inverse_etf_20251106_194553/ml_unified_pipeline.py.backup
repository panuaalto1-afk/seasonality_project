# ml_unified_pipeline.py
# Seasonality project — unified daily ML pipeline (robust writer edition)
# VERSION 5.0: Added Sector Rotation by Regime

import argparse
import os
import sys
import glob
import math
from datetime import datetime, date
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

# ==================== LISÄYKSET: Regime system ====================
from regime_detector import RegimeDetector
from regime_predictor import RegimePredictor
from regime_strategies import RegimeStrategy
from multi_timeframe_regime import MultiTimeframeRegime
from sector_rotation import SectorRotation
# ==================================================================

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
    return os.path.join(reports_root, "runs", tag)

def _find_reports_root() -> str:
    return os.path.join(os.getcwd(), "seasonality_reports")

def _find_price_cache_dir(reports_root: str) -> Optional[str]:
    # 1) Environment override
    env = os.environ.get("PRICE_CACHE_DIR")
    if env and os.path.isdir(env):
        return env

    # 2) Canonical cache
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
    candidates.sort(reverse=True)
    return candidates[0]

def _read_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    cand = None
    for k in ["ticker", "symbol", "ric", "isin"]:
        if k in cols:
            cand = cols[k]
            break
    if cand is None:
        df.columns = ["ticker"] + list(df.columns[1:])
        cand = "ticker"
    df["ticker"] = df[cand].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["ticker"]).drop_duplicates("ticker")
    return df[["ticker"]]

def _read_price_csv(pc_dir: str, ticker: str) -> Optional[pd.DataFrame]:
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
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date", None)
        if date_col is None:
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
    ret = s.pct_change(fill_method=None)
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
                  regime_data: dict,
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
        # Use sector_adjusted_score if available (highest priority)
        if 'sector_adjusted_score' in preds_df.columns:
            sc_long = 'sector_adjusted_score'
        elif 'composite_score' in preds_df.columns:
            sc_long = 'composite_score'
        else:
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
    regime_now = regime_data.get('regime', 'Unknown')
    regime_score = regime_data.get('composite_score', 0.0)
    regime_confidence = regime_data.get('confidence', 0.0)
    
    summary_path = os.path.join(reports, f"summary_{tag}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"[INFO] Reports dir: {reports.replace(os.sep, os.sep*2)}\n")
        f.write(f"\n=== REGIME STATUS ===\n")
        f.write(f"[INFO] Regime today: {regime_now}\n")
        f.write(f"[INFO] Regime score: {regime_score:.4f}\n")
        f.write(f"[INFO] Regime confidence: {regime_confidence:.2%}\n")
        f.write(f"[INFO] Regime duration: {regime_data.get('regime_duration_days', 0)} days\n")
        
        # Multi-timeframe info
        if 'multi_timeframe' in regime_data and regime_data['multi_timeframe']:
            mtf = regime_data['multi_timeframe']
            f.write(f"\n=== MULTI-TIMEFRAME ANALYSIS ===\n")
            f.write(f"[INFO] Daily regime:   {mtf['daily']}\n")
            f.write(f"[INFO] Weekly regime:  {mtf['weekly']}\n")
            f.write(f"[INFO] Monthly regime: {mtf['monthly']}\n")
            f.write(f"[INFO] Composite:      {mtf['composite']} ({mtf['alignment']} alignment)\n")
            f.write(f"[INFO] Trading bias:   {mtf['bias'].upper()} ({mtf['bias_strength']:.1%})\n")
            f.write(f"[INFO] Recommendation: {mtf['recommendation'].upper()}\n")
        
        if 'regime_strategy' in metrics:
            f.write(f"\n=== STRATEGY ===\n")
            f.write(f"[INFO] Strategy type: {metrics['regime_strategy']}\n")
        
        # ==================== Sector rotation info ====================
        if metrics.get('sector_rotation_applied', False):
            f.write(f"\n=== SECTOR ROTATION ===\n")
            f.write(f"[INFO] Sector rotation: ENABLED\n")
            
            # Show top sectors (if we have sector column in preds_df)
            if preds_df is not None and not preds_df.empty and 'sector' in preds_df.columns:
                top_sectors = preds_df['sector'].value_counts().head(5)
                f.write(f"[INFO] Top sectors in signals:\n")
                for sector, count in top_sectors.items():
                    if pd.notna(sector):
                        f.write(f"  - {sector}: {count} signals\n")
        # ==============================================================
        
        if 'prediction_1d' in regime_data:
            f.write(f"\n=== REGIME FORECAST ===\n")
            f.write(f"[INFO] 1-day forecast: {regime_data['prediction_1d']} ({regime_data.get('prediction_1d_prob', 0):.1%})\n")
            f.write(f"[INFO] 5-day forecast: {regime_data['prediction_5d']} ({regime_data.get('prediction_5d_prob', 0):.1%})\n")
            f.write(f"[INFO] Transition probability (5d): {regime_data.get('transition_prob_5d', 0):.1%}\n")
            
            if regime_data.get('transition_prob_5d', 0) > 0.70:
                f.write(f"[WARNING] High regime change probability - consider defensive positioning\n")
        
        f.write(f"\n=== TRADING PARAMETERS ===\n")
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
        for c in ["mom5", "mom20", "mom60", "vol20"]:
            if c in df:
                df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                df[c] = df[c].clip(-1.0, 1.0)
    return df

def predict_from_features(feats_df: pd.DataFrame, regime_data: dict, gate_alpha: float) -> Tuple[pd.DataFrame, dict]:
    """
    Create simple ranking scores (0..1) based on momentum mix.
    Now includes regime-specific filtering, composite scoring, AND sector rotation.
    Returns (preds_df, metrics).
    """
    if feats_df is None or feats_df.empty:
        return pd.DataFrame(columns=["ticker","score_long","score_short"]), {}

    regime = regime_data.get('regime', 'Unknown')

    # Load regime strategy
    strategy = RegimeStrategy(regime)
    
    print(f"\n[INFO] Applying {regime} strategy ({strategy.config['strategy_type']})...")
    print(f"  Signal weights: momentum={strategy.config['signal_weights']['momentum']:.0%}, "
          f"quality={strategy.config['signal_weights']['quality']:.0%}, "
          f"value={strategy.config['signal_weights']['value']:.0%}")

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
    
    # Regime info
    preds["regime"] = regime
    preds["regime_score"] = regime_data.get('composite_score', 0.0)
    preds["regime_confidence"] = regime_data.get('confidence', 0.0)
    
    # Add ml_score (use score_long as proxy)
    preds["ml_score"] = preds["score_long"]
    
    # Apply regime composite scoring
    print(f"  Calculating regime-aware composite scores...")
    
    preds = strategy.calculate_composite_score(preds)
    
    print(f"  Signals ranked by composite score: {len(preds)}")
    if len(preds) > 0:
        print(f"  Top 5 composite scores: {preds['composite_score'].head(5).tolist()}")
    
    # ==================== Apply sector rotation ====================
    print(f"  Applying sector rotation for {regime}...")
    
    rotator = SectorRotation()
    
    # Show sector allocation (compact version for pipeline)
    weights = rotator.regime_sector_weights.get(regime, {})
    sorted_sectors = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    preferred = [s for s, w in sorted_sectors if w >= 0.20]
    print(f"  Preferred sectors (≥20%): {', '.join(preferred) if preferred else 'None'}")
    
    # Apply sector rotation (don't limit top_n here, keep all for gating later)
    preds = rotator.apply_sector_rotation(preds, regime, top_n=None)
    
    print(f"  Signals after sector rotation: {len(preds)}")
    if len(preds) > 0:
        print(f"  Top 5 sector-adjusted scores: {preds['sector_adjusted_score'].head(5).tolist()}")
    # ===============================================================
    
    # Sort by sector_adjusted_score
    preds = preds.sort_values('sector_adjusted_score', ascending=False)

    # Simple metrics
    metrics = {
        "n": int(len(preds)),
        "regime_median_mom20": float(np.median(feats_df["mom20"])) if "mom20" in feats_df else np.nan,
        "gate_alpha": float(gate_alpha),
        "regime_strategy": strategy.config['strategy_type'],
        "strategy_max_positions": strategy.config['max_positions'],
        "sector_rotation_applied": True
    }
    
    return preds, metrics

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
    
    # ==================== REGIME DETECTION ====================
    print("\n" + "="*80)
    print("[STEP] REGIME DETECTION")
    print("="*80 + "\n")
    
    try:
        detector = RegimeDetector(
            macro_price_cache_dir=os.path.join(reports_root, "price_cache"),
            equity_price_cache_dir=pc_dir
        )
        regime_data = detector.detect_regime(date=d_today.strftime("%Y-%m-%d"))
        
        print(f"[INFO] Regime: {regime_data['regime']}")
        print(f"[INFO] Score:  {regime_data['composite_score']:.4f}")
        print(f"[INFO] Confidence: {regime_data['confidence']:.2%}")
        
    except Exception as e:
        print(f"[WARN] Regime detection failed: {e}")
        regime_data = {
            'regime': 'Unknown',
            'composite_score': 0.0,
            'confidence': 0.0,
            'components': {}
        }
    
    print("="*80 + "\n")
    # ==========================================================
    
    # ==================== REGIME PREDICTION ====================
    print("\n" + "="*80)
    print("[STEP] REGIME PREDICTION")
    print("="*80 + "\n")
    
    try:
        predictor = RegimePredictor()
        
        pred_1d = predictor.predict(d_today.strftime("%Y-%m-%d"), horizon_days=1)
        pred_5d = predictor.predict(d_today.strftime("%Y-%m-%d"), horizon_days=5)
        
        regime_duration = pred_1d.get('regime_duration_days', 0)
        
        print(f"📈 1-DAY FORECAST:")
        print(f"  Most likely: {pred_1d['most_likely']} ({pred_1d['predictions'][pred_1d['most_likely']]:.1%})")
        print(f"  Transition probability: {pred_1d['transition_probability']:.1%}")
        
        print(f"\n📈 5-DAY FORECAST:")
        print(f"  Most likely: {pred_5d['most_likely']} ({pred_5d['predictions'][pred_5d['most_likely']]:.1%})")
        print(f"  Transition probability: {pred_5d['transition_probability']:.1%}")
        
        if pred_5d['transition_probability'] > 0.70:
            print(f"\n⚠️  WARNING: High regime change probability ({pred_5d['transition_probability']:.1%})")
            print(f"  Current regime has persisted for {regime_duration} days")
            print(f"  Consider adjusting position sizes or risk limits")
        
        regime_data['regime_duration_days'] = regime_duration
        regime_data['prediction_1d'] = pred_1d['most_likely']
        regime_data['prediction_1d_prob'] = pred_1d['predictions'][pred_1d['most_likely']]
        regime_data['prediction_5d'] = pred_5d['most_likely']
        regime_data['prediction_5d_prob'] = pred_5d['predictions'][pred_5d['most_likely']]
        regime_data['transition_prob_5d'] = pred_5d['transition_probability']
        
    except Exception as e:
        print(f"[WARN] Regime prediction failed: {e}")
        regime_data['regime_duration_days'] = 0
        regime_data['prediction_1d'] = regime_data.get('regime', 'Unknown')
        regime_data['prediction_5d'] = regime_data.get('regime', 'Unknown')
        regime_data['prediction_1d_prob'] = 0.0
        regime_data['prediction_5d_prob'] = 0.0
        regime_data['transition_prob_5d'] = 0.0
    
    print("="*80 + "\n")
    # ===========================================================
    
    # ==================== MULTI-TIMEFRAME REGIME ====================
    print("\n" + "="*80)
    print("[STEP] MULTI-TIMEFRAME REGIME ANALYSIS")
    print("="*80 + "\n")
    
    try:
        mtf = MultiTimeframeRegime(
            macro_price_cache_dir=os.path.join(reports_root, "price_cache"),
            equity_price_cache_dir=pc_dir
        )
        
        mtf_result = mtf.detect_multi_timeframe(d_today.strftime("%Y-%m-%d"))
        
        # Trading bias
        bias = mtf.get_trading_bias(mtf_result)
        
        print(f"\n📊 MULTI-TIMEFRAME SUMMARY:")
        print(f"  Daily:   {mtf_result['daily']['regime']}")
        print(f"  Weekly:  {mtf_result['weekly']['regime']}")
        print(f"  Monthly: {mtf_result['monthly']['regime']}")
        print(f"\n  Composite: {mtf_result['composite']['regime']} ({mtf_result['composite']['alignment']} alignment)")
        print(f"  Trading Bias: {bias['bias'].upper()} ({bias['strength']:.1%} strength)")
        print(f"  Recommendation: {bias['recommendation'].upper()}")
        print(f"\n  → {mtf_result['interpretation']}")
        
        # Add mtf_result to regime_data
        regime_data['multi_timeframe'] = {
            'daily': mtf_result['daily']['regime'],
            'weekly': mtf_result['weekly']['regime'],
            'monthly': mtf_result['monthly']['regime'],
            'composite': mtf_result['composite']['regime'],
            'alignment': mtf_result['composite']['alignment'],
            'bias': bias['bias'],
            'bias_strength': bias['strength'],
            'recommendation': bias['recommendation']
        }
        
    except Exception as e:
        print(f"[WARN] Multi-timeframe analysis failed: {e}")
        regime_data['multi_timeframe'] = None
    
    print("="*80 + "\n")
    # ================================================================

    # Labels (optional)
    labels_df = pd.DataFrame(columns=["ticker","target"])

    print("[STEP] Train per-regime models… (regime + sector aware ranking)")
    preds_df, metrics = predict_from_features(feats_df, regime_data, args.gate_alpha)

    print("[STEP] Predict today…")
    print("[STEP] Write reports…")
    write_reports(run_root, d_today, feats_df, labels_df, preds_df, metrics, regime_data, args.gate_alpha)

    print("[DONE] ml_unified_pipeline complete.")

if __name__ == "__main__":
    main()