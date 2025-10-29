# -*- coding: utf-8 -*-
"""
optio_seasonality_signal.py — v1.1 (price-clean + safer options)

Korjaa:
- price_cache CSV-rivin "tikkeri toisella rivillä" -> puhdistus
- Spot aina hinnasta, ei info/fast_info
- pehmeämmät optio-suodattimet
- ei kirjata fallback-nollarivejä

"""

import os, sys, math, time, glob, traceback
import datetime as dt
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("Please install yfinance: pip install yfinance")
    raise

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# -----------------------------
# Konfiguraatio (päivitä vain polut tarvittaessa)
# -----------------------------

BASE_DIR = r"C:\Users\panua\seasonality_project"
REPORTS_DIR = os.path.join(BASE_DIR, "seasonality_reports")
AGG_DIR = os.path.join(REPORTS_DIR, "aggregates")

# KÄYTÄ SAMA price_cache KUIN toimi aiemmin
PRICE_CACHE_DIR = r"C:\Users\panua\seasonality_project\seasonality_reports\runs\2025-10-04_0903\price_cache"

TODAY = dt.datetime.now().strftime("%Y-%m-%d")
OUT_DIR = os.path.join(AGG_DIR, "optio_signals", TODAY)

UNIVERSE_CSV = os.path.join(AGG_DIR, "Constituents_raw.csv")
USE_FALLBACK_UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA"]

# Likviditeettifiltterit
MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOL = 5_000_000
MAX_TICKERS = 700

# Optioiden haku – hieman pehmeämmäksi
MAX_EXPIRIES   = 4
MONEYNESS_PCT  = 0.20
MIN_OPT_OI     = 100       # aiemmin 500
MAX_SPREAD_PCT = 0.25      # aiemmin 0.10

# Seasonality (jos käytössäsi)
NEAR_WINDOW_DAYS = 5
SEAS_FILES = {
    "candidates": os.path.join(AGG_DIR, "candidates_windows_ranked.csv"),
    "upcoming"  : os.path.join(AGG_DIR, "upcoming_windows_next_45d.csv"),
    "window_map": os.path.join(AGG_DIR, "window_to_dates.csv"),
}

# Painot
W_MPI  = 0.35
W_DBS  = 0.25
W_COMP = 0.20
W_SEAS = 0.15
W_MKT  = 0.05

TOP_N = 60

CACHE_DIR    = os.path.join(AGG_DIR, "optio_signals", "_cache")
IV_OI_CACHE  = os.path.join(CACHE_DIR, "iv_oi_cache.csv")


# -----------------------------
# Yleisapuja
# -----------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None

def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Siivoa price_cache: pakota päivämäärä ja numerot, poista tikkeririvi."""
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()

    # yhtenäistä sarakenimiä
    rename = {}
    cols_l = {c.lower(): c for c in d.columns}
    def pick(*names):
        for n in names:
            if n in cols_l: return cols_l[n]
        return None
    for want, cands in {
        "Date": ("date","time","timestamp"),
        "Open": ("open",),
        "High": ("high",),
        "Low" : ("low",),
        "Close":("close","adj close","adj_close","adjclose"),
        "Volume":("volume",)
    }.items():
        cur = pick(*cands)
        if cur and cur != want: rename[cur] = want
    if rename: d = d.rename(columns=rename)

    if "Date" not in d.columns or "Close" not in d.columns:
        return pd.DataFrame()

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ("Open","High","Low","Close","Volume"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["Date","Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return d.reset_index(drop=True)

def bb_width_pct(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> Tuple[float, bool, bool]:
    if df is None or df.empty or len(df) < window + 2: return np.nan, False, False
    close = df["Close"].astype(float)
    m = close.rolling(window).mean()
    s = close.rolling(window).std(ddof=0)
    upper = m + n_std * s
    lower = m - n_std * s
    width = (upper - lower) / close
    width_pct = float(100.0 * width.iloc[-1])
    broke_up   = bool(close.iloc[-1] > upper.iloc[-1] and close.iloc[-2] <= upper.iloc[-2])
    broke_down = bool(close.iloc[-1] < lower.iloc[-1] and close.iloc[-2] >= lower.iloc[-2])
    return width_pct, broke_up, broke_down

def atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    if df is None or len(df) < window + 1: return np.nan
    high = df["High"].fillna(df["Close"]).astype(float)
    low  = df["Low"] .fillna(df["Close"]).astype(float)
    close= df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return float(100.0 * (atr.iloc[-1] / close.iloc[-1]))

def load_price_history_from_cache(ticker: str) -> Optional[pd.DataFrame]:
    pattern = os.path.join(PRICE_CACHE_DIR, f"{ticker}*.csv")
    files = glob.glob(pattern)
    if not files: return None
    latest = max(files, key=os.path.getmtime)
    try:
        df = pd.read_csv(latest)
        df = clean_price_df(df)
        return df
    except Exception:
        return None

def read_universe() -> List[str]:
    df = safe_read_csv(UNIVERSE_CSV)
    tickers = []
    if df is not None and not df.empty:
        cand_cols = [c for c in df.columns if c.lower() in ("symbol","ticker","tickers","sym")]
        if cand_cols:
            col = cand_cols[0]
            tickers = list(map(lambda x: str(x).strip().upper(), df[col].dropna().unique().tolist()))
    if not tickers:
        tickers = USE_FALLBACK_UNIVERSE
    tickers = [t for t in tickers if t.isalnum() or ("/" not in t and "^" not in t)]
    if len(tickers) > MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    return tickers

def load_seasonality_features() -> pd.DataFrame:
    df = pd.DataFrame(columns=["ticker","SEAS_on","SEAS_strength","SEAS_days_to_open","SEAS_trend"])
    upc = safe_read_csv(SEAS_FILES["upcoming"])
    if upc is not None and not upc.empty:
        tc = None
        for c in upc.columns:
            if c.lower() in ("ticker","symbol"):
                tc = c; break
        if tc is not None:
            days_col = next((c for c in upc.columns if "days" in c.lower()), None)
            trend_col= next((c for c in upc.columns if "trend" in c.lower() or "bias" in c.lower()), None)
            strength_col = next((c for c in upc.columns if "strength" in c.lower() or "score" in c.lower()), None)
            tmp = pd.DataFrame()
            tmp["ticker"] = upc[tc].astype(str).str.upper()
            tmp["SEAS_days_to_open"] = upc[days_col] if days_col else 0
            tmp["SEAS_trend"] = upc[trend_col] if trend_col else "UP"
            tmp["SEAS_strength"] = upc[strength_col] if strength_col else 50
            tmp["SEAS_on"] = (pd.to_numeric(tmp["SEAS_days_to_open"], errors="coerce").fillna(999).astype(float) <= NEAR_WINDOW_DAYS).astype(int)
            df = tmp.copy()
    return df

def read_cache() -> pd.DataFrame:
    ensure_dir(CACHE_DIR)
    if os.path.exists(IV_OI_CACHE):
        try:
            return pd.read_csv(IV_OI_CACHE)
        except Exception:
            pass
    return pd.DataFrame(columns=["ticker","iv30","oi_total","asof"])

def write_cache(df: pd.DataFrame):
    ensure_dir(CACHE_DIR)
    try:
        df.to_csv(IV_OI_CACHE, index=False)
    except Exception as e:
        print("Cache write failed:", e)

# -----------------------------
# Optio- ja price-ominaisuudet
# -----------------------------
def compute_option_features(ticker: str, spot_hint: Optional[float]) -> Dict[str, float]:
    out = dict(spot=np.nan, iv7=np.nan, iv45=np.nan, iv30=np.nan, term_slope=np.nan,
               call_vol=np.nan, put_vol=np.nan, pcr=np.nan, oi_total=np.nan,
               skew25=np.nan, mid_spread_flag=0.0)
    try:
        tk = yf.Ticker(ticker)
        # spot mieluummin hinnasta
        spot = spot_hint
        if spot is None or not np.isfinite(spot):
            spot = tk.fast_info.get("lastPrice") if hasattr(tk, "fast_info") else None
        if spot is None and hasattr(tk, "info"):
            spot = tk.info.get("regularMarketPrice")
        if spot is None or not np.isfinite(spot):
            return out
        out["spot"] = float(spot)

        try:
            expiries = tk.options or []
        except Exception:
            expiries = []
        if not expiries:
            return out

        exp_dates = []
        for e in expiries:
            try:
                exp_dates.append(dt.datetime.strptime(e, "%Y-%m-%d").date())
            except Exception:
                continue
        exp_dates = sorted(exp_dates)[:MAX_EXPIRIES]
        if not exp_dates:
            return out

        total_call_vol = 0.0
        total_put_vol  = 0.0
        total_oi       = 0.0
        seen_wide_spread = False
        iv_by_dte, iv_call_approx_1p, iv_put_approx_1p = [], [], []

        for ed in exp_dates:
            dte = (ed - dt.date.today()).days
            if dte <= 0: continue
            try:
                chain = tk.option_chain(ed.strftime("%Y-%m-%d"))
                calls = chain.calls.copy()
                puts  = chain.puts.copy()
            except Exception:
                continue
            for side_df, is_call in [(calls, True), (puts, False)]:
                if side_df is None or side_df.empty: continue
                for col in ["impliedVolatility","bid","ask","lastPrice","strike","volume","openInterest"]:
                    if col not in side_df.columns:
                        side_df[col] = np.nan
                m_lo, m_hi = (1.0 - MONEYNESS_PCT) * spot, (1.0 + MONEYNESS_PCT) * spot
                sd = side_df[(side_df["strike"] >= m_lo) & (side_df["strike"] <= m_hi)].copy()
                if sd.empty: continue

                mid = (sd["bid"].fillna(0) + sd["ask"].fillna(0)) / 2.0
                spr = (sd["ask"].fillna(0) - sd["bid"].fillna(0))
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.where(mid > 0, spr / mid, np.nan)
                if np.nanmax(rel) > MAX_SPREAD_PCT:
                    seen_wide_spread = True

                sd = sd[pd.to_numeric(sd["openInterest"], errors="coerce").fillna(0) >= MIN_OPT_OI]
                if sd.empty: continue

                iv_med = float(np.nanmedian(pd.to_numeric(sd["impliedVolatility"], errors="coerce"))) if pd.to_numeric(sd["impliedVolatility"], errors="coerce").notna().any() else np.nan
                vol_sum = float(pd.to_numeric(sd["volume"], errors="coerce").fillna(0).sum())
                oi_sum  = float(pd.to_numeric(sd["openInterest"], errors="coerce").fillna(0).sum())
                total_oi += oi_sum
                if is_call: total_call_vol += vol_sum
                else:       total_put_vol  += vol_sum
                if np.isfinite(iv_med): iv_by_dte.append((dte, iv_med))
                if is_call:
                    call_band = sd[sd["strike"] >= 1.10 * spot]
                    if not call_band.empty and pd.to_numeric(call_band["impliedVolatility"], errors="coerce").notna().any():
                        iv_call_approx_1p.append(float(np.nanmedian(pd.to_numeric(call_band["impliedVolatility"], errors="coerce"))))
                else:
                    put_band = sd[sd["strike"] <= 0.90 * spot]
                    if not put_band.empty and pd.to_numeric(put_band["impliedVolatility"], errors="coerce").notna().any():
                        iv_put_approx_1p.append(float(np.nanmedian(pd.to_numeric(put_band["impliedVolatility"], errors="coerce"))))

        out["call_vol"] = total_call_vol
        out["put_vol"]  = total_put_vol
        out["oi_total"] = total_oi
        if total_call_vol + total_put_vol > 0:
            out["pcr"] = float(total_put_vol / max(total_call_vol, 1e-9))
        out["mid_spread_flag"] = 1.0 if seen_wide_spread else 0.0

        if iv_by_dte:
            arr = np.array(iv_by_dte, dtype=float)
            dtes, ivs = arr[:,0], arr[:,1]
            def pick(target):
                idx = int(np.argmin(np.abs(dtes - target)))
                return float(ivs[idx])
            out["iv7"]  = pick(7.0)  if len(arr) else np.nan
            out["iv45"] = pick(45.0) if len(arr) else np.nan
            out["iv30"] = pick(30.0) if len(arr) else np.nan
            if np.isfinite(out["iv7"]) and np.isfinite(out["iv45"]):
                out["term_slope"] = float(out["iv7"] - out["iv45"])

        if iv_put_approx_1p and iv_call_approx_1p:
            out["skew25"] = float(np.nanmedian(iv_put_approx_1p) - np.nanmedian(iv_call_approx_1p))

    except Exception:
        pass
    return out

def compression_trigger_features(ticker: str) -> Dict[str, float]:
    out = {"bb_width_pct": np.nan, "atr_pct": np.nan, "trigger_up": 0.0, "trigger_down": 0.0}
    df = load_price_history_from_cache(ticker)
    if df is None or df.empty: return out
    if "High" not in df.columns: df["High"] = df["Close"]
    if "Low"  not in df.columns: df["Low"]  = df["Close"]
    b, up, dn = bb_width_pct(df, window=20, n_std=2.0)
    a = atr_pct(df, window=14)
    out["bb_width_pct"] = b
    out["atr_pct"] = a
    out["trigger_up"] = 100.0 if up else 0.0
    out["trigger_down"] = 100.0 if dn else 0.0
    return out

def seas_component(seas_row: pd.Series, near_window_days:int=5) -> float:
    if seas_row is None or seas_row.empty: return 0.0
    on = int(pd.to_numeric(seas_row.get("SEAS_on", 0), errors="coerce") or 0)
    strength = float(pd.to_numeric(seas_row.get("SEAS_strength", 50.0), errors="coerce") or 50.0)
    days = float(pd.to_numeric(seas_row.get("SEAS_days_to_open", 0.0), errors="coerce") or 0.0)
    base = 50.0 if on else max(0.0, 50.0 - 3.0 * max(0.0, days - near_window_days))
    return float(min(100.0, base + 0.5 * strength))

def compute_scores(opt: Dict[str, float], comp: Dict[str, float], seas_row: Optional[pd.Series], cache_prev: Optional[pd.Series]) -> Dict[str, float]:
    comp_score = 0.0
    bbw = comp.get("bb_width_pct", np.nan)
    atrp= comp.get("atr_pct", np.nan)
    if np.isfinite(bbw): comp_score += float(np.interp(bbw, [1.0, 12.0], [100.0, 0.0]))
    if np.isfinite(atrp): comp_score += float(np.interp(atrp, [0.3, 5.0], [100.0, 0.0]))
    comp_score *= 0.5
    if comp.get("trigger_up",0.0) >= 100.0 or comp.get("trigger_down",0.0) >= 100.0:
        comp_score = max(comp_score, 85.0)

    iv30 = opt.get("iv30", np.nan)
    term = opt.get("term_slope", np.nan)
    pcr  = opt.get("pcr", np.nan)
    oi_total = opt.get("oi_total", np.nan)
    skew25   = opt.get("skew25", np.nan)

    d_iv30, d_oi = np.nan, np.nan
    if cache_prev is not None and not isinstance(cache_prev, float):
        prev_iv = cache_prev.get("iv30", np.nan)
        prev_oi = cache_prev.get("oi_total", np.nan)
        if np.isfinite(iv30) and np.isfinite(prev_iv) and prev_iv > 0:
            d_iv30 = 100.0 * (iv30 - prev_iv) / prev_iv
        if np.isfinite(oi_total) and np.isfinite(prev_oi) and prev_oi > 0:
            d_oi = 100.0 * (oi_total - prev_oi) / prev_oi

    mpi = 0.0
    if np.isfinite(term):  mpi += float(np.interp(term, [-0.05, 0.00, 0.10], [0.0, 40.0, 90.0]))
    if np.isfinite(d_iv30):mpi += float(np.interp(d_iv30, [-10.0, 0.0, 20.0], [0.0, 15.0, 35.0]))
    if np.isfinite(d_oi):  mpi += float(np.interp(d_oi, [-20.0, 0.0, 30.0], [0.0, 10.0, 25.0]))
    if opt.get("mid_spread_flag", 0.0) >= 1.0: mpi *= 0.85
    mpi = float(max(0.0, min(100.0, mpi)))

    dbs = 50.0
    if np.isfinite(pcr):  dbs += float(np.interp(pcr, [0.4, 1.0, 2.5], [25.0, 0.0, -25.0]))
    if np.isfinite(skew25):dbs += float(np.interp(skew25, [-0.05, 0.0, 0.10], [15.0, 0.0, -15.0]))
    if np.isfinite(term): dbs += float(np.interp(term, [-0.05, 0.0, 0.10], [-10.0, 0.0, 10.0]))
    dbs = float(max(0.0, min(100.0, dbs)))

    ei = 0.0
    if np.isfinite(d_iv30): ei += float(np.interp(d_iv30, [-25.0, -10.0, 0.0], [90.0, 60.0, 0.0]))
    if np.isfinite(d_oi):  ei += float(np.interp(d_oi,   [-40.0, -10.0, 0.0], [60.0, 25.0, 0.0]))
    if comp.get("trigger_up",0.0) < 100.0 and comp.get("trigger_down",0.0) < 100.0:
        ei += 10.0
    ei = float(max(0.0, min(100.0, ei)))

    seas = seas_component(seas_row, near_window_days=5) if seas_row is not None else 0.0
    mkt  = 50.0

    priority = (W_MPI*mpi) + (W_DBS*dbs) + (W_COMP*comp_score) + (W_SEAS*seas) + (W_MKT*mkt)
    priority = float(max(0.0, min(100.0, priority)))

    return {
        "MPI": round(mpi,1),
        "DBS": round(dbs,1),
        "EI":  round(ei,1),
        "Compression": round(comp_score,1),
        "SEAS": round(seas,1),
        "Priority": round(priority,1),
        "dIV30_pct": round(d_iv30,2) if np.isfinite(d_iv30) else np.nan,
        "dOI_pct":   round(d_oi,2)   if np.isfinite(d_oi)   else np.nan,
    }

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(OUT_DIR)
    print(f"[INFO] Output folder: {OUT_DIR}")

    tickers = read_universe()
    print(f"[INFO] Universe tickers: {len(tickers)} (cap {MAX_TICKERS})")

    # Likviditeettifiltterit hintacachesta
    filtered = []
    for t in tickers:
        try:
            dfp = load_price_history_from_cache(t)
            if dfp is None or dfp.empty:
                continue
            last = float(dfp["Close"].iloc[-1])
            if last < MIN_PRICE: continue
            if MIN_AVG_DOLLAR_VOL > 0:
                v_mean = pd.to_numeric(dfp.get("Volume", pd.Series(np.nan)), errors="coerce").tail(20).mean()
                dv = float(last * (v_mean if np.isfinite(v_mean) else 0))
                if np.isfinite(dv) and dv < MIN_AVG_DOLLAR_VOL:
                    continue
            filtered.append(t)
        except Exception:
            continue

    tickers = filtered
    print(f"[INFO] After basic liquidity filters: {len(tickers)}")

    seas_df = load_seasonality_features()
    seas_map = {row["ticker"]: row for _, row in (seas_df.iterrows() if seas_df is not None and not seas_df.empty else [])}

    cache = read_cache()
    cache_map = {}
    if cache is not None and not cache.empty:
        cache["ticker"] = cache["ticker"].astype(str).str.upper()
        cache_map = {r["ticker"]: r for _, r in cache.iterrows()}

    rows = []
    iterable = tqdm(tickers, desc="Processing") if TQDM else tickers
    for t in iterable:
        try:
            price_df = load_price_history_from_cache(t)
            spot_hint = float(price_df["Close"].iloc[-1]) if price_df is not None and not price_df.empty else np.nan

            opt  = compute_option_features(t, spot_hint)
            comp = compression_trigger_features(t)
            seas_row   = seas_map.get(t, None)
            cache_prev = cache_map.get(t, None)
            scores     = compute_scores(opt, comp, seas_row, cache_prev)

            # Jos mitään järkevää ei saatu, ohitetaan rivi
            if not np.isfinite(opt.get("spot", np.nan)):
                print(f"[WARN] {t}: ei spot-hintaa -> skip")
                continue
            if all((scores["MPI"]==0.0, scores["DBS"]==0.0, np.isnan(comp.get("bb_width_pct", np.nan)))):
                print(f"[WARN] {t}: ei kelvollisia featureita -> skip")
                continue

            row = {
                "Ticker": t,
                "Spot": round(opt.get("spot", np.nan), 2) if np.isfinite(opt.get("spot", np.nan)) else np.nan,
                "Priority": scores["Priority"],
                "MPI": scores["MPI"],
                "DBS": scores["DBS"],
                "Compression": scores["Compression"],
                "SEAS": scores["SEAS"],
                "EI": scores["EI"],
                "dIV30_pct": scores["dIV30_pct"],
                "dOI_pct": scores["dOI_pct"],
                "PCR": round(opt.get("pcr", np.nan), 3) if np.isfinite(opt.get("pcr", np.nan)) else np.nan,
                "TermSlope": round(opt.get("term_slope", np.nan), 4) if np.isfinite(opt.get("term_slope", np.nan)) else np.nan,
                "Skew25": round(opt.get("skew25", np.nan), 4) if np.isfinite(opt.get("skew25", np.nan)) else np.nan,
                "BB_width_pct": round(comp.get("bb_width_pct", np.nan), 2) if np.isfinite(comp.get("bb_width_pct", np.nan)) else np.nan,
                "ATR_pct": round(comp.get("atr_pct", np.nan), 2) if np.isfinite(comp.get("atr_pct", np.nan)) else np.nan,
                "TriggerUp": int(comp.get("trigger_up", 0.0) >= 100.0),
                "TriggerDown": int(comp.get("trigger_down", 0.0) >= 100.0),
                "SEAS_on": int(seas_row.get("SEAS_on", 0)) if seas_row is not None else 0,
                "SEAS_days_to_open": int(pd.to_numeric(seas_row.get("SEAS_days_to_open", 0), errors="coerce")) if seas_row is not None else np.nan,
                "SEAS_trend": str(seas_row.get("SEAS_trend", "")) if seas_row is not None else "",
            }
            rows.append(row)
        except Exception:
            traceback.print_exc()
            continue

    if not rows:
        print("[WARN] No rows produced. Check data sources / universe / options availability.")
        return

    df = pd.DataFrame(rows)

    def save_list(df_sub: pd.DataFrame, name: str):
        path_csv  = os.path.join(OUT_DIR, f"{name}.csv")
        path_html = os.path.join(OUT_DIR, f"{name}.html")
        df_sub.to_csv(path_csv, index=False)
        html = df_sub.to_html(index=False)
        with open(path_html, "w", encoding="utf-8") as f:
            f.write(f"<h3>{name}</h3>\n")
            f.write(html)
        return path_csv, path_html

    # Listat — suodatetaan vain todelliset rivit
    long_cand = df[(df["DBS"] >= 55) & (df["Priority"] >= 60) & (df["SEAS_trend"].isin(["UP","NEUTRAL",""]))].copy()
    long_cand = long_cand.sort_values(["Priority","MPI","Compression"], ascending=False).head(TOP_N)
    long_paths = save_list(long_cand, "top_breakout_long")

    short_cand = df[(df["DBS"] <= 45) & (df["Priority"] >= 60)].copy()
    short_cand = short_cand.sort_values(["Priority","MPI","Compression"], ascending=False).head(TOP_N)
    short_paths = save_list(short_cand, "top_breakout_short")

    exit_cand = df[(df["EI"] >= 70) | ((df["TriggerUp"] == 0) & (df["TriggerDown"] == 0) & (df["SEAS_on"] == 1) & (df["SEAS_days_to_open"] <= 1))].copy()
    exit_cand = exit_cand.sort_values(["EI","Priority"], ascending=False).head(TOP_N)
    exit_paths = save_list(exit_cand, "exit_alerts")

    # Päivitä IV/OI -cache
    cache_rows = []
    asof = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for t in df["Ticker"]:
        try:
            price_df = load_price_history_from_cache(t)
            spot_hint = float(price_df["Close"].iloc[-1]) if price_df is not None and not price_df.empty else np.nan
            opt = compute_option_features(t, spot_hint)
            cache_rows.append({"ticker": t, "iv30": opt.get("iv30", np.nan), "oi_total": opt.get("oi_total", np.nan), "asof": asof})
        except Exception:
            cache_rows.append({"ticker": t, "iv30": np.nan, "oi_total": np.nan, "asof": asof})
    cache_df = pd.DataFrame(cache_rows)
    write_cache(cache_df)

    print(f"[DONE] Long list:  {long_paths[0]}")
    print(f"[DONE] Short list: {short_paths[0]}")
    print(f"[DONE] Exit alerts:{exit_paths[0]}")
    print(f"[STATS] kept_rows={len(df)}")
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()


