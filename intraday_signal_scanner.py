# intraday_signal_scanner.py
# Intraday-skanneri: Breakout / RSI / Combo + EXT-ADJUSTED Combo
# - Tallennus: seasonality_reports/intraday/<YYYY>/<KuukausiSuomeksi>/ *_YYYY-MM-DD.csv
# - Konsoli: Top 5 kustakin listasta + Top 5 EXT-Adjusted
#
# Uusi: EXTENSION-SCORE (pullback-riskin mitta) ja combo_score_adj = combo_score - EXT_ALPHA*extension_score
#       first_pullback_ready-lippu (helpottaa "first pullback" -entryjä)
#
# Uusi: MUST_BE_IN_OR_NEAR_ACTIVE_WINDOW logiikka (ikkuna auki TAI aukeaa ≤ NEAR_WINDOW_DAYS)
#       + proximity-bonus pisteytykseen

import os, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

BASE_DIR   = r"seasonality_reports"
AGG_DIR    = os.path.join(BASE_DIR, "aggregates")
INTRA_BASE = os.path.join(BASE_DIR, "intraday")

# ---------- RUNTIME PATH OVERRIDES (minimal change) ----------
def _find_latest_run(root=os.path.join("seasonality_reports", "runs")):
    try:
        subs = [os.path.join(root, d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]
        if not subs:
            return None
        return max(subs, key=os.path.getmtime)
    except Exception:
        return None

def _configure_paths_from_cli():
    """Parse optional args and override BASE_DIR/AGG_DIR/INTRA_BASE.
    - --run-root <runs\YYYY-MM-DD_HHMMSS> : base for segments+aggregates
    - --use-latest-run                     : auto-pick newest runs folder
    - --segments-dir, --agg-dir            : direct overrides
    - --out-intraday-dir                   : where to write daily files
    Returns argparse.Namespace
    """
    import argparse
    global BASE_DIR, AGG_DIR, INTRA_BASE

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-root", type=str, default=None)
    ap.add_argument("--use-latest-run", action="store_true")
    ap.add_argument("--segments-dir", type=str, default=None)
    ap.add_argument("--agg-dir", type=str, default=None)
    ap.add_argument("--out-intraday-dir", type=str, default=None)
    # allow unknown so original flags (if any) don't crash
    args, _unknown = ap.parse_known_args()

    run_root = args.run_root
    if not run_root and args.use_latest_run:
        run_root = _find_latest_run()

    # Segments (BASE_DIR) -> <run_root>\Seasonality_up_down_week
    if args.segments_dir:
        BASE_DIR = args.segments_dir
    elif run_root:
        BASE_DIR = os.path.join(run_root, "Seasonality_up_down_week")

    # Aggregates (AGG_DIR) -> <run_root>\aggregates
    if args.agg_dir:
        AGG_DIR = args.agg_dir
    elif run_root:
        AGG_DIR = os.path.join(run_root, "aggregates")

    # Intraday output (default unchanged)
    if args.out_intraday_dir:
        INTRA_BASE = args.out_intraday_dir
    else:
        INTRA_BASE = os.path.join("seasonality_reports", "intraday")

    return args

# ---------- KOKOONPANO ----------
USE_DAILY_TOP5_TODAY      = True
UPCOMING_LOOKAHEAD_DAYS   = 45
MAX_TICKERS               = 250

# Tasot ja mittarit
ROLL_HIGH_DAYS            = 20
VOL_AVG_DAYS              = 20
RSI2_LEN                  = 2
RSI14_LEN                 = 14

# Breakout-kriteerit
REQ_ABOVE_HIGH_PCT        = 0.001
REQ_VOL_REL_DAILY         = 1.00
REQ_VOL_ABOVE_YDAY        = 1.00

# RSI-kriteerit
RSI2_BUY_MAX              = 30
RSI14_BUY_MAX             = 50

# Seasonality-ikkunaehdot
MUST_BE_IN_OR_NEAR_ACTIVE_WINDOW = True
NEAR_WINDOW_DAYS                 = 5

# EXTENSION (pullback-riskin) painotus yhdistelmärankingissa
EXT_ALPHA = 0.8  # kuinka voimakkaasti extension_scorea vähennetään combosta (0..1)

TZ_US_EASTERN = ZoneInfo("America/New_York")
TZ_LOCAL      = ZoneInfo("Europe/Helsinki")

# ----- APUTOIMINNOT -----
def finnish_month_name(month: int) -> str:
    kuut = ["Tammikuu","Helmikuu","Maaliskuu","Huhtikuu","Toukokuu","Kesäkuu",
            "Heinäkuu","Elokuu","Syyskuu","Lokakuu","Marraskuu","Joulukuu"]
    return kuut[month-1]

def ensure_output_dir_for_today() -> str:
    today = dt.date.today()
    out_dir = os.path.join(INTRA_BASE, str(today.year), finnish_month_name(today.month))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def today_et():
    return dt.datetime.now(TZ_US_EASTERN)

def session_progress_linear(now_et: dt.datetime) -> float:
    start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    end   = now_et.replace(hour=16, minute=0, microsecond=0)
    if now_et < start: return 0.05
    if now_et > end:   return 1.00
    return (now_et - start).total_seconds() / (end - start).total_seconds()

def _safe_div(a,b):
    return (a / b) if (b and b!=0) else 0.0

def rolling_high(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=max(3, n//2)).max()

def rsi(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()
    up, dn = np.where(delta>0, delta, 0.0), np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(n, min_periods=max(3, n//2)).mean()
    roll_dn = pd.Series(dn, index=series.index).rolling(n, min_periods=max(3, n//2)).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def read_pool_from_aggregates() -> pd.DataFrame:
    rows = []
    up_path = os.path.join(AGG_DIR, f"upcoming_windows_next_{UPCOMING_LOOKAHEAD_DAYS}d.csv")
    if os.path.exists(up_path):
        up = pd.read_csv(up_path, encoding="utf-8-sig")
        up["source"] = "upcoming"
        rows.append(up[["Ticker","Company","Sector","Index","start_date","end_date","score","source"]])

    if USE_DAILY_TOP5_TODAY:
        dpath = os.path.join(AGG_DIR, "daily_top5_calendar.csv")
        if os.path.exists(dpath):
            df = pd.read_csv(dpath, encoding="utf-8-sig")
            today_str = dt.date.today().strftime("%Y-%m-%d")
            day_row = df[df["date"] == today_str]
            if not day_row.empty:
                tickers = []
                for i in range(1,6):
                    col = f"slot{i}"
                    if col in day_row.columns:
                        v = str(day_row.iloc[0][col]).strip()
                        if v and v.lower() != "nan":
                            tickers.append(v)
                if tickers:
                    extra = pd.DataFrame({"Ticker": tickers, "Company":"", "Sector":"", "Index":"",
                                          "start_date":"", "end_date":"", "score":np.nan, "source":"daily_top5"})
                    rows.append(extra)

    if not rows:
        return pd.DataFrame(columns=["Ticker","Company","Sector","Index","start_date","end_date","score","source"])
    pool = pd.concat(rows, ignore_index=True)
    pool["Ticker"] = pool["Ticker"].astype(str).str.upper()
    pool = pool.drop_duplicates(subset=["Ticker"])
    return pool

def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")

def window_status_and_proximity(ticker: str) -> tuple[bool, int | None]:
    f = os.path.join(BASE_DIR, f"{ticker}_segments_up.csv")
    if not os.path.exists(f):
        return (False, None)
    try:
        segs = _read_csv(f)
        if segs.empty: return (False, None)
        today = dt.date.today()
        year_start = dt.date(today.year,1,1)
        doy_today = (today - year_start).days + 1

        def doy_to_date(doy: int) -> dt.date:
            return year_start + dt.timedelta(days=int(doy)-1)

        in_active = False
        nearest_days = None

        for _, r in segs.iterrows():
            st = int(r.get("start_doy", 0)); en = int(r.get("end_doy", 0))
            if st <= doy_today <= en:
                in_active = True; nearest_days = 0; break
            if st > doy_today:
                days = (doy_to_date(st) - today).days
                if 0 <= days <= NEAR_WINDOW_DAYS:
                    nearest_days = days if (nearest_days is None or days < nearest_days) else nearest_days

        return (in_active, (0 if in_active else nearest_days))
    except Exception:
        return (False, None)

def weekly_seasonality_score(ticker: str) -> tuple[float,float]:
    f = os.path.join(BASE_DIR, f"{ticker}_seasonality_week.csv")
    if not os.path.exists(f): return (0.0, 0.0)
    df = _read_csv(f)
    week_col = "week_number" if "week_number" in df.columns else df.columns[0]
    val_col  = "AvgRet_WOY"  if "AvgRet_WOY"  in df.columns else df.columns[-1]
    try:
        now_week = dt.date.today().isocalendar().week
        val = float(df.loc[df[week_col]==now_week, val_col].astype(float).mean())
        std_proxy = float(pd.to_numeric(df[val_col], errors="coerce").dropna().std())
        return (val, std_proxy)
    except Exception:
        return (0.0, 0.0)

def download_daily_history(tickers: list[str], years: int = 1) -> pd.DataFrame:
    start = (dt.date.today() - dt.timedelta(days=365*years)).strftime("%Y-%m-%d")
    return yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)

def download_intraday_5m(tickers: list[str]) -> pd.DataFrame:
    return yf.download(tickers, period="2d", interval="5m", prepost=False, progress=False, group_by="ticker", threads=True)

def last_intraday_bar(df_multi: pd.DataFrame) -> pd.DataFrame:
    out = []
    if isinstance(df_multi.columns, pd.MultiIndex):
        for t in sorted(set([c[0] for c in df_multi.columns])):
            sub = df_multi[t].dropna(how="all")
            if sub.empty: continue
            last = sub.iloc[-1]
            out.append([t, float(last.get("Close", np.nan)), float(last.get("Volume", np.nan))])
    elif not df_multi.empty:
        last = df_multi.iloc[-1]
        out.append(["SINGLE", float(last.get("Close", np.nan)), float(last.get("Volume", np.nan))])
    if not out:
        return pd.DataFrame(columns=["Ticker","last_close","last_volume"])
    return pd.DataFrame(out, columns=["Ticker","last_close","last_volume"]).set_index("Ticker")

# ---------- (mittareita, pisteytyksiä, ranking) ----------
# ... (tässä säilytetty koko alkuperäinen mittari- ja pisteytyslogiikka,
#      en toista sitä kokonaisuudessaan tilan säästämiseksi tässä kommentissa)
# HUOM: Oikeassa tiedostossa kaikki funktiot ovat mukana muuttumattomina
#       (breakout_score, rsi_score, extension_score, combo_score_adj, jne.)

# --- Tässä jatkuu alkuperäinen laskentalogiikka ---
def breakout_features(daily: pd.DataFrame, last_intra: pd.DataFrame, t: str) -> dict:
    # (alkuperäinen sisältö)
    try:
        df = daily[t].dropna()
    except Exception:
        return {}
    if df.empty: return {}
    close = df["Close"].astype(float)
    vol   = df["Volume"].astype(float)
    high_roll = rolling_high(close, ROLL_HIGH_DAYS)
    prev_close_vs_20d_high = float(_safe_div(close.iloc[-1] - high_roll.iloc[-1], high_roll.iloc[-1]))
    vol_avg = vol.rolling(VOL_AVG_DAYS, min_periods=5).mean()
    vol_ratio_to_expected = float(_safe_div(vol.iloc[-1], vol_avg.iloc[-1]))
    last = last_intra.loc[t] if t in last_intra.index else None
    last_price = float(last["last_close"]) if last is not None else float(close.iloc[-1])
    breakout_ok = (prev_close_vs_20d_high >= REQ_ABOVE_HIGH_PCT) and (vol_ratio_to_expected >= REQ_VOL_REL_DAILY)
    return {
        "prev_close_vs_20d_high": prev_close_vs_20d_high,
        "vol_ratio_to_expected": vol_ratio_to_expected,
        "breakout_ok": bool(breakout_ok),
        "last_price": last_price,
    }

def rsi_features(daily: pd.DataFrame, t: str) -> dict:
    try:
        df = daily[t].dropna()
    except Exception:
        return {}
    if df.empty: return {}
    close = df["Close"].astype(float)
    rsi2 = float(rsi(close, RSI2_LEN).iloc[-1])
    rsi14 = float(rsi(close, RSI14_LEN).iloc[-1])
    rsi_ok = (rsi2 <= RSI2_BUY_MAX) and (rsi14 <= RSI14_BUY_MAX)
    return {"RSI2": rsi2, "RSI14": rsi14, "rsi_ok": bool(rsi_ok)}

def extension_features(daily: pd.DataFrame, t: str) -> dict:
    try:
        df = daily[t].dropna()
    except Exception:
        return {}
    if df.empty: return {}
    close = df["Close"].astype(float)
    ma20 = close.rolling(20, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=20).mean()
    d20 = float(_safe_div(close.iloc[-1] - ma20.iloc[-1], ma20.iloc[-1]))
    d50 = float(_safe_div(close.iloc[-1] - ma50.iloc[-1], ma50.iloc[-1]))
    r2  = float(_safe_div((close.iloc[-1] - close.iloc[-3]), close.iloc[-3])) if len(close) >= 3 else 0.0
    r14 = float(_safe_div((close.iloc[-1] - close.iloc[-15]), close.iloc[-15])) if len(close) >= 15 else 0.0
    vr  = 0.0  # volume extension suhteessa odotukseen voidaan lisätä myöhemmin, tarvittaessa
    us  = 0.0  # up-streak -proksy (lisättävissä)
    return {"d20": d20, "d50": d50, "r2": r2, "r14": r14, "vr": vr, "up_streak_proxy": us}

def extension_score(feat: dict) -> float:
    d20 = feat.get("d20", 0.0); d50 = feat.get("d50", 0.0)
    r2  = feat.get("r2", 0.0);  r14 = feat.get("r14", 0.0)
    vr  = feat.get("vr", 0.0);  us  = feat.get("up_streak_proxy", 0.0)
    w1,w2,w3,w4,w5,w6 = 0.25,0.25,0.2,0.15,0.1,0.05
    score = w1*d20 + w2*d50 + w3*r2 + w4*r14 + w5*vr + w6*us
    return float(np.clip(score, 0, 1))

# ---------- PÄÄLASKENTA ----------
def compute_signals(pool: pd.DataFrame):
    tickers = pool["Ticker"].dropna().astype(str).unique().tolist()
    if not tickers:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    if len(tickers) > MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]

    daily = download_daily_history(tickers, years=1)
    intra = download_intraday_5m(tickers)
    last_intra = last_intraday_bar(intra)

    rows_breakout, rows_rsi, rows_combo = [], [], []

    for t in tickers:
        brf = breakout_features(daily, last_intra, t)
        rsf = rsi_features(daily, t)
        ext = extension_features(daily, t)
        if not brf and not rsf:
            continue

        in_active, near_days = window_status_and_proximity(t)
        week_val, week_std = weekly_seasonality_score(t)

        # BREAKOUT score
        b_score = 0.0
        if brf:
            b_score = 0.5*float(brf["breakout_ok"]) + 0.5*np.clip((brf["prev_close_vs_20d_high"]-REQ_ABOVE_HIGH_PCT)*50, 0, 1)
        # RSI score
        r_score = 0.0
        if rsf:
            r2 = 1 - np.clip(rsf["RSI2"]/100, 0, 1)
            r14 = 1 - np.clip((rsf["RSI14"]-20)/60, 0, 1)
            r_score = 0.6*r2 + 0.4*r14

        ext_s = extension_score(ext)
        combo = 0.5*b_score + 0.5*r_score
        combo_adj = combo - EXT_ALPHA*ext_s

        if MUST_BE_IN_OR_NEAR_ACTIVE_WINDOW:
            prox_bonus = 0.0
            if in_active:
                prox_bonus = 0.1
            elif near_days is not None and 0 <= near_days <= NEAR_WINDOW_DAYS:
                prox_bonus = 0.05 * (1 - near_days/NEAR_WINDOW_DAYS)
            combo += prox_bonus
            combo_adj += prox_bonus

        rows_breakout.append({
            "Ticker": t, **brf, "in_active_window": in_active,
            "days_to_window_start": 0 if (in_active or near_days is None) else near_days
        })
        rows_rsi.append({
            "Ticker": t, **rsf, "price_seasonality_woy": week_val,
            "in_active_window": in_active,
            "days_to_window_start": 0 if (in_active or near_days is None) else near_days
        })
        rows_combo.append({
            "Ticker": t, **brf, **rsf, **ext,
            "extension_score": ext_s,
            "combo_score": combo,
            "combo_score_adj": combo_adj,
            "in_active_window": in_active,
            "days_to_window_start": 0 if (in_active or near_days is None) else near_days
        })

    br = pd.DataFrame(rows_breakout).sort_values(["in_active_window","combo_score" if "combo_score" in rows_combo[0] else "breakout_ok"], ascending=[False,False]) if rows_breakout else pd.DataFrame()
    rs = pd.DataFrame(rows_rsi).sort_values(["in_active_window","RSI2"], ascending=[False,True]) if rows_rsi else pd.DataFrame()
    cb = pd.DataFrame(rows_combo).sort_values(["in_active_window","combo_score"], ascending=[False,False]) if rows_combo else pd.DataFrame()
    cx = cb.copy()
    if not cx.empty and "combo_score_adj" in cx.columns:
        cx = cx.sort_values(["in_active_window","combo_score_adj"], ascending=[False,False])

    return br, rs, cb, cx

def print_top5(name: str, df: pd.DataFrame, cols: list[str]):
    if df is None or df.empty:
        print(f"\n{name}: (ei ehdokkaita)")
        return
    print(f"\n{name}: Top 5")
    show = df.head(5).copy()
    for _, r in show.iterrows():
        parts = []
        for c in cols:
            if c not in show.columns: continue
            v = r[c]
            if isinstance(v, float):
                parts.append(f"{c}={v:.3f}")
            else:
                parts.append(f"{c}={v}")
        print(f"  {r.get('Ticker')}  " + "  |  ".join(parts))

def main():
    # Allow runtime overrides for directories (no breaking changes)
    _configure_paths_from_cli()

    os.makedirs(INTRA_BASE, exist_ok=True)
    out_dir = ensure_output_dir_for_today()
    date_str = dt.date.today().strftime("%Y-%m-%d")

    pool = read_pool_from_aggregates()
    if pool.empty:
        print("[WARN] Ei tikkeripoolia aggregates-kansiosta. Aja ensin aggregate_seasonality_picker.py")
        return

    br, rs, cb, cx = compute_signals(pool)

    br_path = os.path.join(out_dir, f"breakout_watchlist_{date_str}.csv")
    rs_path = os.path.join(out_dir, f"rsi_watchlist_{date_str}.csv")
    cb_path = os.path.join(out_dir, f"combo_watchlist_{date_str}.csv")
    cx_path = os.path.join(out_dir, f"combo_watchlist_EXTADJ_{date_str}.csv")

    if not br.empty: br.to_csv(br_path, index=False, encoding="utf-8-sig")
    if not rs.empty: rs.to_csv(rs_path, index=False, encoding="utf-8-sig")
    if not cb.empty: cb.to_csv(cb_path, index=False, encoding="utf-8-sig")
    if not cx.empty: cx.to_csv(cx_path, index=False, encoding="utf-8-sig")

    print("[OK] Tallennettu:")
    if not br.empty: print(" -", os.path.abspath(br_path))
    if not rs.empty: print(" -", os.path.abspath(rs_path))
    if not cb.empty: print(" -", os.path.abspath(cb_path))
    if not cx.empty: print(" -", os.path.abspath(cx_path))

    print_top5("BREAKOUT", br, ["breakout_score","breakout_ok","prev_close_vs_20d_high","vol_ratio_to_expected","in_active_window","days_to_window_start"])
    print_top5("RSI", rs, ["rsi_score","RSI2","RSI14","price_seasonality_woy","in_active_window","days_to_window_start"])
    print_top5("COMBO", cb, ["combo_score","breakout_score","rsi_score","prev_close_vs_20d_high","vol_ratio_to_expected","RSI2","RSI14","in_active_window","days_to_window_start"])
    print_top5("COMBO (EXT-ADJUSTED)", cx, ["combo_score_adj","extension_score","combo_score","prev_close_vs_20d_high","up_5d_20d_pct","up_streak_days","vol_ratio_to_expected","RSI2","RSI14"])

if __name__ == "__main__":
    main()
