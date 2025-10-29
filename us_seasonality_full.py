# us_seasonality_full.py (v4 – greedy Top-K non-overlapping windows, up & down lists)
# Robust scraping (Wikipedia + fallback), prices via yfinance, flexible funda filter,
# DOY seasonality with smoothing, DP + greedy Top-K refinement, SPX/NASDAQ/Russell lists,
# UTF-8-SIG CSVs, global best windows.

import os, time, math, warnings, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore")
pd.options.display.width = 180

# ========= CONFIG =========
START = "2014-01-01"
END   = None
LOOKBACK_YEARS_FOR_SEAS = 10

MIN_PRICE_USD   = 5.0
MIN_MCAP_USD    = 50_000_000
MIN_GROWTH_TTM  = 0.10  # require at least revenue TTM growth; NI may be NaN

OUTDIR = "seasonality_reports"
CSV_KW = dict(index=False, encoding="utf-8-sig")

# DOY curve smoothing & base DP
SMOOTH_DOY_WINDOW = 7
MAX_SEGMENTS_DP   = 8
PENALTY_MULT      = 2.0

# Greedy Top-K windows (final non-overlapping selection)
TOPK_PER_TICKER   = 6
MIN_SEG_LEN       = 15       # days (inclusive) – voit säätää esim. 15–25
MAX_SEG_LEN       = 120      # days (inclusive) – jos haluat lyhyempiä, laske tätä
MIN_STRENGTH_UP   = 0.0007   # ensikynnys (up)
MIN_STRENGTH_DOWN = 0.0007   # ensikynnys (down)
ADAPT_ROUNDS      = 3        # jos ei löydy, laske kynnystä 30%/kierros

# Optional Russell 2000 constituents CSV (column 'Ticker'); otherwise only ETF proxy is used
RUSSELL2000_CSV = None

# ========= Helpers: robust HTML → table =========
def _fetch_html(url, timeout=30, retries=3):
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/125.0.0.0 Safari/537.36")}
    last = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise last

def _read_html_table(url, table_idx=None, must_have_cols=None, fallback_csv=None):
    try:
        html = _fetch_html(url)
        tables = pd.read_html(html)
        cand = list(range(len(tables)))
        if table_idx is not None and 0 <= table_idx < len(tables):
            cand = [table_idx] + [i for i in cand if i != table_idx]
        if must_have_cols:
            scored = []
            for i in cand:
                cols = [str(c).lower() for c in tables[i].columns]
                score = sum(any(key.lower() in col for col in cols) for key in must_have_cols)
                scored.append((score, i))
            scored.sort(reverse=True)
            return tables[scored[0][1]]
        return tables[cand[0]]
    except Exception:
        if fallback_csv:
            return pd.read_csv(fallback_csv)
        raise

def _normalize_tickers(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c].astype(str).str.strip().str.replace(".", "-", regex=False)
    return df.iloc[:,0].astype(str).str.strip().str.replace(".", "-", regex=False)

def _normalize_names(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)

def _normalize_sector(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    for c in df.columns:
        lc = str(c).lower()
        if "sector" in lc or "industry" in lc:
            return df[c]
    return pd.Series(np.nan, index=df.index)

# ========= Universe (SPX, NDX, DJIA [+ optional RUT]) =========
def get_constituents():
    spx = _read_html_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                           table_idx=0, must_have_cols=["Symbol","Security","GICS"],
                           fallback_csv="https://datahub.io/core/s-and-p-500-companies/r/constituents.csv").copy()
    spx["Ticker"]  = _normalize_tickers(spx, ["Symbol","Ticker"])
    spx["Company"] = _normalize_names(spx, ["Security","Company","Name"])
    spx["Sector"]  = _normalize_sector(spx, ["GICS Sector","Sector"])
    spx = spx[["Ticker","Company","Sector"]]; spx["Index"] = "SPX"

    ndx = _read_html_table("https://en.wikipedia.org/wiki/Nasdaq-100",
                           table_idx=3, must_have_cols=["Ticker","Symbol","Company","Name"],
                           fallback_csv="https://datahub.io/core/nasdaq-100-companies/r/constituents.csv").copy()
    ndx["Ticker"]  = _normalize_tickers(ndx, ["Ticker","Symbol"])
    ndx["Company"] = _normalize_names(ndx, ["Company","Name"])
    ndx["Sector"]  = _normalize_sector(ndx, ["GICS Sector","Sector"])
    ndx = ndx[["Ticker","Company","Sector"]]; ndx["Index"] = "NDX"

    dj = _read_html_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                          table_idx=None, must_have_cols=["Symbol","Ticker","Company","Name"],
                          fallback_csv="https://datahub.io/core/dow-jones/r/constituents.csv").copy()
    dj["Ticker"]  = _normalize_tickers(dj, ["Symbol","Ticker"])
    dj["Company"] = _normalize_names(dj, ["Company","Name"])
    dj["Sector"]  = _normalize_sector(dj, ["Industry","Sector"])
    dj = dj[["Ticker","Company","Sector"]]; dj["Index"] = "DJIA"

    frames = [spx, ndx, dj]
    if RUSSELL2000_CSV and os.path.exists(RUSSELL2000_CSV):
        r2k = pd.read_csv(RUSSELL2000_CSV)
        r2k["Ticker"] = r2k["Ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
        r2k["Company"] = np.nan; r2k["Sector"] = np.nan; r2k["Index"]="RUT"
        frames.append(r2k[["Ticker","Company","Sector","Index"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker","Index"])

# ========= Prices & returns =========
def fetch_prices(tickers, start=START, end=END):
    tickers = sorted(set([t for t in tickers if isinstance(t,str) and t.strip()]))
    if not tickers: return pd.DataFrame()
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series): px = px.to_frame()
    return px

def daily_returns(close_df):
    return close_df.sort_index().pct_change().dropna(how="all")

def weekly_returns(close_s):
    return close_s.dropna().resample("W-FRI").last().pct_change().dropna()

# ========= Fundamentals =========
def ttm_growth_from_income_stmt(t):
    try:
        tk = yf.Ticker(t)
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        mcap  = info.get("marketCap", np.nan)
        price = info.get("currentPrice", np.nan)

        q = tk.quarterly_financials
        if q is None or q.empty: q = tk.quarterly_income_stmt
        if q is None or q.empty: return (np.nan, np.nan, mcap, price)

        q = q.copy(); q.columns = pd.to_datetime(q.columns); q = q.sort_index(axis=1)

        def _find_row(df, keys):
            lower = [idx.lower() for idx in df.index]
            for key in keys:
                if key in df.index: return df.loc[key]
                for i,nm in enumerate(lower):
                    if key.lower() in nm: return df.iloc[i]
            return pd.Series(dtype=float)

        rev = _find_row(q, ["Total Revenue","TotalRevenue","Revenue"])
        ni  = _find_row(q, ["Net Income","NetIncome"])

        def _ttm_g(s):
            s = s.dropna().astype(float)
            if len(s) < 8: return np.nan
            s = s.sort_index()
            cur, prev = s.iloc[-4:].sum(), s.iloc[-8:-4].sum()
            if not np.isfinite(cur) or not np.isfinite(prev) or prev == 0: return np.nan
            return (cur/prev) - 1.0

        return (_ttm_g(rev), _ttm_g(ni), mcap, price)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)

# ========= DOY seasonality =========
def compute_doy_series(close_s, max_years=LOOKBACK_YEARS_FOR_SEAS):
    s = close_s.dropna().copy()
    if s.empty: return pd.Series(dtype=float), pd.Series(dtype=float), {}
    yrs = sorted(set(s.index.year))
    if len(yrs) > max_years:
        s = s[s.index.year.isin(yrs[-max_years:])]
    ret = s.pct_change().dropna()
    doy_mean = ret.groupby(ret.index.dayofyear).mean().rename("AvgRet_DOY")
    df = ret.to_frame("r"); df["y"]=df.index.year; df["doy"]=df.index.dayofyear
    pivot = df.pivot_table(index="y", columns="doy", values="r", aggfunc="mean")
    hit = (pivot > 0).sum(axis=0) / pivot.shape[0]
    hit = hit.reindex(doy_mean.index).astype(float).rename("Hit_DOY")
    return doy_mean, hit, {"years_used": len(set(df["y"]))}

def smooth_series(s, win=SMOOTH_DOY_WINDOW):
    return s.rolling(win, min_periods=max(1, win//2), center=True).mean() if not s.empty else s

def doy_to_date(doy, year=2024):
    return (dt.date(year,1,1)+dt.timedelta(days=int(doy)-1)).strftime("%b-%d")

# ---- window scoring
def _window_stats(doy_mean, hit, st, en):
    idx = range(st, en+1)
    m = float(doy_mean.reindex(idx).mean())
    h = float(hit.reindex(idx).mean())
    L = en - st + 1
    return m, h, L

def _strength_up(m,h,L):   # for positive windows only
    return (m * max(0.0,h) * math.sqrt(max(1.0,L))) if m>0 else -np.inf

def _strength_down(m,h,L): # for negative windows only
    return (abs(m) * max(0.0,1.0-h) * math.sqrt(max(1.0,L))) if m<0 else -np.inf

def _greedy_topk_windows(doy_mean, hit, topk=TOPK_PER_TICKER,
                         min_len=MIN_SEG_LEN, max_len=MAX_SEG_LEN,
                         min_strength_up=MIN_STRENGTH_UP, min_strength_down=MIN_STRENGTH_DOWN,
                         adapt_rounds=ADAPT_ROUNDS):
    """Return two lists: top up-windows and top down-windows (non-overlapping)."""
    s = smooth_series(doy_mean, SMOOTH_DOY_WINDOW).dropna()
    if s.empty:
        return [], []
    # normalize to 1..N
    N = len(s)
    s.index = range(1, N+1)
    h = hit.reindex(doy_mean.index)
    h.index = range(1, len(h)+1)

    def _greedy_one(kind="up", base_threshold=0.001):
        chosen = []
        mask = np.ones(N, dtype=bool)   # True = free day
        thr = base_threshold
        for _round in range(adapt_rounds):
            while len(chosen) < topk:
                best = None
                # scan windows only on free stretches
                i = 1
                while i <= N:
                    if not mask[i-1]:
                        i += 1
                        continue
                    # find end of free block
                    j = i
                    while j <= N and mask[j-1]:
                        j += 1
                    block_end = j-1
                    # enumerate windows inside the free block
                    a = i
                    while a <= block_end - min_len + 1:
                        bmax = min(block_end, a + max_len - 1)
                        b = a + min_len - 1
                        while b <= bmax:
                            m, hh, L = _window_stats(s, h, a, b)
                            sc = _strength_up(m,hh,L) if kind=="up" else _strength_down(m,hh,L)
                            if sc >= thr:
                                if (best is None) or (sc > best["strength"]):
                                    best = {"start_doy": a, "end_doy": b, "length": L,
                                            "seg_mean": m, "hit_rate": hh, "strength": sc}
                            b += 1
                        a += 1
                    i = block_end + 1

                if best is None:
                    break
                chosen.append(best)
                # mask the chosen window (no overlaps)
                mask[best["start_doy"]-1 : best["end_doy"]] = False

            if chosen:  # got something at this threshold
                break
            # adapt threshold downwards if nothing found
            thr *= 0.7
        # decorate with dates
        for seg in chosen:
            seg["approx_start"] = doy_to_date(seg["start_doy"])
            seg["approx_end"]   = doy_to_date(seg["end_doy"])
        return sorted(chosen, key=lambda r: r["strength"], reverse=True)

    up  = _greedy_one(kind="up",   base_threshold=min_strength_up)
    down= _greedy_one(kind="down", base_threshold=min_strength_down)
    return up, down

# ========= Correlation helpers =========
def corr_topn_vs_benchmark(rets_df, bench_ret, min_overlap=200, topn=15):
    rows=[]
    for t in rets_df.columns:
        sr = rets_df[t].dropna()
        ix = sr.index.intersection(bench_ret.index)
        if len(ix) >= min_overlap:
            corr = np.corrcoef(sr.loc[ix], bench_ret.loc[ix])[0,1]
            rows.append([t, float(corr), int(len(ix))])
    df = pd.DataFrame(rows, columns=["Ticker","corr","overlap_days"]).sort_values("corr", ascending=False)
    return df.head(topn), df.sort_values("corr", ascending=True).head(topn)

# ========= MAIN =========
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) Universe & prices
    cons = get_constituents()
    cons.to_csv(os.path.join(OUTDIR, "constituents_raw.csv"), **CSV_KW)
    print(f"[INFO] Constituents: {len(cons)} rows, tickers={cons['Ticker'].nunique()}")

    tickers = cons["Ticker"].dropna().unique().tolist()
    prices  = fetch_prices(tickers, START, END)
    if prices.empty:
        print("[ERROR] Price download failed."); return
    rets = daily_returns(prices)
    print(f"[INFO] Prices: {prices.shape}")

    # Benchmarks
    spx_close = fetch_prices(["^GSPC"], START, END)["^GSPC"]; spx_ret = daily_returns(spx_close.to_frame())["^GSPC"]
    qqq_close = fetch_prices(["QQQ"], START, END)["QQQ"];    qqq_ret = daily_returns(qqq_close.to_frame())["QQQ"]
    iwm_close = fetch_prices(["IWM"], START, END)["IWM"];    iwm_ret = daily_returns(iwm_close.to_frame())["IWM"]

    # 2) Fundamentals + filter
    funda_rows=[]
    for t in prices.columns:
        rev_g, ni_g, mcap, price = ttm_growth_from_income_stmt(t)
        funda_rows.append([t, rev_g, ni_g, mcap, price]); time.sleep(0.02)
    funda = pd.DataFrame(funda_rows, columns=["Ticker","rev_growth_ttm","ni_growth_ttm","market_cap","last_price"])
    base  = cons.merge(funda, on="Ticker", how="left")
    base.to_csv(os.path.join(OUTDIR, "universe_with_funda.csv"), **CSV_KW)

    filt = (
        (funda["last_price"] >= MIN_PRICE_USD) &
        (funda["market_cap"] >= MIN_MCAP_USD) &
        (funda["rev_growth_ttm"].fillna(-9e9) >= MIN_GROWTH_TTM) &
        ((funda["ni_growth_ttm"].isna()) | (funda["ni_growth_ttm"] >= MIN_GROWTH_TTM))
    )
    universe = funda[filt]["Ticker"].dropna().unique().tolist()
    if not universe:
        print("[WARN] 0 after funda filter → fallback price & mcap only.")
        filt2 = (funda["last_price"] >= MIN_PRICE_USD) & (funda["market_cap"] >= MIN_MCAP_USD)
        universe = funda[filt2]["Ticker"].dropna().unique().tolist()

    cons_f = cons[cons["Ticker"].isin(universe)].copy()
    cons_f.to_csv(os.path.join(OUTDIR, "universe_filtered.csv"), **CSV_KW)
    print(f"[INFO] Filtered universe size: {len(universe)}")

    # 3) SPX / QQQ / IWM correlation lists (whole price universe)
    spx_top, spx_anti = corr_topn_vs_benchmark(rets, spx_ret)
    spx_top.to_csv(os.path.join(OUTDIR, "spx_top15_same_direction.csv"), **CSV_KW)
    spx_anti.to_csv(os.path.join(OUTDIR, "spx_top15_anti_direction.csv"), **CSV_KW)

    qqq_top, qqq_anti = corr_topn_vs_benchmark(rets, qqq_ret)
    qqq_top.to_csv(os.path.join(OUTDIR, "nasdaq_top15_same_direction.csv"), **CSV_KW)
    qqq_anti.to_csv(os.path.join(OUTDIR, "nasdaq_top15_anti_direction.csv"), **CSV_KW)

    iwm_top, iwm_anti = corr_topn_vs_benchmark(rets, iwm_ret)
    iwm_top.to_csv(os.path.join(OUTDIR, "russell_top15_same_direction.csv"), **CSV_KW)
    iwm_anti.to_csv(os.path.join(OUTDIR, "russell_top15_anti_direction.csv"), **CSV_KW)

    print("\n== TOP 15 corr with SPX ==")
    print(spx_top.to_string(index=False))
    print("\n== TOP 15 anti-corr with SPX ==")
    print(spx_anti.to_string(index=False))

    # 4) Seasonality – greedy Top-K per ticker
    best_up_all, best_down_all = [], []
    for t in universe:
        if t not in prices.columns: continue
        s = prices[t].dropna()
        if s.empty: continue

        doy_mean, hit, meta = compute_doy_series(s)
        up_segs, down_segs = _greedy_topk_windows(doy_mean, hit)

        # write per ticker
        up_df = pd.DataFrame(up_segs)
        down_df = pd.DataFrame(down_segs)
        up_df.to_csv(os.path.join(OUTDIR, f"{t}_segments_up.csv"), **CSV_KW)
        down_df.to_csv(os.path.join(OUTDIR, f"{t}_segments_down.csv"), **CSV_KW)

        # weekly table with week_number
        wk = weekly_returns(s)
        wk_df = wk.groupby(wk.index.isocalendar().week).mean().rename("AvgRet_WOY").to_frame()
        wk_df["week_number"] = wk_df.index
        wk_df[["week_number","AvgRet_WOY"]].to_csv(os.path.join(OUTDIR, f"{t}_seasonality_week.csv"), **CSV_KW)

        # collect to global lists
        for seg in up_segs:
            best_up_all.append({"Ticker":t, **seg})
        for seg in down_segs:
            best_down_all.append({"Ticker":t, **seg})

    # 5) Global master lists + top windows
    up_master = pd.DataFrame(best_up_all)
    down_master = pd.DataFrame(best_down_all)
    up_master.sort_values(["strength","seg_mean","length"], ascending=[False,False,False], inplace=True)
    down_master.sort_values(["strength","seg_mean"], ascending=[False, True], inplace=True)  # more negative mean is stronger down

    up_master.to_csv(os.path.join(OUTDIR, "best_up_windows_all.csv"), **CSV_KW)
    down_master.to_csv(os.path.join(OUTDIR, "best_down_windows_all.csv"), **CSV_KW)

    # global “top 6” windows (aggregate by day index on up windows)
    def global_windows_from_segments(all_segs, top_n=6, year_for_label=2024):
        if not all_segs: return []
        N=366; agg=np.zeros(N+1)
        for seg in all_segs:
            st,en = seg["start_doy"], seg["end_doy"]
            m = seg.get("seg_mean", 0.0)
            if not np.isfinite(m): continue
            agg[st:en+1] += m
        wins=[]; i=1
        while i<=N:
            if agg[i]>0:
                j=i; acc=0.0
                while j<=N and agg[j]>0: acc+=agg[j]; j+=1
                wins.append({"start_doy":i,"end_doy":j-1,"score":acc,"length":(j-1)-i+1})
                i=j
            else: i+=1
        wins.sort(key=lambda w: (w["score"]/max(1,w["length"])), reverse=True)
        out=[]
        for w in wins[:top_n]:
            out.append({"start_doy":w["start_doy"],"end_doy":w["end_doy"],"length":w["length"],
                        "avg_strength":w["score"]/max(1,w["length"]),
                        "approx_start":doy_to_date(w["start_doy"], year_for_label),
                        "approx_end":doy_to_date(w["end_doy"], year_for_label)})
        return out

    global6 = global_windows_from_segments(best_up_all, top_n=6)
    gdf = pd.DataFrame(global6)
    gdf.to_csv(os.path.join(OUTDIR, "global_top_seasonality_windows.csv"), **CSV_KW)

    print("\n== Global top seasonality windows (max 6, UP) ==")
    if not gdf.empty: print(gdf.to_string(index=False))
    else: print("No global windows found.")

    print(f"\nValmiit CSV:t: {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()
