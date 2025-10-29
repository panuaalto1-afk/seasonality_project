#!/usr/bin/env python
# -*- coding: utf-8 -*-
# optio_seasonality_price_enricher.py (v1.3)

import argparse, sys, io
from pathlib import Path
import datetime as dt
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None  # sallitaan offline-ajo

# ---------- IO ----------

def sniff_delimiter(sample_path: Path) -> str:
    try:
        with open(sample_path, 'r', encoding='utf-8', errors='ignore') as f:
            head = f.read(4096)
        return ';' if head.count(';') > head.count(',') else ','
    except Exception:
        return ','

def read_csv_robust(p: Path) -> pd.DataFrame:
    if not p or not p.exists(): return pd.DataFrame()
    sep = sniff_delimiter(p)
    try:    return pd.read_csv(p, sep=sep, engine='python')
    except: return pd.read_csv(p, sep=sep, engine='python', encoding='latin-1')

def write_csv_robust(df: pd.DataFrame, p: Path, sep: str = ','):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, sep=sep, lineterminator='\n')

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def latest_dated_folder(base: Path) -> Path | None:
    if not base.exists(): return None
    cands = []
    for d in base.iterdir():
        if d.is_dir():
            try: dt.datetime.strptime(d.name, "%Y-%m-%d"); cands.append(d)
            except: pass
    return sorted(cands, key=lambda x: x.name)[-1] if cands else None

def ticker_col_guess(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() in ("ticker","symbol","sy","tkr"): return c
    return df.columns[0] if (not df.empty and df.dtypes.iloc[0] == object) else None

# ---------- Price cleaning ----------

def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pakkoparsii Date + hinnat ja siivoaa tikkeririvin/duplikaatit."""
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    # yhtenäistä nimet
    lower = {c.lower(): c for c in d.columns}
    def pick(*names):
        for n in names:
            if n in lower: return lower[n]
        return None
    ren = {}
    for want, cands in {
        "Date": ("date","time","timestamp"),
        "Open": ("open",),
        "High": ("high",),
        "Low":  ("low",),
        "Close":("close","adj close","adj_close","adjclose"),
        "Volume":("volume",)
    }.items():
        cur = pick(*cands)
        if cur and cur != want:
            ren[cur] = want
    if ren: d = d.rename(columns=ren)

    if "Date" not in d.columns or "Close" not in d.columns:
        return pd.DataFrame()

    # pakkoparsitaan ja siivotaan
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ("Open","High","Low","Close","Volume"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Date","Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return d.reset_index(drop=True)

# ---------- Indicators / features ----------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(H: pd.Series, L: pd.Series, C: pd.Series, length: int = 14) -> pd.Series:
    prev = C.shift(1)
    tr = pd.concat([(H-L).abs(), (H-prev).abs(), (L-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def compute_price_features(df: pd.DataFrame) -> dict:
    d = clean_price_df(df)
    if d.empty: return {}
    d['date'] = d['Date'].dt.date
    d['ret5']  = d['Close'].pct_change(5,  fill_method=None)
    d['ret20'] = d['Close'].pct_change(20, fill_method=None)
    d['ret63'] = d['Close'].pct_change(63, fill_method=None)
    d['ma20']  = d['Close'].rolling(20,  min_periods=1).mean()
    d['ma50']  = d['Close'].rolling(50,  min_periods=1).mean()
    d['ma200'] = d['Close'].rolling(200, min_periods=1).mean()
    d['hh20']  = d['High'].rolling(20,  min_periods=1).max() if 'High' in d.columns else d['Close'].rolling(20,1).max()
    d['ll20']  = d['Low' ].rolling(20,   min_periods=1).min() if 'Low'  in d.columns else d['Close'].rolling(20,1).min()
    d['atr20'] = atr(d.get('High', d['Close']), d.get('Low', d['Close']), d['Close'], 20)
    d['rsi2']  = rsi(d['Close'], 2)
    d['rsi14'] = rsi(d['Close'], 14)
    last = d.iloc[-1]
    def f(x): 
        try: return float(x) if pd.notna(x) else np.nan
        except: return np.nan
    return {
        'last_date': str(last['date']),
        'close': f(last['Close']),
        'ret5_%': f(last['ret5']*100), 'ret20_%': f(last['ret20']*100), 'ret63_%': f(last['ret63']*100),
        'dist_ma20_%': f((last['Close']/last['ma20']-1)*100),
        'dist_ma50_%': f((last['Close']/last['ma50']-1)*100),
        'dist_ma200_%': f((last['Close']/last['ma200']-1)*100),
        'dist_20d_high_%': f((last['Close']/last['hh20']-1)*100),
        'dist_20d_low_%' : f((last['Close']/last['ll20']-1)*100),
        'atr20_%': f((last['atr20']/last['Close'])*100),
        'rsi2': f(last['rsi2']), 'rsi14': f(last['rsi14']),
    }

def beta_to_spy(px: pd.DataFrame, spy: pd.DataFrame, lookback: int = 252) -> float | None:
    try:
        a = clean_price_df(px); b = clean_price_df(spy)
        if a.empty or b.empty: return None
        a = a[['Date','Close']].rename(columns={'Close':'c'})
        b = b[['Date','Close']].rename(columns={'Close':'c'})
        m = pd.merge(a,b,on='Date',how='inner',suffixes=('','_spy')).tail(lookback)
        if len(m) < 60: return None
        r  = m['c'].pct_change(fill_method=None).dropna()
        rs = m['c_spy'].pct_change(fill_method=None).dropna()
        if len(r) != len(rs): 
            n = min(len(r),len(rs)); r = r.tail(n); rs = rs.tail(n)
        var = float(np.var(rs))
        return None if var == 0 else float(np.cov(r,rs)[0,1]/var)
    except Exception:
        return None

# ---------- Crosslead ----------

def read_crosslead_summary(run_base: Path) -> pd.DataFrame:
    if not run_base or not run_base.exists(): return pd.DataFrame()
    def _read(name: str) -> pd.DataFrame:
        for ext in ('.csv','.CSV','.txt'):
            p = run_base / f"{name}{ext}"
            if p.exists(): return read_csv_robust(p)
        return pd.DataFrame()
    leaders = _read('leaders_rank')
    followers = _read('followers_map')
    cont = _read('continuation_stats_A')
    stab = _read('window_stability')
    out = {}
    if not leaders.empty:
        lc = {str(c).lower():c for c in leaders.columns}
        lcol = lc.get('leader') or lc.get('symbol') or lc.get('ticker') or leaders.columns[0]
        rcol = lc.get('rank') or leaders.columns[1]
        scol = lc.get('strength')
        for _,r in leaders.iterrows():
            t = str(r[lcol]).strip().upper()
            out.setdefault(t,{})['leader_rank'] = float(pd.to_numeric(r.get(rcol,np.nan), errors='coerce'))
            if scol: out[t]['leader_strength'] = float(pd.to_numeric(r.get(scol,np.nan), errors='coerce'))
    if not followers.empty:
        lc = {str(c).lower():c for c in followers.columns}
        lcol = lc.get('leader') or followers.columns[0]
        fcol = lc.get('follower') or lc.get('symbol') or lc.get('ticker') or followers.columns[1]
        lagc = lc.get('lag') or lc.get('lead_lag')
        corr = lc.get('corr') or lc.get('strength') or lc.get('score')
        for _,r in followers.iterrows():
            f = str(r[fcol]).strip().upper(); L = str(r[lcol]).strip().upper()
            d = out.setdefault(f,{})
            cur = abs(d.get('lead_strength', -1e9))
            cand = abs(float(pd.to_numeric(r.get(corr,0), errors='coerce'))) if corr else 0.0
            if cand >= cur:
                d['best_leader'] = L
                d['lead_strength'] = float(pd.to_numeric(r.get(corr,np.nan), errors='coerce')) if corr else np.nan
                d['lead_lag_days'] = int(pd.to_numeric(r.get(lagc,np.nan), errors='coerce')) if lagc else np.nan
    if not cont.empty:
        lc = {str(c).lower():c for c in cont.columns}
        tcol = lc.get('ticker') or lc.get('symbol') or cont.columns[0]
        pcol = lc.get('p_cont') or lc.get('continuation_prob') or lc.get('probability')
        if pcol:
            for _,r in cont.iterrows():
                t = str(r[tcol]).strip().upper()
                out.setdefault(t,{})['continuation_prob'] = float(pd.to_numeric(r.get(pcol,np.nan), errors='coerce'))
    if not stab.empty:
        lc = {str(c).lower():c for c in stab.columns}
        tcol = lc.get('ticker') or lc.get('symbol') or stab.columns[0]
        scol = lc.get('stability') or lc.get('stability_score') or lc.get('window_stability')
        if scol:
            for _,r in stab.iterrows():
                t = str(r[tcol]).strip().upper()
                out.setdefault(t,{})['window_stability'] = float(pd.to_numeric(r.get(scol,np.nan), errors='coerce'))
    return pd.DataFrame.from_dict(out, orient='index').reset_index().rename(columns={'index':'ticker'}) if out else pd.DataFrame()

# ---------- Optio-lähteet ----------

def read_optio_signals(optio_dir: Path) -> pd.DataFrame:
    files = list(optio_dir.glob("*.csv")) + list(optio_dir.glob("*.CSV"))
    if not files: return pd.DataFrame()
    def parse_one(p: Path) -> pd.DataFrame:
        df = read_csv_robust(p)
        if df.empty: return df
        tcol = ticker_col_guess(df)
        if tcol is None: return pd.DataFrame()
        df['ticker'] = df[tcol].astype(str).str.upper().str.strip()
        rank_cols = [c for c in df.columns if str(c).lower() in ('rank','score','signal_score','position')]
        df['rank_or_score'] = pd.to_numeric(df[rank_cols[0]], errors='coerce') if rank_cols else np.nan
        base = p.name.lower()
        if 'long' in base:  df = df[['ticker','rank_or_score']].rename(columns={'rank_or_score':'long_rank'}).assign(in_long_list=1)
        elif 'short' in base: df = df[['ticker','rank_or_score']].rename(columns={'rank_or_score':'short_rank'}).assign(in_short_list=1)
        elif 'exit' in base: df = df[['ticker','rank_or_score']].rename(columns={'rank_or_score':'exit_rank'}).assign(in_exit_list=1)
        else: df = df[['ticker','rank_or_score']].rename(columns={'rank_or_score':f'{p.stem}_score'}).assign(**{f'in_{p.stem}':1})
        return df
    parts = [parse_one(p) for p in files]
    parts = [p for p in parts if not p.empty]
    if not parts: return pd.DataFrame()
    m = parts[0]
    for p in parts[1:]:
        m = m.merge(p, on='ticker', how='outer')
    return m.fillna(0)

# ---------- Price cache ----------

def price_file_candidates(price_dir: Path, t: str):
    return [price_dir / f"{t}.csv", price_dir / f"{t.upper()}.csv", price_dir / f"{t.lower()}.csv"]

def load_price_cache(price_dir: Path, ticker: str):
    for cand in price_file_candidates(price_dir, ticker):
        if cand.exists():
            sep = sniff_delimiter(cand); df = read_csv_robust(cand)
            return df, cand, sep
    return None, None, ','

def maybe_refresh_online(ticker: str, last_date: dt.date, do_refresh: bool) -> pd.DataFrame | None:
    if not do_refresh or yf is None: return None
    try:
        start = (last_date + dt.timedelta(days=1)).isoformat()
        y = yf.download(ticker, start=start, progress=False, interval='1d', auto_adjust=False)
        if y is None or y.empty: return None
        return y.reset_index()[['Date','Open','High','Low','Close','Adj Close','Volume']]
    except Exception:
        return None

def append_and_save_if_needed(src: pd.DataFrame, add: pd.DataFrame, orig_path: Path, sep: str, save_cache: bool) -> pd.DataFrame:
    if add is None or add.empty: return src
    src_cols = [str(c) for c in src.columns]
    add2 = add.copy()
    name_map = {}
    for c in add2.columns:
        cc = str(c)
        for s in src_cols:
            ss = str(s)
            if cc.lower().replace(' ','') == ss.lower().replace(' ',''):
                name_map[c] = s; break
    add2 = add2.rename(columns=name_map)
    dcol = [s for s in src_cols if s.lower() == 'date']
    if dcol: add2[dcol[0]] = pd.to_datetime(add2[dcol[0]]).dt.strftime('%Y-%m-%d')
    merged = pd.concat([src, add2], ignore_index=True, sort=False)
    if dcol: merged = merged.drop_duplicates(subset=[dcol[0]], keep='last').sort_values(dcol[0])
    if save_cache and orig_path:
        target = merged[src.columns] if set(src.columns).issubset(merged.columns) else merged
        write_csv_robust(target, orig_path, sep=sep)
    return merged

# ---------- Ranking ----------

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def combo_score_series(df: pd.DataFrame, side: str) -> pd.Series:
    X = df.copy()
    if side == 'long':
        opt = -zscore(X.get('long_rank', pd.Series(np.nan, index=X.index)).fillna(X.get('long_rank', pd.Series([0])).max()))
        r20 = zscore(X['ret20_%'].fillna(0))
        ext = -zscore(X['dist_20d_high_%'].abs().fillna(0))
        ma  = -zscore(X['dist_ma20_%'].abs().fillna(0))
        rsi_p = -zscore(X['rsi2'].fillna(50))
    else:
        opt = -zscore(X.get('short_rank', pd.Series(np.nan, index=X.index)).fillna(X.get('short_rank', pd.Series([0])).max()))
        r20 = -zscore(X['ret20_%'].fillna(0))
        ext = -zscore(X['dist_20d_low_%'].abs().fillna(0))
        ma  = -zscore(X['dist_ma20_%'].abs().fillna(0))
        rsi_p =  zscore(X['rsi2'].fillna(50))
    lead = zscore(X.get('lead_strength', pd.Series(0,index=X.index)).fillna(0))
    cont = zscore(X.get('continuation_prob', pd.Series(0,index=X.index)).fillna(0))
    stab = zscore(X.get('window_stability', pd.Series(0,index=X.index)).fillna(0))
    beta = -zscore(X.get('beta_spy', pd.Series(1.0,index=X.index)).fillna(1.0).abs())
    w = dict(opt=0.40,r20=0.20,ext=0.10,ma=0.05,rsi=0.05,lead=0.07,cont=0.07,stab=0.04,beta=0.02)
    return (w['opt']*opt + w['r20']*r20 + w['ext']*ext + w['ma']*ma + w['rsi']*rsi_p +
            w['lead']*lead + w['cont']*cont + w['stab']*stab + w['beta']*beta)

def build_combo_rank(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df.empty: return df
    X = df.copy()
    X[f'combo_score_{side}'] = combo_score_series(X, side)
    return X.sort_values(f'combo_score_{side}', ascending=False)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=str)
    ap.add_argument('--reports-dir', default='', type=str, help='oletus: ROOT\\seasonality_reports')
    ap.add_argument('--optio-date', default='latest', type=str)
    ap.add_argument('--refresh-online', default='false', type=str)
    ap.add_argument('--save-cache', default='false', type=str)
    ap.add_argument('--price-dir', default='', type=str)
    ap.add_argument('--force-sides', default='false', type=str, help='true => tee long/short myös ilman optio-listoja')
    ap.add_argument('--top-n', default='60', type=str, help='force-sides: listan maksimipituus')
    ap.add_argument('--round', default='3', type=str, help='pyöristyksen desimaalit (numeerisille)')
    args = ap.parse_args()

    root = Path(args.root)
    reports_dir = Path(args.reports_dir) if args.reports_dir else (root / 'seasonality_reports')

    # Oletus price_dir jos ei annettu: uusin runs\...\price_cache reports-dirissä
    if args.price_dir:
        price_dir = Path(args.price_dir)
    else:
        runs = reports_dir / 'runs'
        latest_run = latest_dated_folder(runs) if runs.exists() else None
        price_dir = (latest_run / 'price_cache') if latest_run else (root / 'price_cache')

    optio_base = reports_dir / 'aggregates' / 'optio_signals'
    optio_dir  = latest_dated_folder(optio_base) if args.optio_date == 'latest' else (optio_base / args.optio_date)
    if not optio_dir or not optio_dir.exists():
        print(f"[ERROR] Ei löydy optio-signaaleja: {optio_base}\\YYYY-MM-DD", file=sys.stderr); sys.exit(2)

    runs_dir = reports_dir / 'runs'
    crosslead_dir = None
    if runs_dir.exists():
        cands = []
        for r in runs_dir.iterdir():
            sub = r / 'crosslead_price_only_v1_2'
            if sub.exists(): cands.append(sub)
        if cands: crosslead_dir = sorted(cands, key=lambda p: p.parent.name)[-1]

    out_dir = reports_dir / 'aggregates' / 'optio_signals_enriched' / optio_dir.name
    ensure_dir(out_dir)

    log = io.StringIO()
    def logprint(*a): s = " ".join([str(x) for x in a]); print(s); print(s, file=log)

    logprint(f"[INFO] root={root}")
    logprint(f"[INFO] reports_dir={reports_dir}")
    logprint(f"[INFO] optio_dir={optio_dir}")
    logprint(f"[INFO] price_dir={price_dir}")
    logprint(f"[INFO] crosslead_dir={crosslead_dir}")

    optio = read_optio_signals(optio_dir)
    if optio.empty:
        logprint("[ERROR] Optiosignaali-CSV:t puuttuvat."); Path(out_dir/'log.txt').write_text(log.getvalue(), encoding='utf-8'); sys.exit(1)
    tickers = sorted(optio['ticker'].unique().tolist()); logprint(f"[INFO] Tickereitä: {len(tickers)}")

    # SPY beta
    spy_df, _, _ = load_price_cache(price_dir, 'SPY')

    refresh    = args.refresh_online.lower() in ('1','true','yes','y')
    save_cache = args.save_cache.lower() in ('1','true','yes','y')
    force_sides= args.force_sides.lower() in ('1','true','yes','y')
    top_n      = int(args.top_n)
    rnd        = int(args.round)

    # Hinta featuret
    rows = []
    for t in tickers:
        src, path, sep = load_price_cache(price_dir, t)
        if src is None or src.empty:
            logprint(f"[WARN] Ei price_cache-tiedostoa: {t}")
            continue
        src = clean_price_df(src)
        if src.empty:
            logprint(f"[WARN] Hintahistoria tyhjeni siivouksessa: {t}")
            continue
        try:
            last_date = pd.to_datetime(src["Date"]).dt.date.max()
        except Exception:
            last_date = None
        add = maybe_refresh_online(t, last_date, refresh) if last_date else None
        if add is not None and not add.empty:
            src = append_and_save_if_needed(src, add, path, sep, save_cache)
            src = clean_price_df(src)
            logprint(f"[UPDATE] {t}: lisätty {len(add)} riviä. (save_cache={save_cache})")
        feats = compute_price_features(src)
        if not feats:
            logprint(f"[WARN] Ei featureita: {t}")
            continue
        feats.update({'ticker': t, 'beta_spy': beta_to_spy(src, spy_df, 252) if spy_df is not None else None})
        rows.append(feats)

    price_feats = pd.DataFrame(rows)
    if price_feats.empty:
        logprint("[ERROR] Hintasuureita ei muodostunut."); Path(out_dir/'log.txt').write_text(log.getvalue(), encoding='utf-8'); sys.exit(1)

    # Crosslead
    crosslead = read_crosslead_summary(crosslead_dir) if crosslead_dir else pd.DataFrame()
    if not crosslead.empty: logprint(f"[INFO] Crosslead-rivejä: {len(crosslead)}")
    else: logprint("[INFO] Crosslead-yhteenvetoa ei löytynyt; jatketaan ilman.")

    # Merge + siivoukset
    base = optio.merge(price_feats, on='ticker', how='left')
    if not crosslead.empty: base = base.merge(crosslead, on='ticker', how='left')

    def coalesce_duplicate_columns(df: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
        out = df.copy()
        for base in bases:
            cand = [c for c in out.columns if c == base or c.startswith(base + "_")]
            if len(cand) <= 1: continue
            stacked = pd.concat([out[c] for c in cand], axis=1)
            out[base] = stacked.bfill(axis=1).iloc[:, 0]
            for c in cand:
                if c != base and c in out.columns:
                    out.drop(columns=c, inplace=True, errors="ignore")
        return out

    base = coalesce_duplicate_columns(base, 
            bases=['exit_rank','in_exit_list','long_rank','in_long_list','short_rank','in_short_list'])

    if 'last_date' in base.columns:
        base['last_date'] = base['last_date'].astype(str).replace('NaT','')

    num_cols = ['close','ret5_%','ret20_%','ret63_%','dist_ma20_%','dist_ma50_%','dist_ma200_%',
                'dist_20d_high_%','dist_20d_low_%','atr20_%','rsi2','rsi14','beta_spy',
                'leader_rank','lead_strength','lead_lag_days','continuation_prob','window_stability',
                'long_rank','short_rank','exit_rank']
    for c in num_cols:
        if c in base.columns: base[c] = pd.to_numeric(base[c], errors='coerce')

    # combo_score myös all-tauluun
    base['combo_score_long']  = combo_score_series(base, 'long')
    base['combo_score_short'] = combo_score_series(base, 'short')

    # pyöristys
    for c in [col for col in base.columns if base[col].dtype.kind in 'fc']:
        base[c] = base[c].round(rnd)

    # Maskit
    def col_or_default(df, name, default=0): return df[name] if name in df.columns else pd.Series(default, index=df.index)
    long_mask  = (col_or_default(base, 'in_long_list', 0)  == 1) | base.get('long_rank',  pd.Series(np.nan,index=base.index)).notna()
    short_mask = (col_or_default(base, 'in_short_list', 0) == 1) | base.get('short_rank', pd.Series(np.nan,index=base.index)).notna()

    out_all  = base.copy()
    out_long = build_combo_rank(base.loc[long_mask].copy(),  'long')  if long_mask.any()  else pd.DataFrame()
    out_short= build_combo_rank(base.loc[short_mask].copy(), 'short') if short_mask.any() else pd.DataFrame()

    if out_long.empty and force_sides:
        logprint(f"[INFO] Long-listaa ei lähteessä -> muodostetaan price+crosslead -lista (top {top_n}).")
        out_long = build_combo_rank(base.copy(), 'long').head(top_n)
    if out_short.empty and force_sides:
        logprint(f"[INFO] Short-listaa ei lähteessä -> muodostetaan price+crosslead -lista (top {top_n}).")
        out_short = build_combo_rank(base.copy(), 'short').head(top_n)

    write_csv_robust(out_all,  out_dir/'optio_price_enriched_all.csv')
    if not out_long.empty:
        write_csv_robust(out_long,  out_dir/'optio_price_enriched_long.csv');  logprint("[OK] optio_price_enriched_long.csv kirjoitettu.")
    else:
        logprint("[INFO] Long-listaa ei muodostunut (maski tyhjä).")
    if not out_short.empty:
        write_csv_robust(out_short, out_dir/'optio_price_enriched_short.csv'); logprint("[OK] optio_price_enriched_short.csv kirjoitettu.")
    else:
        logprint("[INFO] Short-listaa ei muodostunut (maski tyhjä).")

    Path(out_dir/'log.txt').write_text(log.getvalue(), encoding='utf-8')
    logprint("[DONE] Enrichment valmis:", out_dir)

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
