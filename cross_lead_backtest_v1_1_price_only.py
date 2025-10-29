# -*- coding: utf-8 -*-
"""
cross_lead_backtest_v1_2_price_only_5y.py  (LOW-MEM, 5y default)

- Price & Volume only (BUD/REU/HVU/GNG + Event Strength, ei indikaattoreita)
- Walk-forward: Train 24 kk -> Test 3 kk, rullaava
- Market-neutralointi: EW-markkina + beta (train-ikkuna)
- Lead–lag A->B: L in {1..5}, H in {1,3,5}
- FDR (BH), stabiilisuus ikkunittain
- Päivän "1 osake/päivä" -valinta (D+1 Open, oletus H=3)
- LOW-MEM: Ei rakenneta NxN-korrelaatiomatriisia; TOP-K lasketaan
  A-kohtaisesti min_periods-tarkastuksella.

Uutta tässä versiossa:
- Oletuksena rajaa datan viimeiseen 5 vuoteen (--years-back 5).
- Vaihtoehtoisesti voi antaa --date-start / --date-end.
- Tarvittaessa voi rajata ikkunamäärää: --max-windows N (esim. 6).

Ulostulot: seasonality_reports\\runs\\{TIMESTAMP}\\crosslead_price_only_v1_2\\
"""

import os
import sys
import json
import math
import argparse
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


# -------------------------
# Helpers
# -------------------------

def now_ts():
    return dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def list_csv_files(folder):
    return [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.lower().endswith(".csv")]

def read_price_csv(path):
    """Autodetect , ; \t. Columns: Date, Open, High, Low, Close/(Adj Close), Volume."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

    lower = {c.lower(): c for c in df.columns}
    if 'date' not in lower:
        return None

    close_col = None
    for cand in ['adj close', 'adj_close', 'close']:
        if cand in lower:
            close_col = lower[cand]; break
    open_col = lower.get('open')
    high_col = lower.get('high')
    low_col  = lower.get('low')
    vol_col  = lower.get('volume')
    if not all([open_col, high_col, low_col, close_col, vol_col]):
        return None

    df = df[[lower['date'], open_col, high_col, low_col, close_col, vol_col]].copy()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()
    df = df[(df[['Open','High','Low','Close']] > 0).all(axis=1)]
    df = df[df['Volume'] >= 0]
    if len(df) < 5:
        return None
    return df

def compute_basic_series(df):
    s = df.copy()
    s['prev_close'] = s['Close'].shift(1)
    s['ret1'] = s['Close'].pct_change()
    rng = (s['High'] - s['Low'])
    prev = s['prev_close']
    s['day_range_norm'] = np.where(prev > 0, rng / prev, np.nan)
    span = (s['High'] - s['Low'])
    s['CLV'] = np.where(span > 0, (s['Close'] - s['Low']) / span, 0.5)
    s['RVOL'] = s['Volume'] / s['Volume'].rolling(60, min_periods=20).mean()
    s['RVOL'] = s['RVOL'].shift(1)  # ei lookaheadia
    return s

def forward_returns(close_s, max_h=15):
    out = {}
    for k in range(1, max_h+1):
        out[k] = (close_s.shift(-k) / close_s) - 1.0
    return out

def hold_return_from_fwd(R_fwd, L, H):
    rL  = R_fwd.get(L); rLH = R_fwd.get(L+H)
    if rL is None or rLH is None: return None
    return (1.0 + rLH) / (1.0 + rL) - 1.0

def ew_market_series(ret_dict):
    aligned = pd.concat(ret_dict.values(), axis=1)
    counts = (~aligned.isna()).sum(axis=1).astype(float)
    sums = aligned.fillna(0.0).sum(axis=1)
    ew = sums / counts.replace(0.0, np.nan)
    ew.name = 'MKT_EW'
    return ew

def regression_beta(y, x):
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 30:
        return 0.0, 0.0
    Y = df.iloc[:,0].values
    X = df.iloc[:,1].values
    X1 = np.column_stack([np.ones_like(X), X])
    try:
        b = np.linalg.lstsq(X1, Y, rcond=None)[0]
        alpha, beta = b[0], b[1]
        return beta, alpha
    except Exception:
        return 0.0, 0.0

def proportions_z_test(p1, n1, p2, n2):
    if min(n1, n2) == 0: return 0.0, 1.0
    p = (p1*n1 + p2*n2) / (n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    if se == 0: return 0.0, 1.0
    z = (p1 - p2) / se
    from math import erf, sqrt
    p_two = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    return z, p_two

def t_test_diff_of_means(m1, v1, n1, m2, v2, n2):
    if min(n1, n2) <= 1: return 0.0, 1.0
    se = math.sqrt(v1/n1 + v2/n2)
    if se == 0: return 0.0, 1.0
    t = (m1 - m2) / se
    from math import erf, sqrt
    p_two = 2*(1 - 0.5*(1+erf(abs(t)/sqrt(2))))
    return t, p_two

def fdr_bh(pvals, q=0.10):
    if len(pvals) == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    p = np.array(pvals, dtype=float)
    n = len(p); order = np.argsort(p)
    ranked = np.empty(n, dtype=int); ranked[order] = np.arange(1, n+1)
    p_adj = p * n / ranked
    p_adj_sorted = p_adj[order]
    for i in range(n-2, -1, -1):
        p_adj_sorted[i] = min(p_adj_sorted[i], p_adj_sorted[i+1])
    p_adj = np.empty_like(p); p_adj[order] = p_adj_sorted
    reject = p_adj <= q
    return reject, p_adj

def expected_shortfall_bottom(x, q=0.05, min_n=20):
    x = pd.Series(x).dropna(); n = len(x)
    if n == 0: return np.nan
    if n < min_n: return float(np.nanpercentile(x, q*100))
    k = max(1, int(math.floor(n*q)))
    xs = np.sort(x.values)
    return float(xs[:k].mean())

def shrink_prob_success(wins, trials, a=2.0, b=2.0):
    if trials <= 0: return np.nan
    return (wins + a) / (trials + a + b)

def shrink_mean_simple(mean, n, n0=20.0, grand_mean=0.0):
    w = n / (n + n0)
    return w * mean + (1 - w) * grand_mean

def rolling_windows_by_calendar(dates, train_days=504, test_days=63, min_tail=252):
    n = len(dates); i = train_days
    while i + test_days <= n:
        tr_start = dates[i - train_days]; tr_end = dates[i - 1]
        te_start = dates[i]; te_end = dates[i + test_days - 1]
        if i - train_days >= 0 and (i + test_days) <= n and (train_days >= min_tail):
            yield (tr_start, tr_end, te_start, te_end)
        i += test_days

def corr_series_min_periods(df_float32, s_float32, min_periods=60):
    """
    df_float32: DataFrame (float32), s_float32: Series (float32), samassa indeksissä.
    Palauttaa sarjan: corr(s, jokainen df:n sarake), NaN jos yhteisiä havaintoja < min_periods.
    """
    res = np.empty(df_float32.shape[1], dtype=np.float32)
    s_vals = s_float32.values
    s_notna = ~np.isnan(s_vals)
    for j, col in enumerate(df_float32.columns):
        x = df_float32[col].values
        valid = s_notna & ~np.isnan(x)
        n = int(valid.sum())
        if n < min_periods:
            res[j] = np.nan
            continue
        xv = x[valid].astype(np.float64, copy=False)
        yv = s_vals[valid].astype(np.float64, copy=False)
        xm = xv.mean(); ym = yv.mean()
        xs = xv - xm; ys = yv - ym
        denom = np.sqrt((xs*xs).sum()) * np.sqrt((ys*ys).sum())
        if denom == 0:
            res[j] = np.nan
        else:
            res[j] = float((xs*ys).sum() / denom)
    return pd.Series(res, index=df_float32.columns, dtype='float32')


# -------------------------
# Core
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-dir", required=True)
    ap.add_argument("--out-root", default="seasonality_reports\\runs")
    ap.add_argument("--min-history", type=int, default=500)
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--lags", nargs="+", type=int, default=[1,2,3,4,5])
    ap.add_argument("--holds", nargs="+", type=int, default=[1,3,5])
    ap.add_argument("--g-gap-bps", type=float, default=80.0)
    ap.add_argument("--event-quantile", type=float, default=0.90)
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--fdr", type=float, default=0.20)
    ap.add_argument("--default-hold", type=int, default=3)
    ap.add_argument("--risk-esq", type=float, default=0.05)
    ap.add_argument("--n0-shrink", type=float, default=20.0)
    # Uudet rajaukset
    ap.add_argument("--years-back", type=int, default=5, help="Käytä vain viimeiset N vuotta (oletus 5)")
    ap.add_argument("--date-start", type=str, default=None, help="Pakota aloituspäivä YYYY-MM-DD")
    ap.add_argument("--date-end", type=str, default=None, help="Pakota lopetuspäivä YYYY-MM-DD")
    ap.add_argument("--max-windows", type=int, default=None, help="Käsittele vain viimeiset N ikkunaa")
    args = ap.parse_args()

    run_ts = now_ts()
    out_dir = ensure_dir(os.path.join(args.out_root, run_ts, "crosslead_price_only_v1_2"))
    print(f"[INFO] Output -> {out_dir}")
    with open(os.path.join(out_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "run_ts": run_ts}, f, indent=2)

    price_dir = args.price_dir
    if not os.path.isdir(price_dir):
        print(f"[ERROR] price-dir '{price_dir}' not found."); sys.exit(1)

    paths = list_csv_files(price_dir)
    if not paths:
        print(f"[ERROR] No CSVs in {price_dir}"); sys.exit(1)

    tickers, data = [], {}
    for p in paths:
        sym = os.path.splitext(os.path.basename(p))[0].upper()
        df = read_price_csv(p)
        if df is None: continue
        df = compute_basic_series(df)
        adv60 = (df['Close'] * df['Volume']).rolling(60, min_periods=40).mean()
        adv60_med = adv60.median()
        if len(df) >= args.min_history and (adv60_med is not None) and (adv60_med >= args.min_adv):
            data[sym] = df; tickers.append(sym)

    if not data:
        print("[ERROR] No eligible tickers after filters."); sys.exit(1)

    # Kalenteri kaikista päivistä
    all_dates_full = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in data.values()])))
    last_all = pd.to_datetime(args.date_end) if args.date_end else all_dates_full.max()
    if args.date_start:
        first_all = pd.to_datetime(args.date_start)
    else:
        # viimeiset N vuotta
        first_all = last_all - pd.DateOffset(years=args.years_back)

    # Rajaa aikaväli
    all_dates = all_dates_full[(all_dates_full >= first_all) & (all_dates_full <= last_all)]
    if len(all_dates) < 600:
        print(f"[ERROR] Too few days in selected range ({len(all_dates)}). Widen range or lower train/test.")
        sys.exit(1)

    print(f"[INFO] Using calendar {first_all.date()} .. {last_all.date()}  (days: {len(all_dates)})")

    # Tuottosarjat & fwd-ret
    daily_ret = {sym: df['ret1'] for sym, df in data.items()}
    fwd_ret   = {sym: forward_returns(df['Close'], max_h=20) for sym, df in data.items()}
    mkt = ew_market_series(daily_ret)

    window_stability_rows = []
    cont_stats_A_windows = defaultdict(list)
    daily_candidate_pool = defaultdict(list)

    # Luo kaikki ikkunat tästä rajatusta kalenterista
    windows = list(rolling_windows_by_calendar(all_dates))
    if args.max-windows if False else False:  # just to avoid linter hit
        pass
    if args.max_windows is not None and len(windows) > args.max_windows:
        windows = windows[-args.max_windows:]
    print(f"[INFO] Windows to process: {len(windows)}")

    for (tr_start, tr_end, te_start, te_end) in windows:
        print(f"[INFO] Window Train[{tr_start.date()}..{tr_end.date()}] -> Test[{te_start.date()}..{te_end.date()}]")
        quant = args.event_quantile
        g_gap = args.g_gap_bps / 10000.0

        # Train returns frame (float32) vain tähän ikkunaan
        cols = []
        for sym in tickers:
            r = daily_ret[sym].loc[(daily_ret[sym].index >= tr_start) & (daily_ret[sym].index <= tr_end)]
            cols.append(r.rename(sym))
        corr_df = pd.concat(cols, axis=1).astype('float32')

        # Thresholdit, betat, baselinet
        train_rets, train_ranges, train_rvols = {}, {}, {}
        betas = {}
        baselines_prob, baselines_mean, es_by_BH = {}, {}, {}

        for sym, df in data.items():
            df_tr = df.loc[(df.index >= tr_start) & (df.index <= tr_end)].copy()
            if len(df_tr) < 100: continue

            r  = df_tr['ret1'].dropna()
            re = df_tr['day_range_norm'].dropna()
            rv = df_tr['RVOL'].dropna()

            train_rets[sym]   = np.nan if r.empty  else float(r.quantile(quant))
            train_ranges[sym] = np.nan if re.empty else float(re.quantile(quant))
            train_rvols[sym]  = np.nan if rv.empty else float(rv.quantile(quant))

            y = daily_ret[sym].loc[(daily_ret[sym].index >= tr_start) & (daily_ret[sym].index <= tr_end)]
            x = mkt.loc[(mkt.index >= tr_start) & (mkt.index <= tr_end)]
            betas[sym], _ = regression_beta(y, x)

            baselines_prob.setdefault(sym, {})
            baselines_mean.setdefault(sym, {})
            es_by_BH.setdefault(sym, {})
            for H in args.holds:
                Rf = fwd_ret[sym].get(H)
                if Rf is None:
                    baselines_prob[sym][H] = (np.nan, 0)
                    baselines_mean[sym][H] = (np.nan, np.nan, 0)
                    es_by_BH[sym][H] = np.nan
                    continue
                Rf_tr = Rf.loc[(Rf.index >= tr_start) & (Rf.index <= tr_end)].dropna()
                n = int(Rf_tr.notna().sum())
                if n == 0:
                    baselines_prob[sym][H] = (np.nan, 0)
                    baselines_mean[sym][H] = (np.nan, np.nan, 0)
                    es_by_BH[sym][H] = np.nan
                    continue
                p_up = float((Rf_tr > 0).mean())
                mu   = float(Rf_tr.mean())
                var  = float(Rf_tr.var(ddof=1)) if n > 1 else 0.0
                es   = expected_shortfall_bottom(Rf_tr.values, q=args.risk_esq)
                baselines_prob[sym][H] = (p_up, n)
                baselines_mean[sym][H] = (mu, var, n)
                es_by_BH[sym][H] = es

        # Eventit testijaksolla (cooldown)
        events_by_A = defaultdict(list)
        for sym, df in data.items():
            df_te = df.loc[(df.index >= te_start) & (df.index <= te_end)].copy()
            if len(df_te) == 0: continue
            r_q  = train_rets.get(sym, np.nan)
            re_q = train_ranges.get(sym, np.nan)
            rv_q = train_rvols.get(sym, np.nan)
            if np.isnan(r_q) or np.isnan(re_q) or np.isnan(rv_q): continue

            bud = (df_te['ret1'] >= r_q)
            reu = (df_te['day_range_norm'] >= re_q) & (df_te['Close'] > df_te['Open']) & (df_te['CLV'] >= 0.75)
            hvu = (df_te['Close'] > df_te['Open']) & (df_te['RVOL'] >= rv_q)
            prev_high = df['High'].shift(1).reindex(df_te.index)
            gng = (df_te['Open'] >= prev_high * (1.0 + g_gap)) & (df_te['Close'] >= df_te['Open'])

            ret_strength = np.where(df_te['ret1'] >= r_q,
                                    np.clip((df_te['ret1'] - r_q) / (abs(r_q) + 1e-6), 0, 1), 0.0)
            rng_strength = np.where(df_te['day_range_norm'] >= re_q,
                                    np.clip((df_te['day_range_norm'] - re_q) / (abs(re_q) + 1e-6), 0, 1), 0.0)
            clv_strength  = df_te['CLV'].clip(0,1).fillna(0.5)
            rvol_strength = np.where(df_te['RVOL'] >= rv_q,
                                     np.clip(df_te['RVOL'] / (rv_q + 1e-6), 0, 2) / 2.0, 0.0)
            S = 0.35*ret_strength + 0.25*rng_strength + 0.20*clv_strength + 0.20*rvol_strength
            S = pd.Series(S, index=df_te.index).fillna(0.0)

            evt_dates = list(df_te.index[(bud | reu | hvu | gng)])
            if not evt_dates: continue

            cands = [{'date': d, 'S': float(S.loc[d])} for d in evt_dates]
            cands.sort(key=lambda x: x['date'])
            kept, last = [], None; cd = args.cooldown
            for c in cands:
                if last is None:
                    kept.append(c); last = c['date']; continue
                if (c['date'] - last).days <= cd:
                    if c['S'] > kept[-1]['S']:
                        kept[-1] = c; last = c['date']
                else:
                    kept.append(c); last = c['date']
            events_by_A[sym].extend(kept)

        # TOP-K korrelat vain niille A:ille joilla on eventtejä
        def topk_for_A(symA, K):
            if symA not in corr_df.columns:
                return []
            sA = corr_df[symA]
            corrs = corr_series_min_periods(corr_df, sA, min_periods=60)
            corrs = corrs.drop(labels=[symA], errors='ignore').dropna()
            if len(corrs) == 0:
                return []
            return list(corrs.sort_values(ascending=False).head(K).index)

        # A->B analyysi
        contA_returns_byH = defaultdict(lambda: defaultdict(list))
        for A, evts in events_by_A.items():
            if not evts: continue
            Bs = topk_for_A(A, args.topk)
            if not Bs: continue

            beta_B = {b: betas.get(b, 0.0) for b in Bs}

            # A:n oma jatkuminen
            for e in evts:
                t0 = e['date']
                for H in args.holds:
                    RfA = fwd_ret[A].get(H)
                    if RfA is None or t0 not in RfA.index: continue
                    contA_returns_byH[A][H].append(float(RfA.loc[t0]))

            # A -> B lead-lag
            for B in Bs:
                base_prob = baselines_prob.get(B, {})
                base_mean = baselines_mean.get(B, {})
                esBH = es_by_BH.get(B, {})

                for L in args.lags:
                    for H in args.holds:
                        rets_abs, rets_neu = [], []
                        for e in evts:
                            t0 = e['date']
                            RtB = hold_return_from_fwd(fwd_ret[B], L, H)
                            if RtB is None or t0 not in RtB.index: continue
                            r = float(RtB.loc[t0])
                            if not np.isfinite(r): continue
                            rets_abs.append(r)

                            # market-hold
                            try:
                                idx_all = mkt.index
                                pos = idx_all.get_loc(t0)
                                start_i = pos + L; end_i = pos + L + H
                                if end_i < len(idx_all):
                                    mkt_slice = mkt.iloc[start_i:end_i+1].dropna()
                                    mkt_hold = float(mkt_slice.add(1.0).prod() - 1.0) if len(mkt_slice)>0 else 0.0
                                else:
                                    mkt_hold = 0.0
                            except Exception:
                                mkt_hold = 0.0
                            b = beta_B.get(B, 0.0)
                            rets_neu.append(r - b*mkt_hold)

                        trials = len(rets_abs)
                        if trials == 0: continue

                        p0, n0 = base_prob.get(H, (np.nan, 0))
                        mu0, v0, nmu0 = base_mean.get(H, (np.nan, np.nan, 0))
                        es0 = esBH.get(H, np.nan)
                        if not np.isfinite(p0): p0 = 0.5
                        if not np.isfinite(mu0): mu0 = 0.0
                        if (not np.isfinite(v0)) or v0 < 1e-12:
                            v0 = np.var(rets_abs, ddof=1) if trials>1 else 0.0
                        if not np.isfinite(es0): es0 = expected_shortfall_bottom(rets_abs, q=args.risk_esq)

                        arr_abs = np.array(rets_abs, dtype=float)
                        p_up = float((arr_abs > 0).mean())
                        mu   = float(arr_abs.mean())
                        var  = float(arr_abs.var(ddof=1)) if trials>1 else 0.0

                        z, p_prop = proportions_z_test(p_up, trials, p0, max(1, n0))
                        t, p_mean = t_test_diff_of_means(mu, var, trials, mu0, v0, max(1, nmu0))

                        window_stability_rows.append({
                            'window_train_start': tr_start, 'window_train_end': tr_end,
                            'window_test_start': te_start, 'window_test_end': te_end,
                            'A': A, 'B': B, 'L': L, 'H': H,
                            'p_up': p_up, 'trials': trials,
                            'lift_prob': p_up - p0, 'p_prop': p_prop,
                            'mean_ret': mu, 'lift_mean': mu - mu0, 'p_mean': p_mean
                        })

                        # Päiväkohtainen EV-proxy
                        p_up_sh = shrink_prob_success(wins=int((arr_abs > 0).sum()), trials=trials, a=2, b=2)
                        mu_sh   = shrink_mean_simple(mu, n=trials, n0=args.n0_shrink, grand_mean=mu0)
                        es_pen  = es0 if np.isfinite(es0) else 0.02
                        EV = p_up_sh * mu_sh - es_pen
                        for e in evts:
                            d = e['date']
                            daily_candidate_pool[d].append({'B': B, 'from_A': A, 'best_L': L, 'H': H,
                                                            'EV': EV, 'p_up_sh': p_up_sh, 'mu_sh': mu_sh})

        # A:n jatkumisen kooste tälle ikkunalle
        for A, dH in contA_returns_byH.items():
            for H, arr in dH.items():
                if not arr: continue
                arr = np.array(arr, dtype=float)
                cont_stats_A_windows[A].append({
                    'window_train_start': tr_start, 'window_train_end': tr_end,
                    'window_test_start': te_start, 'window_test_end': te_end,
                    'A': A, 'H': H, 'n': len(arr),
                    'p_up': float((arr>0).mean()),
                    'mean_ret': float(arr.mean()),
                    'es5': expected_shortfall_bottom(arr, q=0.05)
                })

        # FDR tälle testijaksolle (p_prop)
        idxs = [i for i,r in enumerate(window_stability_rows) if r['window_test_start']==te_start]
        if idxs:
            pvals = [window_stability_rows[i]['p_prop'] for i in idxs]
            reject, p_adj = fdr_bh(pvals, q=args.fdr)
            for j,irow in enumerate(idxs):
                window_stability_rows[irow]['p_adj'] = float(p_adj[j])
                window_stability_rows[irow]['rej']   = bool(reject[j])

    # -------- Aggregointi --------
    stability_df = pd.DataFrame(window_stability_rows)
    if 'rej' not in stability_df.columns:
        stability_df['rej'] = False; stability_df['p_adj'] = 1.0

    group_cols = ['A','B','L','H']
    if len(stability_df):
        stab = (stability_df
                .assign(pos_sig=lambda d: (d['lift_prob']>0) & (d['rej']==True))
                .groupby(group_cols)
                .agg(windows=('pos_sig','size'),
                     pos_sig=('pos_sig','sum'),
                     avg_lift=('lift_prob','mean'),
                     avg_mean_lift=('lift_mean','mean'),
                     avg_padj=('p_adj','mean'),
                     total_trials=('trials','sum'))
                .reset_index())
        stab['stability_ratio'] = np.where(stab['windows']>0, stab['pos_sig']/stab['windows'], np.nan)
    else:
        stab = pd.DataFrame(columns=group_cols+['windows','pos_sig','avg_lift','avg_mean_lift','avg_padj','total_trials','stability_ratio'])

    best_rows = []
    for (A,B), g in stab.groupby(['A','B']):
        g2 = g.sort_values(['stability_ratio','avg_lift','total_trials'], ascending=[False,False,False]).iloc[0]
        best_rows.append({
            'A': A, 'B': B,
            'best_L': int(g2['L']), 'best_H': int(g2['H']),
            'stability_ratio': float(g2['stability_ratio']),
            'avg_excess_lift_prob': float(g2['avg_lift']),
            'avg_excess_lift_mean': float(g2['avg_mean_lift']),
            'avg_p_adj': float(g2['avg_padj']),
            'total_trials': int(g2['total_trials'])
        })
    leadlag_df = pd.DataFrame(best_rows).sort_values(['stability_ratio','avg_excess_lift_prob','total_trials'], ascending=[False,False,False])

    leaders_rank = (leadlag_df.groupby('A')
                    .agg(n_sig_pairs=('B','size'),
                         avg_stability=('stability_ratio','mean'),
                         avg_excess_lift=('avg_excess_lift_prob','mean'))
                    .reset_index()
                    .sort_values(['n_sig_pairs','avg_stability','avg_excess_lift'], ascending=[False,False,False]))

    followers_rows = []
    for A, g in leadlag_df.groupby('A'):
        gg = g.sort_values(['stability_ratio','avg_excess_lift_prob','total_trials'], ascending=[False,False,False]).head(15)
        for _, r in gg.iterrows():
            followers_rows.append({
                'A': A, 'B': r['B'],
                'best_L': int(r['best_L']), 'best_H': int(r['best_H']),
                'stability_ratio': float(r['stability_ratio']),
                'avg_excess_lift_prob': float(r['avg_excess_lift_prob']),
                'avg_excess_lift_mean': float(r['avg_excess_lift_mean']),
                'avg_p_adj': float(r['avg_p_adj']),
                'total_trials': int(r['total_trials'])
            })
    followers_df = pd.DataFrame(followers_rows)

    cont_rows = []
    for A, rows in cont_stats_A_windows.items():
        if not rows: continue
        dfA = pd.DataFrame(rows)
        for H, g in dfA.groupby('H'):
            n = int(g['n'].sum())
            if n == 0: continue
            p_up_avg = np.average(g['p_up'], weights=g['n'])
            mean_ret_avg = np.average(g['mean_ret'], weights=g['n'])
            es5_avg = np.average(g['es5'], weights=g['n'])
            cont_rows.append({'A': A, 'H': int(H), 'n': n,
                              'p_up': float(p_up_avg), 'mean_ret': float(mean_ret_avg), 'es5': float(es5_avg)})
    continuation_df = pd.DataFrame(cont_rows).sort_values(['A','H'])

    # Daily picks
    default_H = args.default_hold
    picks = []
    for d, items in sorted(daily_candidate_pool.items()):
        cand = pd.DataFrame(items)
        if len(cand)==0: continue
        agg = cand.groupby('B').agg(EV=('EV','mean'),
                                    n_sources=('from_A','nunique'),
                                    best_L=('best_L','median')).reset_index()
        top = agg.sort_values(['EV','n_sources'], ascending=[False,False]).iloc[0]
        picks.append({'date': d, 'B': top['B'], 'EV': float(top['EV']),
                      'n_sources': int(top['n_sources']), 'best_L': int(top['best_L']), 'H': default_H})
    picks_df = pd.DataFrame(picks).sort_values('date')

    # Picks performance
    perf_rows = []
    if len(picks_df):
        for _, r in picks_df.iterrows():
            d = r['date']; B = r['B']; H = int(r['H'])
            dfB = data.get(B)
            if dfB is None or d not in dfB.index: continue
            try: idx = dfB.index.get_loc(d)
            except Exception: continue
            if idx+1+H >= len(dfB.index): continue
            entry_open = float(dfB['Open'].iloc[idx+1])
            exit_close = float(dfB['Close'].iloc[idx+1+H])
            ret = (exit_close / entry_open) - 1.0
            perf_rows.append({'date': d, 'B': B, 'ret': ret, 'H': H,
                              'entry_open': entry_open, 'exit_close': exit_close,
                              'EV_at_pick': r['EV'], 'n_sources': r['n_sources'], 'best_L': r['best_L']})
    perf_df = pd.DataFrame(perf_rows).sort_values('date')

    def summarize_strategy(df):
        if len(df)==0:
            return pd.DataFrame([{'trades':0,'win%':np.nan,'avg':np.nan,'CAGR':np.nan,'Sharpe':np.nan,'maxDD':np.nan}])
        rets = df['ret'].values
        trades = len(rets)
        winp = float((rets>0).mean())
        avg = float(np.mean(rets))
        equity = (1.0 + pd.Series(rets, index=df['date'])).cumprod()
        days = (df['date'].max() - df['date'].min()).days + 1
        years = max(1e-9, days/365.25)
        CAGR = float(equity.iloc[-1]**(1/years) - 1.0) if len(equity)>1 else np.nan
        sd = np.std(rets, ddof=1) if trades>1 else np.nan
        Sharpe = float((avg / sd) * math.sqrt(252)) if (sd and sd>1e-12) else np.nan
        roll_max = equity.cummax()
        dd = equity/roll_max - 1.0
        maxDD = float(dd.min()) if len(dd) else np.nan
        return pd.DataFrame([{'trades': trades, 'win%': winp, 'avg': avg, 'CAGR': CAGR, 'Sharpe': Sharpe, 'maxDD': maxDD}])

    strat_summary = summarize_strategy(perf_df)

    perf_by_period = pd.DataFrame()
    if len(perf_df):
        perf_df['year'] = perf_df['date'].dt.year
        perf_df['quarter'] = perf_df['date'].dt.to_period('Q').astype(str)
        perf_by_period = (perf_df.groupby(['year','quarter'])
                          .agg(trades=('ret','size'),
                               winp=('ret', lambda x: float((x>0).mean())),
                               avg=('ret','mean'),
                               sum=('ret','sum'))
                          .reset_index()
                          .sort_values(['year','quarter']))

    # Write outputs
    def w(df, name):
        path = os.path.join(out_dir, name); df.to_csv(path, index=False)
    w(leadlag_df, "lead_lag_matrix.csv")
    w(leaders_rank, "leaders_rank.csv")
    w(followers_df, "followers_map.csv")
    w(continuation_df, "continuation_stats_A.csv")
    w(stability_df, "window_stability.csv")
    w(picks_df, "daily_picks.csv")
    w(strat_summary, "strategy_performance.csv")
    w(perf_by_period, "performance_by_period.csv")

    with open(os.path.join(out_dir, "WARNINGS.txt"), "w", encoding="utf-8") as f:
        f.write("Survivorship-bias risk: if delisted stocks are missing from price_cache, results may be optimistic.\n")

    print(f"\n[OK] Done. Output in: {out_dir}\n")


if __name__ == "__main__":
    main()
