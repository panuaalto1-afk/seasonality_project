#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Backtest auto_deciderille (kevyt versio, EI kopioi price_cachea per päivä).

- Käyttää yhtä olemassa olevaa price_cachea (runs\...\price_cache) koko ajon ajan
- Välittää sekä ml_unified_pipeline:lle että auto_deciderille `--vintage_cutoff`-päivän → ne lukevat hinnat vain <= cutoff
- Tallentaa equity_curven, trade-logit ja yhteenvedon
"""

import os, sys, glob, argparse, subprocess, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL=True
except Exception:
    HAVE_MPL=False

def _safe_mkdir(p): os.makedirs(p, exist_ok=True)
def _dmy(s): return datetime.strptime(s, "%Y-%m-%d")
def _iter_days(a,b):
    d=_dmy(a); e=_dmy(b)
    while d<=e:
        if d.weekday()<5: yield d
        d+=timedelta(days=1)

def _run(cmd:list):
    print(">>", " ".join([f'"{x}"' if (" " in x and not x.startswith("--")) else x for x in cmd]))
    subprocess.run(cmd, check=True)

def _latest_price_cache(project_root:str)->str:
    base=os.path.join(project_root,"seasonality_reports","runs")
    cands=[]
    for r in glob.glob(os.path.join(base,"*")):
        pc=os.path.join(r,"price_cache")
        if os.path.isdir(pc): cands.append(pc)
    if not cands: raise RuntimeError("Ei löydy price_cachea runs/ alta.")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def _close_from_cache(price_cache: str, ticker: str, cutoff: datetime) -> float:
    p1=os.path.join(price_cache,f"{ticker}.csv")
    p2=os.path.join(price_cache,f"{ticker.upper()}.csv")
    p = p1 if os.path.isfile(p1) else (p2 if os.path.isfile(p2) else "")
    if not p: return np.nan
    df=pd.read_csv(p)
    cols={c.lower():c for c in df.columns}
    dcol = cols.get("date") or cols.get("timestamp") or list(df.columns)[0]
    ccol = cols.get("close") or cols.get("adj close") or cols.get("adj_close") or list(df.columns)[-1]
    df[dcol]=pd.to_datetime(df[dcol], errors="coerce")
    df=df[df[dcol] <= pd.Timestamp(cutoff)]
    if df.empty: return np.nan
    try: return float(df[ccol].iloc[-1])
    except Exception:
        try: return float(str(df[ccol].iloc[-1]).replace(",", "."))
        except Exception: return np.nan

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--universe_csv", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    # ML
    ap.add_argument("--feature_mode", choices=["split","shared"], default="split")
    ap.add_argument("--gate_alpha", type=float, default=0.03)
    ap.add_argument("--train_years", type=int, default=7)
    ap.add_argument("--min_samples_per_regime", type=int, default=500)
    # options (ei käytetä oletuksena)
    ap.add_argument("--use_options", default="false")
    # kulut/slippage (bps per sisään ja ulos)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--slip_bps", type=float, default=5.0)
    # decider
    ap.add_argument("--decider_config", default="")
    ap.add_argument("--name", default="bt_decider_light")
    return ap.parse_args()

def main():
    args=parse_args()
    py=sys.executable
    project=args.project_root

    uni=pd.read_csv(args.universe_csv)
    tcol=None
    for c in uni.columns:
        if str(c).lower() in ("ticker","symbol"): tcol=c; break
    if tcol is None: raise RuntimeError("universe_csv: 'ticker'/'symbol' sarake puuttuu.")
    tickers=sorted(set(uni[tcol].astype(str).str.upper().str.strip().tolist()))  # vain listaus; featuret/labels tekee pipeline

    out_root=os.path.join(project,"seasonality_reports","backtests", args.name)
    _safe_mkdir(out_root)
    equity_csv=os.path.join(out_root,"equity_curve.csv")
    trades_csv=os.path.join(out_root,"trades.csv")
    summary_txt=os.path.join(out_root,"summary.txt")
    for p in (equity_csv,trades_csv):
        if os.path.isfile(p): os.remove(p)

    base_cache=_latest_price_cache(project)
    portfolio_state=os.path.join(out_root,"portfolio_state.json")
    if os.path.isfile(portfolio_state): os.remove(portfolio_state)

    cfg_path=args.decider_config
    if not cfg_path:
        cfg_path=os.path.join(out_root,"auto_decider_config.json")
        with open(cfg_path,"w",encoding="utf-8") as f:
            f.write("""{
  "allow_index_shorts": true,
  "weights": { "ml": 0.70, "opt": 0.00, "seas": 0.15, "sector": 0.10, "regime": 0.05 },
  "th_long_riskon": 0.60, "th_long_neutral": 0.65,
  "max_positions": 5, "max_entries_day": 1, "max_entries_week": 3, "max_weight_pct": 20.0,
  "max_pos_per_sector": 2,
  "exit": { "p_up_cutoff": 0.45, "sma100_lookback": 100, "sma20_lookback": 20, "atr_lookback": 14, "tp_atr_mult": 1.5, "tp_pct": 0.10, "max_hold_days_no_gain": 15 }
}""")

    cost = (args.cost_bps + args.slip_bps)/10000.0

    rows_equity=[]
    rows_trades=[]
    live = {}  # ticker -> {side, weight, entry_px, entry_adj, entry_date, regime_entry}

    for day in _iter_days(args.start, args.end):
        dstr=day.strftime("%Y-%m-%d")
        run_root=os.path.join(project,"seasonality_reports","runs_bt", f"{day.strftime('%Y-%m-%d')}_BT")
        _safe_mkdir(run_root)

        # ML (lukee samaa base_cachea, mutta --vintage_cutoff rajaa hinnat)
        _run([py, os.path.join(project,"ml_unified_pipeline.py"),
              "--run_root", run_root,
              "--today", dstr,
              "--train_years", str(args.train_years),
              "--universe_csv", args.universe_csv,
              "--feature_mode", args.feature_mode,
              "--gate_alpha", str(args.gate_alpha),
              "--min_samples_per_regime", str(args.min_samples_per_regime),
              "--vintage_cutoff", dstr])

        # (valinnainen) optiot → jätetään oletuksena pois; jos käytät, välitä cutoff myös sinne

        # Decider --commit (sama base_cache + cutoff)
        _run([py, os.path.join(project,"auto_decider.py"),
              "--project_root", project,
              "--universe_csv", args.universe_csv,
              "--run_root", run_root,
              "--price_cache_dir", base_cache,
              "--today", dstr,
              "--portfolio_state", portfolio_state,
              "--config", cfg_path,
              "--vintage_cutoff", dstr,
              "--commit", "true"])

        # Lue state (commitin jälkeen)
        st={"positions":[]}
        try:
            with open(portfolio_state,"r",encoding="utf-8") as f:
                st=json.load(f)
        except Exception:
            pass
        new_positions=st.get("positions",[])

        # Poistuneet → realisoi treidi
        new_set=set([p["ticker"] for p in new_positions])
        old_set=set(live.keys())
        exited = old_set - new_set
        for t in sorted(exited):
            info=live[t]
            exit_px=_close_from_cache(base_cache, t, day)
            if np.isnan(exit_px): del live[t]; continue
            if info["side"]=="LONG":
                exit_adj = exit_px * (1.0 - cost)
                ret = exit_adj/info["entry_adj"] - 1.0
            else:
                exit_adj = exit_px * (1.0 + cost)
                ret = info["entry_adj"]/exit_adj - 1.0
            held_days=(day - _dmy(info["entry_date"])).days
            rows_trades.append({
                "ticker": t, "side": info["side"],
                "entry_date": info["entry_date"], "exit_date": dstr,
                "ret_pct": round(100*ret, 3), "held_days": held_days
            })
            del live[t]

        # Uudet/nykyiset
        for p in new_positions:
            t=str(p["ticker"]).upper().strip()
            side=str(p.get("side","LONG")).upper()
            w=float(p.get("weight_pct",0.0))
            entry=float(p.get("entry_px", np.nan))
            if t not in live:
                entry_adj = entry * (1.0 + cost) if side=="LONG" else entry * (1.0 - cost)
                live[t]={"side":side,"weight":w,"entry_px":entry,"entry_adj":entry_adj,
                         "entry_date": p.get("entry_date", dstr)}

        # Mark-to-market
        used_w=sum([info["weight"] for info in live.values()])
        cash=max(0.0, 100.0 - used_w)
        pv=0.0
        for t,info in live.items():
            close=_close_from_cache(base_cache, t, day)
            if np.isnan(close) or info["entry_adj"]==0: continue
            factor = close / info["entry_adj"] if info["side"]=="LONG" else info["entry_adj"] / close
            pv += info["weight"] * factor
        equity_today=cash + pv
        rows_equity.append({"date": dstr, "equity": round(equity_today,4), "cash": round(cash,2), "invested_w": round(used_w,2)})

    # CSV:t
    eq_df=pd.DataFrame(rows_equity); eq_df.to_csv(equity_csv, index=False)
    tr_df=pd.DataFrame(rows_trades); tr_df.to_csv(trades_csv, index=False)

    # Yhteenveto
    eq_df["ret"]=eq_df["equity"].pct_change().fillna(0.0)
    cum = (eq_df["equity"].iloc[-1]/eq_df["equity"].iloc[0]-1.0) if len(eq_df)>=2 else 0.0
    peak=-1e9; maxdd=0.0
    for v in eq_df["equity"]:
        if v>peak: peak=v
        dd=(peak - v)/peak if peak>0 else 0.0
        if dd>maxdd: maxdd=dd
    ar = ((1+cum)**(252/max(1,len(eq_df))) -1) if len(eq_df)>2 else cum
    vol = (eq_df["ret"].std()*np.sqrt(252)) if len(eq_df)>2 else 0.0
    sharpe = (ar/vol) if vol>0 else np.nan

    with open(summary_txt,"w",encoding="utf-8") as f:
        f.write("\n".join([
            f"Backtest: {args.start} .. {args.end}  (days={len(eq_df)})",
            f"Final equity: {eq_df['equity'].iloc[-1]:.2f}",
            f"Cumulative: {cum*100:.2f}%",
            f"MaxDD: {maxdd*100:.2f}%",
            f"Ann.Return (rough): {ar*100:.2f}%",
            f"Ann.Vol: {vol*100:.2f}%",
            f"Sharpe-ish: {sharpe:.2f}",
            "",
            f"Equity CSV: {equity_csv}",
            f"Trades CSV: {trades_csv}"
        ]))

    if HAVE_MPL:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(pd.to_datetime(eq_df["date"]), eq_df["equity"])
            plt.title("Equity curve"); plt.xlabel("Date"); plt.ylabel("Equity (base=100)")
            out_png=os.path.join(out_root,"equity_curve.png")
            plt.tight_layout(); plt.savefig(out_png, dpi=120)
            print("[OK] Plot          :", out_png)
        except Exception:
            pass

    print("[OK] Equity curve  :", equity_csv)
    print("[OK] Trades log    :", trades_csv)
    print("[OK] Summary       :", summary_txt)

if __name__=="__main__":
    main()
