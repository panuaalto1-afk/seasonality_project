# regime_overlay.py  v2025-10-17f
import argparse, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT_DEFAULT = PROJECT_ROOT / "seasonality_reports"
UNIVERSE_CSV = REPORTS_ROOT_DEFAULT / "constituents_raw.csv"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# -------- IO --------
def read_csv_close(path: Path) -> pd.Series | None:
    if not path.exists(): return None
    try:
        df = pd.read_csv(path)
        dcol = "Date" if "Date" in df.columns else (df.columns[0])
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.tz_localize(None)
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        if "Close" not in df.columns: return None
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=[dcol, "Close"]).sort_values(dcol)
        s = df.set_index(dcol)["Close"]; s.name = path.stem
        return s
    except Exception as e:
        print(f"[WARN] read_csv_close({path.name}) failed: {e}")
        return None

def read_universe(universe_csv: Path) -> list[str]:
    if not universe_csv.exists(): return []
    df = pd.read_csv(universe_csv)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol") or list(df.columns)[0]
    tickers = [str(x).strip().upper().replace(".", "-") for x in df[tcol].dropna().astype(str)]
    blacklist = {"SPY","QQQ","IWM","DIA","XLU","XLK","XLY","XLE","XLF","XLI","XLP","XLV","XLC","XLB","XLRE"}
    return [t for t in tickers if t not in blacklist and t.isalpha()]

# -------- Metrics --------
def compute_breadth_metrics(pc_dir: Path, universe: list[str], asof: dt.date) -> pd.DataFrame:
    ser = []
    for t in universe:
        s = read_csv_close(pc_dir / f"{t}.csv")
        if s is None or s.empty: continue
        s = s[s.index >= (pd.Timestamp(asof) - pd.Timedelta(days=1200))]
        ser.append(s.rename(t))
    if not ser: raise RuntimeError("Ei osakesarjoja breadthiin.")

    wide = pd.concat(ser, axis=1).apply(pd.to_numeric, errors="coerce").sort_index().ffill().dropna(how="all")
    sma50  = wide.rolling(50,  min_periods=30).mean()
    sma200 = wide.rolling(200, min_periods=120).mean()

    denom = wide.count(axis=1).clip(lower=1)
    pct_above_50  = (wide.gt(sma50)).sum(axis=1)  / denom
    pct_above_200 = (wide.gt(sma200)).sum(axis=1) / denom

    ret1 = wide.pct_change(1)
    adv  = (ret1.gt(0)).sum(axis=1).astype(float)
    dec  = (ret1.lt(0)).sum(axis=1).astype(float)
    ad_line = (adv - dec).cumsum()
    ad_slope_5d = ad_line.diff(5)

    return pd.DataFrame({
        "pct_sma50": pct_above_50,
        "pct_sma200": pct_above_200,
        "ad_slope_5d": ad_slope_5d
    }).dropna(how="all")

def read_proxy(pc_dir: Path, sym: str): return read_csv_close(pc_dir / f"{sym}.csv")

def pct_rank(x: pd.Series, lookback=252):
    if x is None or x.empty: return pd.Series(dtype=float)
    roll = x.rolling(lookback, min_periods=max(5, lookback//5))
    return (x - roll.min()) / (roll.max() - roll.min())

def latest(series: pd.Series, asof: dt.date):
    if series is None or series.empty: return np.nan
    s = series.loc[:pd.Timestamp(asof)]
    if s.empty: return np.nan
    return float(s.iloc[-1])

# -------- Composite --------
def build_composite(pc_dir: Path, breadth_df: pd.DataFrame, asof: dt.date):
    vix = read_proxy(pc_dir, "^VIX") or read_proxy(pc_dir, "VIX"); vix_pct = pct_rank(vix, 252); vix_comp = 1.0 - vix_pct
    xlu, spy = read_proxy(pc_dir, "XLU"), read_proxy(pc_dir, "SPY")
    util_mom = - (xlu/spy).pct_change(21) if (xlu is not None and spy is not None) else None
    iwm, qqq = read_proxy(pc_dir, "IWM"), read_proxy(pc_dir, "QQQ")
    cap_rot = (iwm.pct_change(21) - qqq.pct_change(21)) if (iwm is not None and qqq is not None) else None

    b_scaled = ((breadth_df["pct_sma50"] + breadth_df["pct_sma200"])/2.0).clip(0,1)

    comp_df = pd.DataFrame(index=breadth_df.index)
    comp_df["breadth"] = b_scaled
    if util_mom is not None: comp_df["util_mom"] = util_mom.reindex(comp_df.index)
    if cap_rot  is not None: comp_df["cap_rot"]  = cap_rot.reindex(comp_df.index)
    if vix_comp is not None and not vix_comp.empty: comp_df["vix_comp"] = vix_comp.reindex(comp_df.index)

    base_w = {"breadth":0.4, "cap_rot":0.3, "util_mom":0.2, "vix_comp":0.1}
    avail = [c for c in comp_df.columns if comp_df[c].notna().any()]
    wsum = sum(base_w.get(c,0) for c in avail)
    weights = {c:(base_w.get(c,0)/wsum if wsum>0 else 0) for c in avail}

    for c in comp_df.columns:
        if c=="breadth": continue
        comp_df[c] = pct_rank(comp_df[c], 252)

    comp_df["composite"] = 0.0
    for c,w in weights.items():
        comp_df["composite"] = comp_df["composite"].add(comp_df[c].ffill().fillna(0)*w, fill_value=0)

    today_vals = {
        "breadth": latest(comp_df["breadth"], asof),
        "cap_rot": latest(comp_df["cap_rot"], asof) if "cap_rot" in comp_df.columns else np.nan,
        "util_mom": latest(comp_df["util_mom"], asof) if "util_mom" in comp_df.columns else np.nan,
        "vix_comp": latest(comp_df["vix_comp"], asof) if "vix_comp" in comp_df.columns else np.nan,
        "composite": latest(comp_df["composite"], asof)
    }
    return comp_df, today_vals

def classify_regime(x: float) -> str:
    if pd.isna(x): return "Neutral"
    if x >= 0.66: return "RiskON"
    if x <= 0.33: return "RiskOFF"
    return "Neutral"

_SECTOR_ETFS = {
    "COMMUNICATION_SERVICES":"XLC","CONSUMER_DISCRETIONARY":"XLY","CONSUMER_STAPLES":"XLP",
    "ENERGY":"XLE","FINANCIALS":"XLF","HEALTH_CARE":"XLV","INDUSTRIALS":"XLI",
    "INFORMATION_TECHNOLOGY":"XLK","MATERIALS":"XLB","REAL_ESTATE":"XLRE","UTILITIES":"XLU"
}

def _sector_21d_momentum(pc_dir: Path, asof: dt.date) -> dict:
    out={}
    for sec,etf in _SECTOR_ETFS.items():
        s = read_proxy(pc_dir, etf)
        if s is None or s.empty: out[sec]=np.nan; continue
        roc = s.pct_change(21).loc[:pd.Timestamp(asof)]
        out[sec] = float(roc.iloc[-1]) if not roc.empty else np.nan
    return out

def load_mapping(reports_root=REPORTS_ROOT_DEFAULT, universe_csv=None) -> pd.DataFrame:
    rr = Path(reports_root)
    uc = Path(universe_csv) if universe_csv else (rr/"constituents_raw.csv")
    if not uc.exists(): return pd.DataFrame(columns=["symbol","sector","cap_bucket"])
    df = pd.read_csv(uc)
    cols={c.lower():c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol") or list(df.columns)[0]
    scol = cols.get("sector") or "sector"
    ccol = cols.get("cap_bucket") or cols.get("size") or "cap_bucket"
    out = pd.DataFrame({"symbol": df[tcol].astype(str).str.upper().str.replace(".","-",regex=False)})
    out["sector"] = df[scol] if scol in df.columns else np.nan
    out["cap_bucket"] = df[ccol] if ccol in df.columns else np.nan
    out["sector"] = out["sector"].astype(str).str.upper().str.replace(" ","_")
    return out

# -------- Compat: build_regime_snapshot --------
def build_regime_snapshot(*args, **kwargs):
    """
    (A) Uusi:  build_regime_snapshot(run_root, today=YYYY-mm-dd) -> dict snapshot
    (B) Vanha: build_regime_snapshot(price_cache_dir, mapping_df, date) -> (snapshot, sector_mom_dict, cap_mom_float)
    """
    run_or_pc = Path(args[0]) if len(args)>=1 else None
    mapping_df = args[1] if len(args)>=2 and isinstance(args[1], pd.DataFrame) else None
    day_arg = kwargs.get("today", None)
    if day_arg is None and len(args)>=2 and not isinstance(args[1], pd.DataFrame): day_arg = args[1]
    if day_arg is None and len(args)>=3: day_arg = args[2]
    asof = (dt.date.fromisoformat(str(day_arg)) if day_arg is not None and not isinstance(day_arg, dt.date)
            else (day_arg or dt.date.today()))

    pc_dir = None
    if run_or_pc is not None:
        pc_dir = (run_or_pc/"price_cache") if (run_or_pc/"price_cache").exists() else run_or_pc
    if pc_dir is None: raise RuntimeError("build_regime_snapshot: ei price_cachea.")

    # universumi breadthiin
    if mapping_df is not None and "symbol" in mapping_df.columns and not mapping_df.empty:
        universe = (mapping_df["symbol"].astype(str).str.upper().str.replace(".","-",regex=False).tolist())
        blacklist = set(_SECTOR_ETFS.values()) | {"SPY","QQQ","IWM","DIA","^VIX","VIX","TIP","RINF","HYG","LQD","UUP","USO","GLD","CPER","SHY","IEF","TLT"}
        universe = [t for t in universe if t and t.isalpha() and t not in blacklist]
    else:
        uc = UNIVERSE_CSV if UNIVERSE_CSV.exists() else None
        universe = read_universe(uc) if uc else []
        if not universe:
            all_csv = [p.stem for p in pc_dir.glob("*.csv")]
            blacklist = set(_SECTOR_ETFS.values()) | {"SPY","QQQ","IWM","DIA","^VIX","VIX","TIP","RINF","HYG","LQD","UUP","USO","GLD","CPER","SHY","IEF","TLT"}
            universe = [t for t in all_csv if t.isalpha() and t not in blacklist]

    breadth_df = compute_breadth_metrics(pc_dir, universe, asof)
    comp_df, today_vals = build_composite(pc_dir, breadth_df, asof)
    composite = today_vals["composite"]
    risk_regime = classify_regime(composite)

    sector_mom = _sector_21d_momentum(pc_dir, asof)
    iwm, qqq = read_proxy(pc_dir,"IWM"), read_proxy(pc_dir,"QQQ")
    cap_mom = np.nan
    if iwm is not None and qqq is not None:
        r = (iwm.pct_change(21) - qqq.pct_change(21)).loc[:pd.Timestamp(asof)]
        if not r.empty: cap_mom = float(r.iloc[-1])

    snapshot = {
        "date": asof.isoformat(),
        "risk_regime": risk_regime,
        "composite": float(composite) if pd.notna(composite) else np.nan,
        "sector_mom_21d": sector_mom,
        "cap_mom_21d": cap_mom,
    }

    # Palautusmuoto:
    if len(args) >= 3 or (len(args)>=2 and not isinstance(args[1], pd.DataFrame)):
        return snapshot, sector_mom, cap_mom  # VANHA
    else:
        return snapshot  # UUSI

# -------- Uusi: attach_regime_overlay --------
def attach_regime_overlay(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Joustava liima:
      - attach_regime_overlay(df, price_cache_dir, mapping_df, date)
      - attach_regime_overlay(df, snapshot_dict)
      - attach_regime_overlay(df, run_root=..., today=...)

    Lisää sarakkeet: risk_regime, cap_momentum_21d, regime_composite
    """
    snapshot = None; cap_mom = np.nan

    if len(args) >= 3 and isinstance(args[0], (str, Path)) and isinstance(args[1], pd.DataFrame):
        snap, _, cap = build_regime_snapshot(args[0], args[1], args[2])
        snapshot, cap_mom = snap, cap
    elif len(args) >= 1 and isinstance(args[0], dict):
        snapshot = args[0]; cap_mom = snapshot.get("cap_mom_21d", np.nan)
    elif "run_root" in kwargs or "price_cache_dir" in kwargs:
        run_root = kwargs.get("run_root") or kwargs.get("price_cache_dir")
        today = kwargs.get("today", dt.date.today())
        snapshot = build_regime_snapshot(run_root, today=today)
        cap_mom = snapshot.get("cap_mom_21d", np.nan)

    if snapshot is None:
        snapshot = {"risk_regime":"Neutral","composite":np.nan,"cap_mom_21d":np.nan}

    out = df.copy()
    out["risk_regime"] = snapshot.get("risk_regime","Neutral")
    out["cap_momentum_21d"] = cap_mom
    if "composite" in snapshot: out["regime_composite"] = snapshot["composite"]
    return out

# -------- CLI --------
def main():
    print("[regime_overlay] v2025-10-17f")
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=None)
    ap.add_argument("--reports_root", default=str(REPORTS_ROOT_DEFAULT))
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--research", type=int, default=0)
    args = ap.parse_args()

    reports_root = Path(args.reports_root)
    run_root = Path(args.run_root); pc_dir = run_root/"price_cache"
    if not pc_dir.exists(): raise FileNotFoundError(f"price_cache ei löydy: {pc_dir}")
    asof = dt.date.fromisoformat(args.today) if args.today else dt.date.today()

    snap = build_regime_snapshot(run_root, today=asof)
    out_snap = run_root/"reports"/f"regime_state_{asof.strftime('%Y%m%d')}.csv"
    ensure_dir(out_snap.parent)
    pd.DataFrame([{
        "date": snap["date"], "risk_regime": snap["risk_regime"], "composite": snap["composite"]
    }]).to_csv(out_snap, index=False)

    if args.research:
        rr = reports_root/"aggregates"/"regime_research"/asof.isoformat()
        ensure_dir(rr)
        with open(rr/"summary.txt","w",encoding="utf-8") as f:
            f.write(f"AS OF {asof.isoformat()}\nRegime: {snap['risk_regime']}\nComposite: {snap['composite']:.3f}\n")

    print(f"[OK] Regime overlay valmis: {snap['risk_regime']} (composite={snap['composite']:.3f})")
    print(f"[SNAPSHOT] {out_snap}")

if __name__ == "__main__":
    main()
