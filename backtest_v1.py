# backtest_v1.py
# ------------------------------------------------------------
# EOD-backtest Seasonality-projektin tuotoksista (per-run).
# - Hinnat: seasonality_reports/price_cache/*.csv
# - Aggregaatit: seasonality_reports/runs/<RUN>/aggregates/
#   (daily_top5_calendar.csv, upcoming_windows_next_45d.csv)
# - Entry: signaali t -> entry t+1 open, exit SL/TP/time
# - Tulosteet: <RUN>/backtest/{results_summary.csv,trades.csv,equity_curve.csv}
#
# Käyttö:
#   py backtest_v1.py                                # uusin run
#   py backtest_v1.py --run-dir 2025-10-04_2145
#   py backtest_v1.py --require-seasonality 0        # aja ilman kalenteripakkoa
#   py backtest_v1.py --use-breakout 1 --use-rsi 0   # vain breakout
#   py backtest_v1.py --debug 1                      # tulosta diagnostiikkaa
# ------------------------------------------------------------

import os, sys, glob, math, argparse, datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import numpy as np

# --- projektipolut ---
REPORT_ROOT = Path("seasonality_reports")
PRICE_CACHE = REPORT_ROOT / "price_cache"
RUNS_DIR    = REPORT_ROOT / "runs"
VINTAGE_DIR = REPORT_ROOT / "vintage"  # optional reliability

# --- globaali debug-lippu (asetetaan main():ssa args.debug perusteella) ---
DEBUG = False


# ----------------------------- apu: uusin run -------------------------------
def latest_run_dir() -> Optional[Path]:
    if not RUNS_DIR.exists():
        return None
    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


# ----------------------------- robusti hinnanluku ---------------------------
def read_price_csv(fpath: Path) -> Optional[pd.DataFrame]:
    """
    Lue yfinance-tyylinen CSV robustisti:
    - huomioi utf-8-sig (BOM), cp1252
    - erotin autodetect + fallback: ',', ';', '\t'
    - sallii otsikkojen variaatiot (Date/Datetime/Timestamp, Adj Close/adj_close/AdjClose)
    - jos Datea ei löydy nimellä, etsii parhaan päivämääräsarakkeen automaattisesti
    """
    import re

    tries = [
        dict(encoding="utf-8-sig", sep=None, engine="python"),
        dict(encoding="utf-8",     sep=None, engine="python"),
        dict(encoding="cp1252",    sep=None, engine="python"),
        dict(encoding="utf-8-sig", sep=","),
        dict(encoding="utf-8-sig", sep=";"),
        dict(encoding="utf-8-sig", sep="\t"),
    ]

    last_err = None
    df = None
    for kw in tries:
        try:
            df = pd.read_csv(fpath, **kw)
            if df is not None and df.shape[1] >= 2:
                break
        except Exception as e:
            last_err = e
            df = None
    if df is None or df.empty:
        if DEBUG:
            print(f"[DEBUG] CSV read failed: {fpath.name} ({last_err})")
        return None

    def norm(s: str) -> str:
        return re.sub(r"[^a-z]", "", str(s).strip().lower())

    df.columns = [str(c).strip() for c in df.columns]
    lowmap = {norm(c): c for c in df.columns}

    date_col  = lowmap.get("date") or lowmap.get("datetime") or lowmap.get("timestamp")
    open_col  = lowmap.get("open")
    high_col  = lowmap.get("high")
    low_col   = lowmap.get("low")
    close_col = lowmap.get("close") or lowmap.get("adjclose")

    # päivämäärä: jos ei nimellä, valitse paras päivämääräsarake automaattisesti
    if date_col is None:
        best, best_ok = None, 0
        for c in df.columns:
            ser = pd.to_datetime(df[c], errors="coerce", utc=True)
            ok = int(ser.notna().sum())
            if ok > best_ok:
                best_ok = ok
                best = ser.dt.tz_convert(None)
        if best is not None and best_ok > 0:
            ser_date = best
        else:
            if DEBUG:
                print(f"[DEBUG] No date-like column in {fpath.name}; cols={list(df.columns)[:8]}")
            return None
    else:
        ser_date = pd.to_datetime(df[lowmap.get(norm(date_col), date_col)],
                                  errors="coerce", utc=True).dt.tz_convert(None)

    # varmista OHLC
    if open_col is None or high_col is None or low_col is None or close_col is None:
        open_col  = open_col  or lowmap.get("open")
        high_col  = high_col  or lowmap.get("high")
        low_col   = low_col   or lowmap.get("low")
        close_col = close_col or lowmap.get("adjclose") or lowmap.get("close")

    if any(c is None for c in [open_col, high_col, low_col, close_col]):
        if DEBUG:
            print(f"[DEBUG] Missing OHLC in {fpath.name}; cols={list(df.columns)[:8]}")
        return None

    out = pd.DataFrame({
        "Date":  ser_date,
        "Open":  pd.to_numeric(df[open_col],  errors="coerce"),
        "High":  pd.to_numeric(df[high_col],  errors="coerce"),
        "Low":   pd.to_numeric(df[low_col],   errors="coerce"),
        "Close": pd.to_numeric(df[close_col], errors="coerce"),
    }).dropna()

    if out.empty:
        if DEBUG:
            print(f"[DEBUG] Empty after cleanup: {fpath.name}")
        return None

    out = out.sort_values("Date")
    out = out[~out["Date"].duplicated(keep="last")].reset_index(drop=True)
    return out


# ----------------------------- RSI (Wilder) ----------------------------------
def rsi_wilder(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).astype(float)
    loss = (-delta.where(delta < 0, 0.0)).astype(float)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


# ----------------------------- aggregaatit -----------------------------------
def load_aggregates(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg_dir = run_dir / "aggregates"
    dt5 = None
    upc = None
    for name in ["daily_top5_calendar.csv", "daily_top5_calendar*.csv"]:
        matches = sorted(agg_dir.glob(name))
        if matches:
            dt5 = pd.read_csv(matches[-1])
            break
    for name in ["upcoming_windows_next_45d.csv", "upcoming_windows_next_*d.csv"]:
        matches = sorted(agg_dir.glob(name))
        if matches:
            upc = pd.read_csv(matches[-1])
            break
    if dt5 is None:
        dt5 = pd.DataFrame()
    if upc is None:
        upc = pd.DataFrame()
    if DEBUG:
        print(f"[DEBUG] Aggregates: dt5 rows={len(dt5)}, upcoming rows={len(upc)}")
    return dt5, upc


def parse_date_col(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return pd.to_datetime(df[low[cand]], errors="coerce", utc=True).dt.tz_convert(None)
    # automaattinen haku: valitse paras päivämääräsarake
    best = None; best_ok = 0
    for c in df.columns:
        ser = pd.to_datetime(df[c], errors="coerce", utc=True)
        ok = int(ser.notna().sum())
        if ok > best_ok:
            best_ok = ok
            best = ser.dt.tz_convert(None)
    return best


def build_allowed_dates_map(run_dir: Path, near_days: int = 5) -> Dict[str, Set[pd.Timestamp]]:
    """Palauttaa: { TICKER -> set(allowed_dates) } yhdistämällä daily_top5 ja upcoming."""
    dt5, upc = load_aggregates(run_dir)
    allowed: Dict[str, Set[pd.Timestamp]] = {}

    # A) daily_top5_calendar: yksittäiset päivät
    if not dt5.empty:
        tick_col = None
        for c in dt5.columns:
            if c.lower() in ("ticker","symbol","code","tkr"):
                tick_col = c; break
        if tick_col is None:
            tick_col = dt5.columns[0]
        dser = parse_date_col(dt5, ["date","day","session","trade_date"])
        if dser is not None:
            tmp = dt5.assign(__date=dser).dropna(subset=["__date"])
            for t, g in tmp.groupby(tick_col):
                tkr = str(t).upper().replace(".","-").strip()
                allowed.setdefault(tkr, set()).update(set(g["__date"].dt.normalize()))

    # B) upcoming_windows_next_*d: start-end (laajenna ± near_days)
    if not upc.empty:
        tick_col = None
        for c in upc.columns:
            if c.lower() in ("ticker","symbol","code","tkr"):
                tick_col = c; break
        if tick_col is None:
            tick_col = upc.columns[0]

        start = parse_date_col(upc, ["start","window_start","from","begin","from_date","start_date"])
        end   = parse_date_col(upc, ["end","window_end","to","finish","to_date","end_date"])

        if start is None or end is None:
            date_cols = []
            for c in upc.columns:
                ser = parse_date_col(upc[[c]].rename(columns={c:"x"}), ["x"])
                if ser is not None and ser.notna().sum() > 0:
                    date_cols.append((c, ser.notna().sum(), ser))
            date_cols.sort(key=lambda x: x[1], reverse=True)
            if len(date_cols) >= 2:
                start, end = date_cols[0][2], date_cols[1][2]
            elif len(date_cols) == 1:
                start = end = date_cols[0][2]

        if start is not None:
            if end is None:
                end = start
            tmp = upc.assign(__s=start, __e=end).dropna(subset=["__s","__e"])
            for t, g in tmp.groupby(tick_col):
                tkr = str(t).upper().replace(".","-").strip()
                bag: Set[pd.Timestamp] = allowed.setdefault(tkr, set())
                for _, row in g.iterrows():
                    s = pd.Timestamp(row["__s"]).normalize() - pd.Timedelta(days=near_days)
                    e = pd.Timestamp(row["__e"]).normalize() + pd.Timedelta(days=near_days)
                    rng = pd.date_range(s, e, freq="D")
                    bag.update(set(pd.to_datetime(rng).normalize()))
    if DEBUG:
        print(f"[DEBUG] Allowed map size: {len(allowed)} tickers")
    return allowed


# ----------------------------- Reliability (optional) ------------------------
def load_reliability(min_threshold: float = 0.0) -> Dict[Tuple[str,int], float]:
    path = VINTAGE_DIR / "RELIABILITY_by_year.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    tcol = next((low[k] for k in ("ticker","symbol","code") if k in low), df.columns[0])
    ycol = low.get("year") or low.get("yyyy")
    if ycol is None:
        for c in df.columns:
            if "date" in c.lower():
                df["__year"] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_convert(None).dt.year
                ycol = "__year"; break
    rcol = next((low[k] for k in ("reliability","score","corr","agreement") if k in low), None)
    if rcol is None:
        return {}
    out: Dict[Tuple[str,int], float] = {}
    for _, row in df.iterrows():
        t = str(row[tcol]).upper().replace(".","-").strip()
        y = int(row[ycol]) if pd.notna(row[ycol]) else None
        r = float(row[rcol]) if pd.notna(row[rcol]) else None
        if t and y and r is not None and r >= min_threshold:
            out[(t, y)] = r
    return out


# ----------------------------- signaalit -------------------------------------
def compute_signals(df: pd.DataFrame,
                    breakout_lookback: int,
                    rsi_len: int,
                    rsi_thresh: float,
                    allowed_dates: Set[pd.Timestamp],
                    use_breakout: bool,
                    use_rsi: bool) -> pd.DataFrame:
    df = df.copy()
    df["RSI"] = rsi_wilder(df["Close"], rsi_len) if use_rsi else np.nan
    if use_breakout:
        df["MAX_N"] = df["Close"].shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).max()
        df["SIG_BRK"] = (df["Close"] > df["MAX_N"]).fillna(False)
    else:
        df["SIG_BRK"] = False
    if use_rsi:
        df["SIG_RSI"] = (df["RSI"] < rsi_thresh).fillna(False)
    else:
        df["SIG_RSI"] = False

    # seasonality-rajaus
    if allowed_dates:
        df["ALLOW"] = df["Date"].dt.normalize().isin(allowed_dates)
    else:
        df["ALLOW"] = True
    df["SIG_TODAY"] = df["ALLOW"] & (df["SIG_BRK"] | df["SIG_RSI"])
    df["ENTRY_NEXT"] = df["SIG_TODAY"].shift(1).fillna(False)
    return df


# ----------------------------- backtest engine -------------------------------
class Position:
    __slots__ = ("ticker","entry_date","entry_price","shares","sl","tp","reason","bars_held")
    def __init__(self, ticker, entry_date, entry_price, shares, sl, tp):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = float(entry_price)
        self.shares = int(shares)
        self.sl = float(sl)
        self.tp = float(tp)
        self.reason = ""
        self.bars_held = 0


def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = series/roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0


def cagr(start_val: float, end_val: float, days: int) -> float:
    if start_val <= 0 or end_val <= 0 or days <= 0:
        return 0.0
    years = days / 365.0
    return (end_val/start_val)**(1/years) - 1.0


def run_backtest(run_dir: Path,
                 use_breakout: bool,
                 use_rsi: bool,
                 breakout_lookback: int,
                 rsi_len: int,
                 rsi_thresh: float,
                 near_days: int,
                 require_seasonality: bool,
                 use_reliability: bool,
                 reliability_min: float,
                 sl_pct: float,
                 tp_pct: float,
                 max_positions: int,
                 hold_days: int,
                 cost_pct: float,
                 start_capital: float,
                 debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # 1) allowed dates per ticker
    allowed_map = build_allowed_dates_map(run_dir, near_days=near_days)
    if debug:
        print(f"[DEBUG] Allowed-dates map for {len(allowed_map)} tickers.")
    if require_seasonality and not any(len(v)>0 for v in allowed_map.values()):
        print("[ERROR] Seasonality-kalenteria ei löytynyt/parsittu. Aja aggregate-askel tai anna --require-seasonality 0.")
        sys.exit(1)

    # 2) reliability (optional)
    reliab = load_reliability(min_threshold=reliability_min) if use_reliability else {}

    # 3) hinnat
    price_files = {p.stem.upper(): p for p in PRICE_CACHE.glob("*.csv")}
    tickers = sorted(price_files.keys())
    if debug:
        print(f"[DEBUG] Price files: {len(tickers)} found in {PRICE_CACHE}")

    # 4) signaalit
    per_ticker = {}
    for t in tickers:
        f = price_files[t]
        df = read_price_csv(f)
        if df is None or df.empty:
            if debug:
                print(f"[DEBUG] Skip (unreadable/empty): {t}")
            continue
        allow = allowed_map.get(t, set())
        if require_seasonality and not allow:
            continue
        sigdf = compute_signals(df, breakout_lookback, rsi_len, rsi_thresh,
                                allowed_dates=allow,
                                use_breakout=use_breakout,
                                use_rsi=use_rsi)
        sigdf["Ticker"] = t
        per_ticker[t] = sigdf

    if not per_ticker:
        print("[ERROR] Ei yhtään kelvollista tikkeriä signaaleilla. Tarkista hintojen CSV-formaatti ja aggregaatit.")
        sys.exit(1)
    if debug:
        print(f"[DEBUG] Tickers with signals: {len(per_ticker)}")

    # 5) aikajana
    all_dates = sorted(set(pd.concat([df["Date"] for df in per_ticker.values()], ignore_index=True).dt.normalize()))
    last_close: Dict[str, float] = {}

    # 6) portfolio-state
    cash = float(start_capital)
    positions: Dict[str, Position] = {}
    trades: List[Dict] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    alloc = 1.0 / max(1, max_positions)

    # 7) loop
    for i, d in enumerate(all_dates):
        today_rows: Dict[str, pd.Series] = {}
        for t, df in per_ticker.items():
            mask = df["Date"].dt.normalize() == d
            if mask.any():
                row = df.loc[mask].iloc[-1]
                today_rows[t] = row
                last_close[t] = float(row["Close"])

        # exits
        to_close = []
        for t, pos in list(positions.items()):
            if t not in today_rows:
                continue
            row = today_rows[t]
            hi = float(row["High"]); lo = float(row["Low"]); cl = float(row["Close"])
            pos.bars_held += 1
            exit_price = None; reason = ""
            if lo <= pos.sl:
                exit_price = pos.sl; reason = "STOP"
            elif hi >= pos.tp:
                exit_price = pos.tp; reason = "TP"
            elif pos.bars_held >= hold_days:
                exit_price = cl; reason = "TIME"
            if exit_price is not None:
                proceeds = exit_price * pos.shares
                fee = proceeds * cost_pct
                cash += proceeds - fee
                pnl = (exit_price - pos.entry_price) * pos.shares - (pos.entry_price * pos.shares * cost_pct)
                trades.append({
                    "ticker": t,
                    "entry_date": pos.entry_date.date(),
                    "entry_price": pos.entry_price,
                    "exit_date": d.date(),
                    "exit_price": exit_price,
                    "shares": pos.shares,
                    "pnl": pnl,
                    "pnl_pct": (exit_price/pos.entry_price - 1.0),
                    "reason": reason,
                    "bars_held": pos.bars_held,
                })
                to_close.append(t)
        for t in to_close:
            positions.pop(t, None)

        # entries (ENTRY_NEXT True today)
        can_open = max(0, max_positions - len(positions))
        if can_open > 0:
            cands = []
            for t, row in today_rows.items():
                if row.get("ENTRY_NEXT", False) and t not in positions:
                    ok = True
                    if use_reliability:
                        yr = pd.Timestamp(row["Date"]).year
                        r = reliab.get((t, yr), None)
                        if r is None or r < reliability_min:
                            ok = False
                    if ok:
                        cands.append((t, row))
            cands.sort(key=lambda x: x[0])  # deterministinen prioriteetti
            for t, row in cands[:can_open]:
                op = float(row["Open"])
                if not np.isfinite(op) or op <= 0:
                    continue
                target_cash = cash * alloc
                shares = int(target_cash // op)
                if shares <= 0:
                    continue
                cost = op * shares
                fee = cost * cost_pct
                cash -= (cost + fee)
                sl = op * (1.0 - sl_pct)
                tp = op * (1.0 + tp_pct)
                positions[t] = Position(t, pd.Timestamp(row["Date"]), op, shares, sl, tp)

        # mark-to-market
        port_val = cash
        for t, pos in positions.items():
            price = last_close.get(t, pos.entry_price)
            port_val += price * pos.shares
        equity_curve.append((d, port_val))

    # close leftovers
    if positions:
        last_date = all_dates[-1]
        for t, pos in list(positions.items()):
            last_price = last_close.get(t, pos.entry_price)
            proceeds = last_price * pos.shares
            fee = proceeds * cost_pct
            pnl = (last_price - pos.entry_price) * pos.shares - (pos.entry_price * pos.shares * cost_pct)
            trades.append({
                "ticker": t,
                "entry_date": pos.entry_date.date(),
                "entry_price": pos.entry_price,
                "exit_date": last_date.date(),
                "exit_price": last_price,
                "shares": pos.shares,
                "pnl": pnl,
                "pnl_pct": (last_price/pos.entry_price - 1.0),
                "reason": "EOD_CLOSE",
                "bars_held": pos.bars_held,
            })
            cash += proceeds - fee
            positions.pop(t, None)

    eq_df = pd.DataFrame(equity_curve, columns=["date","equity"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])

    start_val = float(eq_df["equity"].iloc[0]) if len(eq_df) else start_capital
    end_val   = float(eq_df["equity"].iloc[-1]) if len(eq_df) else start_capital
    days      = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days if len(eq_df) > 1 else 0
    maxdd     = max_drawdown(eq_df["equity"])
    cg        = cagr(start_val, end_val, days)

    tr_df = pd.DataFrame(trades)
    n_trades = len(tr_df)
    wins = float((tr_df["pnl"] > 0).sum()) if n_trades else 0.0
    winrate = wins / n_trades if n_trades else 0.0
    avg_gain = float(tr_df.loc[tr_df["pnl"]>0,"pnl"].mean()) if n_trades else 0.0
    avg_loss = float(-tr_df.loc[tr_df["pnl"]<0,"pnl"].mean()) if n_trades else 0.0
    pf = (avg_gain/avg_loss) if (avg_gain>0 and avg_loss>0) else np.nan

    # karkea Sharpe (päivälog-ret)
    if len(eq_df) > 1:
        rets = np.log(eq_df["equity"]).diff().fillna(0.0)
        sharpe = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else np.nan
    else:
        sharpe = np.nan

    res = pd.DataFrame([{
        "start_capital": start_capital,
        "end_capital": end_val,
        "CAGR": cg,
        "MaxDD": maxdd,
        "Trades": n_trades,
        "WinRate": winrate,
        "AvgGain": avg_gain,
        "AvgLoss": -avg_loss,
        "ProfitFactor": pf,
        "SharpeApprox": sharpe,
        "Begin": eq_df["date"].iloc[0].date() if len(eq_df) else None,
        "End":   eq_df["date"].iloc[-1].date() if len(eq_df) else None,
    }])

    return res, tr_df, eq_df


# ----------------------------- CLI -------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Seasonality Backtest v1")
    ap.add_argument("--run-dir", default="", help="Run folder name under seasonality_reports\\runs (or absolute)")
    ap.add_argument("--near-days", type=int, default=5, help="In/near seasonality window days")
    ap.add_argument("--use-breakout", type=int, default=1, help="Use breakout signal (1/0)")
    ap.add_argument("--use-rsi", type=int, default=1, help="Use RSI(2) signal (1/0)")
    ap.add_argument("--breakout-lookback", type=int, default=20, help="Breakout lookback days")
    ap.add_argument("--rsi-len", type=int, default=2, help="RSI length")
    ap.add_argument("--rsi-thresh", type=float, default=10.0, help="RSI threshold for entry")
    ap.add_argument("--require-seasonality", type=int, default=1, help="Require seasonality calendars (1/0)")
    ap.add_argument("--use-reliability", type=int, default=0, help="Filter by vintage reliability (1/0)")
    ap.add_argument("--reliability-min", type=float, default=0.0, help="Min reliability if enabled")
    ap.add_argument("--sl-pct", type=float, default=0.10, help="Stop loss percent (0.10 = 10%)")
    ap.add_argument("--tp-pct", type=float, default=0.15, help="Take profit percent (0.15 = 15%)")
    ap.add_argument("--max-positions", type=int, default=5, help="Max concurrent positions")
    ap.add_argument("--hold-days", type=int, default=20, help="Time exit after N bars")
    ap.add_argument("--cost-pct", type=float, default=0.002, help="Side cost (0.002 = 0.2%) per entry/exit")
    ap.add_argument("--start-capital", type=float, default=100000.0, help="Starting capital")
    ap.add_argument("--debug", type=int, default=0, help="Print diagnostics")
    args = ap.parse_args()

    # aseta globaali DEBUG
    global DEBUG
    DEBUG = bool(args.debug)

    # resolve run dir
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = RUNS_DIR / args.run_dir
    else:
        run_dir = latest_run_dir() or REPORT_ROOT
    if not run_dir.exists():
        print(f("[ERROR] Run directory not found: {run_dir}"))
        sys.exit(1)

    out_dir = run_dir / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    if DEBUG:
        print(f"[DEBUG] Using run_dir: {run_dir}")
        print(f"[DEBUG] Price cache: {PRICE_CACHE.resolve()}")

    res, tr, eq = run_backtest(
        run_dir=run_dir,
        use_breakout=bool(args.use_breakout),
        use_rsi=bool(args.use_rsi),
        breakout_lookback=args.breakout_lookback,
        rsi_len=args.rsi_len,
        rsi_thresh=args.rsi_thresh,
        near_days=args.near_days,
        require_seasonality=bool(args.require_seasonality),
        use_reliability=bool(args.use_reliability),
        reliability_min=args.reliability_min,
        sl_pct=args.sl_pct,
        tp_pct=args.tp_pct,
        max_positions=args.max_positions,
        hold_days=args.hold_days,
        cost_pct=args.cost_pct,
        start_capital=args.start_capital,
        debug=DEBUG,
    )

    # kirjoita ulos
    res.to_csv(out_dir / "results_summary.csv", index=False)
    tr.to_csv(out_dir / "trades.csv", index=False)
    eq.to_csv(out_dir / "equity_curve.csv", index=False)

    print(f"[OK] Results saved to: {out_dir}")
    print(" - results_summary.csv")
    print(" - trades.csv")
    print(" - equity_curve.csv")


if __name__ == "__main__":
    main()
