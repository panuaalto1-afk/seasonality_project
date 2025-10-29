# build_prices_from_constituents.py (reuse-first, Windows-friendly)
# -----------------------------------------------------------------------------
# Täyttää <RUN_ROOT>\price_cache osake-CSV:illä kierrättämällä olemassa olevan
# cache:n (hardlink/copy) ja tarvittaessa lataamalla vain puuttuvat/hännät.
#
# Esim:
# python build_prices_from_constituents.py ^
#   --const "C:\...\constituents_raw.csv" ^
#   --run_root "C:\...\runs\2025-10-06_200727" ^
#   --source_cache "C:\...\runs\2025-10-04_0903\price_cache" ^
#   --reuse_first 1 --topoff 1 --no_download
# -----------------------------------------------------------------------------

import os, sys, time, argparse, datetime as dt
from pathlib import Path
import shutil
import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("Asenna:\n  py -m pip install --upgrade yfinance pandas numpy lxml")
    raise

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT_DEFAULT = PROJECT_ROOT / "seasonality_reports"

# ---------- helpers ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def to_yahoo_symbol(t: str) -> str: return str(t).strip().upper().replace(".", "-")
def clean_ticker(t: str) -> str:
    t = str(t).strip().upper()
    return "" if t in {"", "NAN", "NONE", "N/A"} else t
def uniq(seq):
    seen=set(); out=[]
    for x in seq:
        if x and x not in seen: seen.add(x); out.append(x)
    return out

def read_universe(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Universumi CSV puuttuu: {csv_path}")
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol") or list(df.columns)[0]
    tickers = [clean_ticker(x) for x in df[tcol].dropna().astype(str)]
    blacklist = {"SPY","QQQ","IWM","XLU","XLK","XLY","XLE","XLF","XLI","XLP","XLV","XLC","XLB",
                 "SHY","IEF","TLT","HYG","LQD","UUP","USO","GLD","CPER","TIP","RINF","VIX","^VIX"}
    tickers = [t for t in tickers if t not in blacklist]
    return uniq([to_yahoo_symbol(t) for t in tickers])

def read_csv_safe(p: Path) -> pd.DataFrame | None:
    if not p.exists(): return None
    try:
        df = pd.read_csv(p)
        dcol = "Date"
        if "Date" not in df.columns:
            cols = {c.lower(): c for c in df.columns}
            dcol = cols.get("date") or cols.get("timestamp") or df.columns[0]
            df = df.rename(columns={dcol: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception as e:
        print(f"[WARN] read failed {p.name}: {e}")
        return None

def download_prices(ticker: str, start: str, end: str, tries=3, sleep=0.35) -> pd.DataFrame | None:
    last_err=None
    for k in range(tries):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty: raise RuntimeError("No data")
            df = df.reset_index().rename(columns={"Date":"Date"})
            for col in ["Open","High","Low","Close","Adj Close","Volume"]:
                if col not in df.columns:
                    if col=="Adj Close" and "Close" in df.columns:
                        df["Adj Close"]=df["Close"]
                    else:
                        raise RuntimeError(f"Missing column: {col}")
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            return df[["Date","Open","High","Low","Close","Adj Close","Volume"]]
        except Exception as e:
            last_err=e; time.sleep(min(2.0, sleep*(2**k)))
    print(f"[WARN] download failed for {ticker}: {last_err}")
    return None

# ---- robust replace (WinError 32 protected) ----
def _atomic_copy_replace(src: Path, dst: Path, retries=6, base_sleep=0.35) -> str:
    """Kopioi src -> dst atomisesti (tmp + os.replace). Yrittää uudelleen, jos kohde on lukittu."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for i in range(retries):
        try:
            if tmp.exists():
                try: tmp.unlink()
                except PermissionError: time.sleep(base_sleep); continue
            shutil.copy2(src, tmp)
            # Atominen vaihto myös jos dst on olemassa
            os.replace(tmp, dst)
            return "copy"
        except PermissionError:
            time.sleep(base_sleep * (i+1))
        except Exception:
            time.sleep(base_sleep * (i+1))
    # viimeinen yritys: poista kohde ja yritä suora copy2
    try:
        if dst.exists():
            try: dst.unlink()
            except PermissionError: pass
        shutil.copy2(src, dst)
        return "copy"
    except Exception as e:
        raise e

def _hardlink_or_copy(src: Path, dst: Path) -> str:
    # yritä hardlink samalle levylle; muutoin käytä robust copy + replace
    try:
        if src.drive.upper() == dst.drive.upper():
            # jos kohde on olemassa, vaihda atomisesti
            if dst.exists():
                return _atomic_copy_replace(src, dst)
            os.link(src, dst)
            return "link"
    except PermissionError:
        pass
    return _atomic_copy_replace(src, dst)

def topoff_if_needed(ticker: str, target_csv: Path, end_date: dt.date, allow_download: bool, sleep=0.35) -> str:
    df = read_csv_safe(target_csv)
    if df is None or df.empty: return "missing"
    last_dt = pd.to_datetime(df["Date"]).max().date()
    if last_dt >= end_date: return "ok"
    if not allow_download: return f"stale_need_topoff({last_dt}->{end_date})"
    start = (last_dt + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    tail  = download_prices(ticker, start=start, end=end_date.strftime("%Y-%m-%d"), sleep=sleep)
    if tail is None or tail.empty: return "stale_no_data"
    merged = pd.concat([df, tail], ignore_index=True).drop_duplicates(subset=["Date"]).sort_values("Date")
    _atomic_copy_replace(src=Path(target_csv), dst=Path(target_csv))  # no-op, varmistaa polun
    merged.to_csv(target_csv, index=False)
    return f"topped_{len(tail)}"

def infer_market_end_date(source_cache: Path | None, default: dt.date) -> dt.date:
    if source_cache:
        spy = source_cache / "SPY.csv"
        df = read_csv_safe(spy)
        if df is not None and not df.empty:
            return pd.to_datetime(df["Date"]).max().date()
    return default

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Täytä runin price_cache reuse-first -periaatteella.")
    ap.add_argument("--const", required=True)
    ap.add_argument("--reports_root", default=str(REPORTS_ROOT_DEFAULT))
    ap.add_argument("--run_root", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--source_cache", default="")
    ap.add_argument("--reuse_first", type=int, default=1)
    ap.add_argument("--topoff", type=int, default=1)
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.30)
    args = ap.parse_args()

    reports_root = Path(args.reports_root)
    if args.outdir:
        price_dir = Path(args.outdir)
    else:
        run_root = Path(args.run_root) if args.run_root else None
        price_dir = (run_root / "price_cache") if run_root else (reports_root / "price_cache")
    ensure_dir(price_dir)

    source_cache = Path(args.source_cache) if args.source_cache else None
    if source_cache and not source_cache.exists():
        print(f"[WARN] source_cache ei löydy: {source_cache}"); source_cache=None

    today = dt.date.today()
    start = args.start or (today - dt.timedelta(days=365*args.years)).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(args.end).date() if args.end else today
    end_date = infer_market_end_date(source_cache, end_date)

    tickers = read_universe(Path(args.const))
    print(f"Kohde: {price_dir}")
    if source_cache: print(f"Lähde-cache: {source_cache}")
    print(f"Universumi: {len(tickers)} tickers | reuse_first={args.reuse_first} topoff={args.topoff} no_download={args.no_download}")
    print(f"End date used: {end_date}")

    linked=copied=topped=downloaded=skipped=stale=missing=0
    total=len(tickers)

    for i,t in enumerate(tickers,1):
        dst = price_dir / f"{t}.csv"

        # jos olemassa eikä overwrite → mahdollinen topoff
        if dst.exists() and not args.overwrite:
            if args.topoff:
                st = topoff_if_needed(t, dst, end_date, allow_download=(not args.no_download), sleep=args.sleep)
                if st.startswith("topped_"): topped += 1
                elif st.startswith("stale_need_topoff"): stale += 1
                else: skipped += 1
            else:
                skipped += 1
        else:
            reused=False
            if args.reuse_first and source_cache:
                src = source_cache / f"{t}.csv"
                if src.exists():
                    mode = _hardlink_or_copy(src, dst)
                    reused=True
                    if mode=="link": linked+=1
                    else: copied+=1
                    if args.topoff:
                        st = topoff_if_needed(t, dst, end_date, allow_download=(not args.no_download), sleep=args.sleep)
                        if st.startswith("topped_"): topped += 1
                        elif st.startswith("stale_need_topoff"): stale += 1
            if (not reused) and (not args.no_download):
                df = download_prices(t, start=start, end=end_date.strftime("%Y-%m-%d"), sleep=args.sleep)
                if df is None:
                    missing += 1
                else:
                    df.to_csv(dst, index=False); downloaded += 1
            elif (not reused) and args.no_download:
                missing += 1

        if i % 100 == 0 or i == total:
            print(f"[{i}/{total}] L:{linked} C:{copied} T:{topped} DL:{downloaded} SK:{skipped} STALE:{stale} MISS:{missing}")

    print("\n=== SUMMARY ===")
    print(f"linked:{linked} copied:{copied} topped:{topped} downloaded:{downloaded} skipped:{skipped} stale:{stale} missing:{missing}")

if __name__ == "__main__":
    main()
