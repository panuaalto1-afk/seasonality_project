
# build_prices_from_indexes.py (reuse-first for cross-asset proxies)
# ------------------------------------------------------------------
# Esim:
# python build_prices_from_indexes.py ^
#   --run_root "C:\...\runs\2025-10-06_200727" ^
#   --source_cache "C:\...\runs\2025-10-04_0903\price_cache" ^
#   --reuse_first 1 --no_download
# ------------------------------------------------------------------

import os, time, argparse, datetime as dt
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

CROSS_ASSET = ["SHY","IEF","TLT","HYG","LQD","UUP","USO","GLD","CPER","TIP","RINF","SPY","QQQ","IWM","^SPX","^VIX"]
COPPER_FUT = "HG=F"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _atomic_copy_replace(src: Path, dst: Path, retries=6, base_sleep=0.35) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for i in range(retries):
        try:
            if tmp.exists():
                try: tmp.unlink()
                except PermissionError: time.sleep(base_sleep); continue
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            return "copy"
        except PermissionError:
            time.sleep(base_sleep * (i+1))
        except Exception:
            time.sleep(base_sleep * (i+1))
    if dst.exists():
        try: dst.unlink()
        except PermissionError: pass
    shutil.copy2(src, dst); return "copy"

def _hardlink_or_copy(src: Path, dst: Path) -> str:
    try:
        if src.drive.upper() == dst.drive.upper():
            if dst.exists(): return _atomic_copy_replace(src, dst)
            os.link(src, dst)
            return "link"
    except PermissionError:
        pass
    return _atomic_copy_replace(src, dst)

def download_prices(ticker: str, start: str, end: str, tries=3, sleep=0.35) -> pd.DataFrame | None:
    last_err=None
    for k in range(tries):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty: raise RuntimeError("No data")
            df = df.reset_index().rename(columns={"Date":"Date"})
            if "Adj Close" not in df.columns: df["Adj Close"]=df["Close"]
            return df[["Date","Open","High","Low","Close","Adj Close","Volume"]]
        except Exception as e:
            last_err=e; time.sleep(min(2.0, sleep*(2**k)))
    print(f"[WARN] download failed for {ticker}: {last_err}")
    return None

def main():
    ap = argparse.ArgumentParser(description="Täyttää runin price_cache cross-asset proxyilla.")
    ap.add_argument("--reports_root", default=str(REPORTS_ROOT_DEFAULT))
    ap.add_argument("--run_root", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--source_cache", default="")
    ap.add_argument("--reuse_first", type=int, default=1)
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.35)
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
    end   = args.end   or today.strftime("%Y-%m-%d")

    print(f"Kohde: {price_dir}")
    if source_cache: print(f"Lähde-cache: {source_cache}")
    print(f"Universumi: {len(CROSS_ASSET)} proxya | reuse_first={args.reuse_first} no_download={args.no_download}")

    linked=copied=downloaded=skipped=0
    for i,t in enumerate(CROSS_ASSET,1):
        dst = price_dir / f"{t}.csv"
        if dst.exists() and not args.overwrite:
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
            if (not reused) and (not args.no_download):
                what = t if t!="CPER" else COPPER_FUT
                df = download_prices(what, start, end, sleep=args.sleep)
                if df is not None:
                    df.to_csv(dst, index=False); downloaded += 1
        if i % 5 == 0 or i==len(CROSS_ASSET):
            print(f"[{i}/{len(CROSS_ASSET)}] L:{linked} C:{copied} DL:{downloaded} SK:{skipped}")

    print("\n=== SUMMARY ===")
    print(f"linked:{linked} copied:{copied} downloaded:{downloaded} skipped:{skipped}")

if __name__ == "__main__":
    main()