# fetch_prices_all_us.py
# ------------------------------------------------------------
# Lataa KAIKKIEN USA-osakkeiden päivädata yfinance:sta ja tallentaa:
#   seasonality_reports/runs/<RUN>/price_cache/<TICKER>.csv
# Ilman suodatuksia. Universumi NASDAQ Trader -listoista.
# ------------------------------------------------------------
import argparse, io, sys, time
from pathlib import Path
import requests
import pandas as pd
import yfinance as yf

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

REPORT_ROOT = Path("seasonality_reports")
RUNS_DIR    = REPORT_ROOT / "runs"

ENDPOINTS = {
    # Yritä NÄMÄ JÄRJESTYKSESSÄ (www- ja ftp-alidomainit, http/https)
    "nasdaqlisted": [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    ],
    "otherlisted": [
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "http://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ],
}

def log(s: str): print(s, flush=True)
def clean_ticker_yahoo(sym: str) -> str: return str(sym).strip().upper().replace(".", "-")

def fetch_txt_multi(urls: list[str], timeout: int = 20) -> str:
    """Kokeile useita osoitteita + retryt. Palauttaa tekstin tai nostaa poikkeuksen."""
    last_err = None
    for url in urls:
        for attempt in range(3):
            try:
                r = requests.get(
                    url,
                    headers={"User-Agent": UA, "Accept": "text/plain"},
                    timeout=timeout,
                )
                if r.status_code == 200 and "|" in r.text:
                    return r.text
                last_err = RuntimeError(f"HTTP {r.status_code} from {url}")
            except Exception as e:
                last_err = e
            # pientä backoffia
            time.sleep(1.0 + attempt * 1.5)
    raise RuntimeError(f"All endpoints failed for {urls[-1].split('/')[-1]} ({last_err})")

def load_us_tickers() -> list[str]:
    try:
        txt1 = fetch_txt_multi(ENDPOINTS["nasdaqlisted"])
        txt2 = fetch_txt_multi(ENDPOINTS["otherlisted"])
    except Exception as e:
        log(f"[ERROR] Symbollistojen haku epäonnistui: {e}")
        raise

    def parse_nasdaq(text: str) -> pd.DataFrame:
        lines = [ln for ln in text.splitlines() if "|" in ln]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
        df = df[~df["Symbol"].astype(str).str.contains("File Creation Time", na=False)]
        if "ETF" in df.columns: df = df[df["ETF"].astype(str).str.upper().ne("Y")]
        if "Test Issue" in df.columns: df = df[df["Test Issue"].astype(str).str.upper().ne("Y")]
        return df[["Symbol"]].dropna().assign(Symbol=lambda s: s["Symbol"].astype(str).str.strip())

    def parse_other(text: str) -> pd.DataFrame:
        lines = [ln for ln in text.splitlines() if "|" in ln]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
        df = df[~df["ACT Symbol"].astype(str).str.contains("File Creation Time", na=False)]
        if "ETF" in df.columns: df = df[df["ETF"].astype(str).str.upper().ne("Y")]
        if "Test Issue" in df.columns: df = df[df["Test Issue"].astype(str).str.upper().ne("Y")]
        df = df[["ACT Symbol"]].rename(columns={"ACT Symbol": "Symbol"}).dropna()
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        return df

    df1 = parse_nasdaq(txt1)
    df2 = parse_other(txt2)
    syms = pd.concat([df1["Symbol"], df2["Symbol"]], ignore_index=True).dropna().unique().tolist()
    syms = sorted({clean_ticker_yahoo(s) for s in syms if s and s.isascii()})
    return syms

def ensure_dirs(run_name: str) -> Path:
    run_dir = RUNS_DIR / run_name
    dest = run_dir / "price_cache"
    dest.mkdir(parents=True, exist_ok=True)
    return dest

def save_prices_for_chunk(chunk_syms: list[str], dest_dir: Path, overwrite: bool, sleep: float) -> tuple[int,int]:
    # Jätä valmiit pois jos ei overwrite
    syms = [t for t in chunk_syms if overwrite or not (dest_dir / f"{t}.csv").exists()]
    if not syms: return (0, 0)

    ok = 0; fail = 0
    try:
        data = yf.download(
            tickers=syms,
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            for t in syms:
                try:
                    dft = data[t].dropna()
                    if dft.empty: fail += 1; continue
                    dft.reset_index().to_csv(dest_dir / f"{t}.csv", index=False)
                    ok += 1
                except Exception:
                    fail += 1
        else:
            dft = data.dropna()
            if not dft.empty:
                dft.reset_index().to_csv(dest_dir / f"{syms[0]}.csv", index=False)
                ok += 1
            else:
                fail += 1
    except Exception:
        # fallback yksittäin
        for t in syms:
            try:
                d = yf.download(t, period="max", interval="1d", auto_adjust=False,
                                actions=False, progress=False)
                if d is None or d.empty:
                    fail += 1; continue
                d.reset_index().to_csv(dest_dir / f"{t}.csv", index=False)
                ok += 1
                time.sleep(sleep)
            except Exception:
                fail += 1
    return (ok, fail)

def main():
    ap = argparse.ArgumentParser(description="Fetch ALL US stock prices into runs/<RUN>/price_cache")
    ap.add_argument("--run", default="2025-10-04_0903", help="Run-kansion nimi runs-hakemistossa")
    ap.add_argument("--chunk", type=int, default=50, help="Tickerien määrä yhdessä erässä")
    ap.add_argument("--sleep", type=float, default=0.25, help="Viive sek per yksittäinen fallback-lataus")
    ap.add_argument("--limit", type=int, default=0, help="Rajoita universumia (debug)")
    ap.add_argument("--overwrite", action="store_true", help="Yliaja olemassa olevat CSV:t")
    args = ap.parse_args()

    dest_dir = ensure_dirs(args.run)
    log(f"[INFO] Tallennuskansio: {dest_dir.resolve()}")

    log("[STEP] Ladataan USA-tickerilistat NASDAQ Traderista (useilla varadomaineilla)...")
    syms = load_us_tickers()
    log(f"[INFO] Tickerit saatu: {len(syms)} kpl ennen rajausta")

    if args.limit > 0:
        syms = syms[:args.limit]
        log(f"[INFO] LIMIT käytössä -> {len(syms)} tikkeriä")

    if not args.overwrite:
        before = len(syms)
        syms = [t for t in syms if not (dest_dir / f"{t}.csv").exists()]
        log(f"[INFO] Skippasin {before - len(syms)} valmiiksi ladattua (overwrite=OFF)")

    if not syms:
        log("[DONE] Ei ladattavaa."); return

    total = len(syms)
    ok_total = 0; fail_total = 0
    for i in range(0, total, args.chunk):
        part = syms[i:i+args.chunk]
        ok, fail = save_prices_for_chunk(part, dest_dir, overwrite=args.overwrite, sleep=args.sleep)
        ok_total += ok; fail_total += fail
        log(f"[BATCH] {i+len(part)}/{total} | ok+={ok} fail+={fail} | kumulatiivinen ok={ok_total} fail={fail_total}")

    log_path = dest_dir / "download_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"RUN {args.run}  total={total}  ok={ok_total}  fail={fail_total}\n")

    log(f"[DONE] Valmis. OK={ok_total}, FAIL={fail_total}. Loki: {log_path}")

if __name__ == "__main__":
    main()
