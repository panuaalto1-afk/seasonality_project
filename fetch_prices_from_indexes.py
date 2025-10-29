# fetch_prices_all_us.py
# ------------------------------------------------------------
# Lataa KAIKKIEN USA-osakkeiden päivädata yfinance:sta ja tallentaa:
#   seasonality_reports/runs/<RUN>/price_cache/<TICKER>.csv
# Ilman mcap/hintasuuntafiltrejä (kattava universumi).
#
# Lähde-listat: NASDAQ Trader (nasdaqlisted.txt + otherlisted.txt)
# Käyttöesimerkit (venv päällä, projektijuuresta):
#   py fetch_prices_all_us.py                          # käyttää RUN=2025-10-04_0903
#   py fetch_prices_all_us.py --run 2025-10-04_0903 --limit 500
#   py fetch_prices_all_us.py --overwrite --chunk 30 --sleep 0.25
# ------------------------------------------------------------
import argparse
import io
import sys
import time
from pathlib import Path
import requests
import pandas as pd
import yfinance as yf

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

REPORT_ROOT = Path("seasonality_reports")
RUNS_DIR    = REPORT_ROOT / "runs"

WIKI_FALLBACKS = []  # ei käytössä tässä; kaikki tulee NASDAQ Traderista

SOURCES = {
    # NASDAQ Trader viralliset symbolilistat:
    "nasdaqlisted": "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "otherlisted":  "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    # varmistus: joskus ftp-subdomain voi olla 'http' ilman ssl
    "nasdaqlisted_http": "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "otherlisted_http":  "http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}

def log(s: str):
    print(s, flush=True)

def clean_ticker_yahoo(sym: str) -> str:
    # BRK.B -> BRK-B, BF.B -> BF-B jne.
    return str(sym).strip().upper().replace(".", "-")

def fetch_txt(url: str, timeout: int = 25) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    return r.text

def load_us_tickers() -> list[str]:
    txt = None
    # yritä HTTPS ensin
    try:
        txt = fetch_txt(SOURCES["nasdaqlisted"])
        txt2 = fetch_txt(SOURCES["otherlisted"])
    except Exception:
        # fallback HTTP
        txt = fetch_txt(SOURCES["nasdaqlisted_http"])
        txt2 = fetch_txt(SOURCES["otherlisted_http"])

    # nasdaqlisted.txt: pipe-separated, sarakkeet mm. Symbol|Security Name|...|ETF|Test Issue|...
    def parse_nasdaq(text: str) -> pd.DataFrame:
        lines = [ln for ln in text.splitlines() if "|" in ln]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
        # Poista viimeinen inforivi ("File Creation Time")
        df = df[~df["Symbol"].astype(str).str.contains("File Creation Time", na=False)]
        # jätetään ETF:t ja test-issue pois universumista (osakkeet)
        if "ETF" in df.columns:
            df = df[df["ETF"].astype(str).str.upper().ne("Y")]
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper().ne("Y")]
        df = df[["Symbol"]].dropna()
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        return df

    # otherlisted.txt: ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
    def parse_other(text: str) -> pd.DataFrame:
        lines = [ln for ln in text.splitlines() if "|" in ln]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
        df = df[~df["ACT Symbol"].astype(str).str.contains("File Creation Time", na=False)]
        if "ETF" in df.columns:
            df = df[df["ETF"].astype(str).str.upper().ne("Y")]
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper().ne("Y")]
        df = df[["ACT Symbol"]].rename(columns={"ACT Symbol": "Symbol"}).dropna()
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        return df

    df1 = parse_nasdaq(txt)
    df2 = parse_other(txt2)
    all_syms = pd.concat([df1["Symbol"], df2["Symbol"]], ignore_index=True).dropna().unique().tolist()
    # muotoile Yahooon
    all_syms = sorted({clean_ticker_yahoo(s) for s in all_syms if s and s.isascii()})
    return all_syms

def ensure_dirs(run_name: str) -> Path:
    run_dir = RUNS_DIR / run_name
    dest = run_dir / "price_cache"
    dest.mkdir(parents=True, exist_ok=True)
    return dest

def save_prices_for_chunk(chunk_syms: list[str], dest_dir: Path, overwrite: bool, sleep: float) -> tuple[int,int]:
    """Lataa erissä yfinance.download:lla (nopeampi ja vakaampi kuin 1-by-1).
       Palauttaa (ok, fail)."""
    # Jätä jo olemassa olevat pois, ellei overwrite
    syms = []
    for t in chunk_syms:
        f = dest_dir / f"{t}.csv"
        if overwrite or not f.exists():
            syms.append(t)
    if not syms:
        return (0, 0)

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
        # Kun useita tickereitä -> MultiIndex-columns; kun yksi -> normaali DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            for t in syms:
                try:
                    dft = data[t].dropna()
                    if dft.empty:
                        fail += 1; continue
                    out = dft.reset_index()
                    out.to_csv(dest_dir / f"{t}.csv", index=False)
                    ok += 1
                except Exception:
                    fail += 1
        else:
            # yksi tikkeri
            dft = data.dropna()
            if not dft.empty:
                out = dft.reset_index()
                out.to_csv(dest_dir / f"{syms[0]}.csv", index=False)
                ok += 1
            else:
                fail += 1
    except Exception:
        # jos batch epäonnistuu, tee yksitellen varalla
        for t in syms:
            try:
                d = yf.download(t, period="max", interval="1d", auto_adjust=False,
                                actions=False, progress=False)
                if d is None or d.empty:
                    fail += 1; continue
                out = d.reset_index()
                out.to_csv(dest_dir / f"{t}.csv", index=False)
                ok += 1
                time.sleep(sleep)
            except Exception:
                fail += 1
    return (ok, fail)

def main():
    ap = argparse.ArgumentParser(description="Fetch ALL US stock prices into runs/<RUN>/price_cache")
    ap.add_argument("--run", default="2025-10-04_0903", help="Run-kansion nimi runs-hakemistossa")
    ap.add_argument("--chunk", type=int, default=50, help="Tickerien määrä yhdessä erässä")
    ap.add_argument("--sleep", type=float, default=0.20, help="Viive sek per yksittäinen fallback-lataus")
    ap.add_argument("--limit", type=int, default=0, help="Rajoita universumia (debug)")
    ap.add_argument("--overwrite", action="store_true", help="Yliaja olemassa olevat CSV:t")
    args = ap.parse_args()

    dest_dir = ensure_dirs(args.run)
    log(f"[INFO] Tallennuskansio: {dest_dir.resolve()}")

    log("[STEP] Ladataan USA-tickerilistat NASDAQ Traderista...")
    syms = load_us_tickers()
    log(f"[INFO] Tickerit saatu: {len(syms)} kpl ennen rajausta")

    if args.limit > 0:
        syms = syms[:args.limit]
        log(f"[INFO] LIMIT käytössä -> {len(syms)} tikkeriä")

    # Jätä heti valmiit välistä, jos ei overwrite
    if not args.overwrite:
        before = len(syms)
        syms = [t for t in syms if not (dest_dir / f"{t}.csv").exists()]
        log(f"[INFO] Skippasin {before - len(syms)} valmiiksi ladattua (overwrite=OFF)")

    if not syms:
        log("[DONE] Ei ladattavaa.")
        return

    # Etene erissä
    total = len(syms)
    ok_total = 0; fail_total = 0
    for i in range(0, total, args.chunk):
        part = syms[i:i+args.chunk]
        ok, fail = save_prices_for_chunk(part, dest_dir, overwrite=args.overwrite, sleep=args.sleep)
        ok_total += ok; fail_total += fail
        log(f"[BATCH] {i+len(part)}/{total} | ok+={ok} fail+={fail} | kumulatiivinen ok={ok_total} fail={fail_total}")

    # Kirjaa loki
    log_path = dest_dir / "download_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"RUN {args.run}  total={total}  ok={ok_total}  fail={fail_total}\n")

    log(f"[DONE] Valmis. OK={ok_total}, FAIL={fail_total}. Loki: {log_path}")

if __name__ == "__main__":
    main()
