import os, glob, pandas as pd, numpy as np

PRICE_DIR = None  # jätä None => haetaan uusin seasonality_reports\runs\*\price_cache

def find_latest_price_cache():
    cand = glob.glob(os.path.join("seasonality_reports","runs","*","price_cache"))
    cand = [d for d in cand if os.path.isdir(d)]
    if not cand:
        raise FileNotFoundError("Yhtään price_cache-kansiota ei löytynyt.")
    # uusin muutospäivän mukaan
    return max(cand, key=os.path.getmtime)

def adv60_median(csv_path):
    try:
        df = pd.read_csv(csv_path, usecols=["Date","Close","Volume"])
    except Exception:
        # fallback: autodetect-sep
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df = df[["Date","Close","Volume"]]
    df = df.dropna()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna()
    # viimeiset 180 pvä riittää
    df = df.tail(180)
    adv = (df["Close"] * df["Volume"]).rolling(60, min_periods=40).mean().median()
    return float(adv) if np.isfinite(adv) else np.nan

def main():
    price_dir = PRICE_DIR or find_latest_price_cache()
    print(f"[INFO] Using price_dir: {price_dir}")

    rows = []
    for fn in os.listdir(price_dir):
        if not fn.lower().endswith(".csv"): continue
        sym = os.path.splitext(fn)[0].upper()
        path = os.path.join(price_dir, fn)
        a = adv60_median(path)
        rows.append({"ticker": sym, "ADV": a, "adv": a, "adv_usd": a})
    rows.sort(key=lambda r: (-(r["ADV"] if r["ADV"]==r["ADV"] else -1), r["ticker"]))

    out_path = os.path.join("seasonality_reports","universe_with_funda.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote {out_path} with {len(rows)} tickers")

if __name__ == "__main__":
    main()
