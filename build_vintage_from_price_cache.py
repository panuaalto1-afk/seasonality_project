# build_vintage_from_price_cache.py  (tolerantti versio)
# ------------------------------------------------------------
# Lukee kaikki CSV:t kansiosta seasonality_reports/price_cache,
# puhdistaa datan ja laskee walk-forward VINTAGE-seasonalityn.
# - Coerce numerot: Close -> to_numeric(errors='coerce')
# - Pudottaa epäkelvot rivit (esim. ",A,A,A,...") ja duplikaattipäivät
# - Skippaa selvästi väärämuotoiset CSV:t (ei Close/Date)
# Kirjoittaa:
#   seasonality_reports/vintage/<TICKER>_vintage_10y.csv
#   seasonality_reports/vintage/RELIABILITY_by_year.csv
# ------------------------------------------------------------
import os, argparse
import pandas as pd
import numpy as np

PRICE_DIR_DEFAULT = os.path.join("seasonality_reports", "price_cache")
OUT_DIR_DEFAULT   = os.path.join("seasonality_reports", "vintage")

def read_price_csv(fp: str) -> pd.DataFrame:
    # Lue joustavasti ja tunnista sarakkeet case-insensitiivisesti
    df = pd.read_csv(fp, engine="python")
    cols = {c.lower(): c for c in df.columns}

    if "date" not in cols:
        raise ValueError("Date column missing")
    # Jos Close puuttuu mutta Adj Close löytyy, käytä sitä
    if "close" not in cols and "adj close" in cols:
        df["Close"] = df[cols["adj close"]]
        cols["close"] = "Close"
    if "close" not in cols:
        raise ValueError("Close column missing")

    # Uudelleennimeä standardiin
    df = df.rename(columns={cols["date"]: "Date", cols["close"]: "Close"})

    # Coerce → numerot & ajat, tiputa virheet
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

    # Poista duplikaattipäivät (jos esim. editori on lisännyt rivejä)
    df = df.drop_duplicates(subset=["Date"], keep="last")

    # Poista ei-fyysiset arvot
    df = df[(df["Close"] > 0) & np.isfinite(df["Close"])]

    # Minivaatimus: ainakin ~50 datapistettä
    if df.empty or df["Close"].notna().sum() < 50:
        raise ValueError("Not enough valid Close data")

    df = df.set_index("Date")
    return df

def build_walkforward(close: pd.Series, lookback_years: int, ks: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    years = sorted(close.index.year.unique().tolist())
    by_year = {y: close[close.index.year == y] for y in years}
    for Y in years:
        train_years = [yy for yy in years if (yy < Y and yy >= Y - lookback_years)]
        if not train_years:
            continue
        curr = by_year[Y]
        for i, d in enumerate(curr.index):
            for K in ks:
                vals = []
                for yy in train_years:
                    prev = by_year.get(yy)
                    if prev is None or len(prev) <= i + K:
                        continue
                    vals.append(float(prev.iloc[i+K] / prev.iloc[i] - 1.0))
                if not vals:
                    out.loc[d, f"s_mean_k{K}"] = np.nan
                    out.loc[d, f"s_prob_k{K}"] = np.nan
                    out.loc[d, f"s_std_k{K}"]  = np.nan
                    out.loc[d, f"s_n_k{K}"]    = 0
                else:
                    arr = np.array(vals, float)
                    out.loc[d, f"s_mean_k{K}"] = float(np.nanmean(arr))
                    out.loc[d, f"s_prob_k{K}"] = float((arr > 0).mean())
                    out.loc[d, f"s_std_k{K}"]  = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
                    out.loc[d, f"s_n_k{K}"]    = int(len(arr))
    return out

def add_future_line(close: pd.Series, ks: list[int]) -> pd.DataFrame:
    fut = pd.DataFrame(index=close.index)
    for K in ks:
        fut[f"real_ret_k{K}"] = (close.shift(-K) / close - 1.0)
    return fut

def finalize_table(wf: pd.DataFrame, fut: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    vint = wf.join(fut, how="left").copy()
    vint["date"] = vint.index
    vint["year"] = vint["date"].dt.year
    vint["tdoy"] = vint.groupby("year").cumcount()
    for K in ks:
        mcol, rcol = f"s_mean_k{K}", f"real_ret_k{K}"
        vint[f"s_agree_k{K}"] = np.where(
            vint[[mcol, rcol]].notna().all(axis=1),
            np.sign(vint[mcol]) == np.sign(vint[rcol]),
            np.nan
        )
    meta = ["date","year","tdoy"]
    stats = [c for c in vint.columns if c.startswith(("s_mean_","s_prob_","s_std_","s_n_"))]
    futures = [c for c in vint.columns if c.startswith("real_ret_")]
    agrees = [c for c in vint.columns if c.startswith("s_agree_")]
    return vint[meta + stats + futures + agrees].reset_index(drop=True)

def reliability_per_ticker(vint: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    rows = []
    if vint.empty:
        return pd.DataFrame(columns=["ticker","year","k","n","accuracy","corr"])
    df = vint.copy()
    for K in ks:
        mcol, rcol = f"s_mean_k{K}", f"real_ret_k{K}"
        tmp = df.dropna(subset=[mcol, rcol]).copy()
        if tmp.empty: 
            continue
        tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
        for yr, g in tmp.groupby("year"):
            x, y = g[mcol].to_numpy(), g[rcol].to_numpy()
            n = len(g)
            acc = float((np.sign(x) == np.sign(y)).mean()) if n else np.nan
            corr = float(np.corrcoef(x, y)[0,1]) if n > 2 else np.nan
            rows.append({"year": int(yr), "k": int(K), "n": int(n), "accuracy": acc, "corr": corr})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Build VINTAGE from price_cache (tolerant)")
    ap.add_argument("--price-dir", default=PRICE_DIR_DEFAULT)
    ap.add_argument("--out-dir",   default=OUT_DIR_DEFAULT)
    ap.add_argument("--lookback",  type=int, default=10)
    ap.add_argument("--ks",        nargs="+", type=int, default=[10,21,63])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = [f for f in os.listdir(args.price_dir) if f.lower().endswith(".csv")]
    files.sort()
    reli_all, fails = [], []

    for fn in files:
        ticker = os.path.splitext(fn)[0].upper()
        src_fp = os.path.join(args.price_dir, fn)
        out_fp = os.path.join(args.out_dir, f"{ticker}_vintage_{args.lookback}y.csv")
        if (not args.overwrite) and os.path.exists(out_fp):
            print(f"[SKIP] {ticker} (exists)")
            continue
        try:
            px = read_price_csv(src_fp)
            close = px["Close"]
            wf = build_walkforward(close, args.lookback, args.ks)
            fut = add_future_line(close, args.ks)
            vint = finalize_table(wf, fut, args.ks)
            vint.to_csv(out_fp, index=False)
            print(f"[OK] {ticker} -> {out_fp}")

            rel = reliability_per_ticker(vint, args.ks)
            if not rel.empty:
                rel.insert(0, "ticker", ticker)
                reli_all.append(rel)
        except Exception as e:
            print(f"[FAIL] {ticker}: {e}")
            fails.append((ticker, str(e)))

    if reli_all:
        reli = pd.concat(reli_all, ignore_index=True)
        reli.to_csv(os.path.join(args.out_dir, "RELIABILITY_by_year.csv"), index=False)
        print("[SAVE] reliability table")

    if fails:
        with open(os.path.join(args.out_dir, "_failed_files.txt"), "w", encoding="utf-8") as f:
            for t, why in fails:
                f.write(f"{t}\t{why}\n")
        print(f"[NOTE] Some files failed: {_len:=len(fails)}  -> see _failed_files.txt")

    print(f"\nDONE. src={args.price_dir}  out={args.out_dir}  total={len(files)}")
    
if __name__ == "__main__":
    main()
