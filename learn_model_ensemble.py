# learn_model_ensemble.py
# ---------------------------------------------------------
# Yhdistää viimeisten runien ennusteet (ensemble) ja kirjoittaa
# tulokset nykyisen RUN-kansion alle: <run_root>\learn\ensemble_preds.csv
#
# Lukee ensisijaisesti:
#   <run_dir>\reports\*.csv
# Fallbackina yrittää "aggregates" -puusta:
#   <project_root>\seasonality_reports\aggregates\optio_signals_enriched\<YYYY-MM-DD>\optio_price_enriched_all*.csv
#
# CSV-luku on "super-robust": tukee sep=; -ensiriviä ja UTF-16 -enkoodauksia.
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# =========================
# Peruslogitus (sekä print että file)
# =========================
def log(logfp: Path, msg: str) -> None:
    line = str(msg)
    print(line)
    try:
        logfp.parent.mkdir(parents=True, exist_ok=True)
        with logfp.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # ei kaadeta jos lokitus epäonnistuu
        pass


# =========================
# CSV-lukija: erittäin sietokykyinen
# =========================
def read_csv_flexible(fp: Path, logfp: Path) -> Optional[pd.DataFrame]:
    """
    Lue CSV erittäin robustisti:
    - tunnista "sep=..." -direktiivi ensiriviltä (enkoodausta kohden)
    - kokeile erotin/enkoodaus-kombinaatioita (mukana UTF-16-variantit)
    - viimeinen fallback: lue koko tiedosto tekstinä ja parsitaan StringIO:sta käsin
    """
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
        "utf-16",
        "utf-16le",
        "utf-16be",
    ]
    base_sep_candidates = [None, ";", ",", "\t", "|"]  # None => sniffer

    for enc in encodings:
        # 1) kurkkaa ensirivi (onko "sep=;")
        declared_sep, skip = None, 0
        try:
            with fp.open("r", encoding=enc, errors="ignore") as f:
                first_line = f.readline().replace("\x00", "").strip()
            if first_line.lower().startswith("sep=") and len(first_line) >= 5:
                declared_sep = first_line.split("=", 1)[1][:1]
                skip = 1
                log(logfp, f"[READ][HINT] {fp} declared sep via header @ {enc}: {repr(declared_sep)}")
        except Exception as e:
            log(logfp, f"[READ][HINT] {fp} head peek failed @ {enc}: {type(e).__name__}")

        sep_candidates = [declared_sep] if declared_sep else base_sep_candidates

        # 2) suora pandas.read_csv yritys
        for sep in sep_candidates:
            try:
                df = pd.read_csv(
                    fp,
                    sep=sep,
                    engine="python",
                    encoding=enc,
                    skiprows=skip,
                    on_bad_lines="skip",
                    low_memory=False,
                )
                log(logfp, f"[READ][TRY] {fp} (sep={repr(sep)}, enc={enc}, skiprows={skip}) -> shape={df.shape}")
                # jos vain 1 sarake, ei kelpaa
                if df.shape[1] > 1:
                    log(logfp, f"[READ][OK] {fp} (sep={repr(sep)}, enc={enc})")
                    return df
            except Exception as e:
                log(logfp, f"[READ][WARN] {fp} (sep={repr(sep)}, enc={enc}) fail: {type(e).__name__}")

        # 3) fallback: lue tekstinä ja parsi käsin
        try:
            text = fp.read_text(encoding=enc, errors="ignore").replace("\x00", "")
            lines = text.splitlines()
            if lines and lines[0].lower().startswith("sep="):
                fb_sep = lines[0].split("=", 1)[1][:1]
                text = "\n".join(lines[1:])
                df = pd.read_csv(io.StringIO(text), sep=fb_sep, engine="python", on_bad_lines="skip")
                log(logfp, f"[READ][FALLBACK] {fp} (manual sep={repr(fb_sep)}, enc={enc}) -> shape={df.shape}")
                if df.shape[1] > 1:
                    return df
            else:
                for fb_sep in [";", ",", "\t", "|"]:
                    df = pd.read_csv(io.StringIO(text), sep=fb_sep, engine="python", on_bad_lines="skip")
                    log(logfp, f"[READ][FALLBACK] {fp} (manual sep={repr(fb_sep)}, enc={enc}) -> shape={df.shape}")
                    if df.shape[1] > 1:
                        return df
        except Exception as e:
            log(logfp, f"[READ][FALLBACK][WARN] {fp} @ {enc} manual parse fail: {type(e).__name__}")

    log(logfp, f"[READ][ERR] {fp} kaikki yritykset epäonnistuivat tai tulos oli 1 sarake")
    return None


# =========================
# Aput: viimeiset run-kansiot
# =========================
def list_recent_runs(runs_root: Path, current_run_name: str, k: int) -> List[str]:
    names = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        nm = p.name
        if nm == current_run_name:
            continue
        # odotetaan muotoa YYYY-MM-DD_HHMM
        if len(nm) == 16 and nm[4] == "-" and nm[7] == "-" and nm[10] == "_":
            names.append(nm)
    names.sort(reverse=True)  # uusin ensin
    return names[:k]


# =========================
# Aput: ennustesarakkeiden tunnistus
# =========================
TICKER_CANDIDATES = ["ticker", "symbol", "root", "underlying", "name"]
PRED_CANDIDATES = [
    "p_up_5d",
    "p_up_10d",
    "p_up_20d",
    "p_up",
    "prob_up",
    "pred",
    "prediction",
    "score",
    "signal",
    "p_signal",
]


def pick_cols(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    cols = {c.lower(): c for c in df.columns}
    tick = next((cols[c] for c in TICKER_CANDIDATES if c in cols), None)
    pred = next((cols[c] for c in PRED_CANDIDATES if c in cols), None)
    if not tick or not pred:
        return None
    out = df[[tick, pred]].copy()
    out.columns = ["ticker", "pred"]
    # normalisoi
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out.dropna(subset=["ticker"])
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    out = out.dropna(subset=["pred"])
    out = out[out["ticker"] != ""]
    if out.empty:
        return None
    return out


# =========================
# Etsi hyviä kandidaattitiedostoja
# =========================
def enumerate_candidate_files(
    project_root: Path, run_dir: Path, today_str: str, logfp: Path
) -> List[Path]:
    out: List[Path] = []
    # 1) ensisijaisesti tämän runin reportit
    reports = run_dir / "reports"
    if reports.exists():
        for fp in reports.glob("*.csv"):
            out.append(fp)

    # 2) fallback: aggregates / optio_signals_enriched / YYYY-MM-DD / optio_price_enriched_all*.csv
    agg_root = project_root / "seasonality_reports" / "aggregates" / "optio_signals_enriched"

    if agg_root.exists():
        # Poimi päivä-kansiot (YYYY-MM-DD) ja järjestä uusin -> vanhin
        date_dirs = []
        for p in agg_root.iterdir():
            if p.is_dir():
                nm = p.name
                if len(nm) == 10 and nm[4] == "-" and nm[7] == "-":
                    date_dirs.append(nm)
        date_dirs.sort(reverse=True)
        # otetaan esim. 8 viimeisintä päivää fallbackiksi
        for d in date_dirs[:8]:
            for fname in ("optio_price_enriched_all_with_regime.csv", "optio_price_enriched_all.csv"):
                fp = agg_root / d / fname
                if fp.exists():
                    log(logfp, f"[SCAN-FB] {fp}")
                    out.append(fp)

    # Poista duplikaatit säilyttäen järjestys
    seen = set()
    unique_out: List[Path] = []
    for fp in out:
        if fp not in seen:
            unique_out.append(fp)
            seen.add(fp)
    return unique_out


# =========================
# Lue ja kerää ennusteita tiedostoista
# =========================
def collect_predictions(files: Iterable[Path], logfp: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fp in files:
        df = read_csv_flexible(fp, logfp)
        if df is None:
            continue
        sub = pick_cols(df)
        if sub is None:
            # yritetään vielä joskus kun sarakenimet isoilla/erikoisia
            log(logfp, f"[GUESS-FB] ticker=None pred=None ({fp.name})")
            continue
        sub["source"] = fp.name
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["ticker", "pred", "source"])
    return pd.concat(frames, ignore_index=True)


# =========================
# Ensemble: keskiarvo per ticker
# =========================
def make_ensemble(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame(columns=["ticker", "pred_mean", "n_models"])
    g = preds.groupby("ticker")["pred"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "pred_mean", "count": "n_models"})
    g = g.sort_values(["pred_mean", "n_models"], ascending=[False, False]).reset_index(drop=True)
    return g


# =========================
# Pääohjelma
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--today", required=True, help="YYYY-MM-DD")
    ap.add_argument("--k_runs", type=int, default=3, help="Kuinka monta edellistä runia luetaan")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    run_root = Path(args.run_root).resolve()
    today_str = args.today
    k_runs = int(args.k_runs)

    learn_dir = run_root / "learn"
    learn_dir.mkdir(parents=True, exist_ok=True)
    logfp = learn_dir / "ensemble_log.txt"

    log(logfp, f"[START] run_root={run_root}")

    runs_root = run_root.parent
    current_run_name = run_root.name

    # Lista viimeisistä raneista (poislukien tämä)
    last_runs = list_recent_runs(runs_root, current_run_name, k_runs)
    log(logfp, f"[RUNS] k={k_runs} -> {last_runs}")

    # Kerää kandidaattitiedostot:
    files: List[Path] = []
    # 1) nykyinen run
    files.extend(enumerate_candidate_files(project_root, run_root, today_str, logfp))
    # 2) edeltävät runit
    for rn in last_runs:
        try_dir = runs_root / rn
        log(logfp, f"[TRY] {try_dir}")
        files.extend(enumerate_candidate_files(project_root, try_dir, today_str, logfp))

    # Poista duplikaatit säilyttäen järjestys
    seen: set[Path] = set()
    uniq_files: List[Path] = []
    for f in files:
        if f not in seen:
            uniq_files.append(f)
            seen.add(f)

    # Lue ja yhdistä ennusteet
    preds = collect_predictions(uniq_files, logfp)
    if preds.empty:
        log(logfp, "[MISS] Ei löytynyt yhdestäkään reports/ tai aggregates/ -tiedostosta.")
        out = learn_dir / "ensemble_preds.csv"
        pd.DataFrame(columns=["ticker", "pred_mean", "n_models"]).to_csv(out, index=False, encoding="utf-8")
        log(logfp, "[WARN] Ennusteita ei löytynyt viimeisistä runeista. Kirjoitetaan tyhjä CSV.")
        return

    # Tee ensemble
    ens = make_ensemble(preds)

    # Kirjoita ulos
    out_csv = learn_dir / "ensemble_preds.csv"
    preds_csv = learn_dir / "preds_raw_concat.csv"
    preds.to_csv(preds_csv, index=False, encoding="utf-8")
    ens.to_csv(out_csv, index=False, encoding="utf-8")

    log(logfp, f"[DONE] Yhdistetty {len(preds)} riviä -> {len(ens)} tickeriä")
    log(logfp, f"[OUT] {out_csv}")
    log(logfp, f"[RAW] {preds_csv}")


if __name__ == "__main__":
    # Pandasin varoitukset vähemmälle
    pd.options.mode.copy_on_write = True
    try:
        main()
    except KeyboardInterrupt:
        print("Keskeytetty.")
        sys.exit(130)
