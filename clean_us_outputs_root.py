# clean_us_outputs_root.py
# Poistaa (oletuksena siirtää trash-kansioon) us_seasonality_full.py -ajon tuottamat
# koonti/universumi-CSV:t seasonality_reports-kansion JUURESTA jättämällä rauhaan
# osakekohtaiset tiedostot (Seasonality_up_down_week), price_cache, vintage jne.
#
# Kaytto:
#   py clean_us_outputs_root.py                # siirto trash-kansioon (suositus)
#   py clean_us_outputs_root.py --hard-delete  # kova poisto
#   py clean_us_outputs_root.py --dry-run      # esikatselu
#   py clean_us_outputs_root.py --include-aggregates
#
import os, glob, shutil, datetime as dt
from pathlib import Path

REPORTS_DIR = Path("seasonality_reports")
RUNS_DIR    = REPORTS_DIR / "runs"

# Nämä ovat tyypilliset us_seasonality_full.py:n JUUREEN kirjoittamat koonti-/universumi-CSV:t.
# Huom: matchataan VAIN juuritasolta (ei alikansioista).
FILE_PATTERNS = [
    "best_*_windows_all*.csv",
    "global_top_seasonality_windows*.csv",
    "*_top15_same_direction*.csv",
    "*_top15_anti_direction*.csv",
    "universe_filtered*.csv",
    "universe_with_funda*.csv",
    "constituents_raw*.csv",
]

# Valinnaisesti mukaan (rootissa ollut aiemmin)
FOLDER_AGGREGATES = "aggregates"

def ts_now():
    return dt.datetime.now().strftime("%Y-%m-%d_%H%M")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def discover_root_files(base: Path, patterns):
    found = []
    for pat in patterns:
        for p in base.glob(pat):
            if p.is_file() and p.parent.resolve() == base.resolve():
                found.append(p)
    return sorted(found)

def move_to_trash(paths, include_dirs=False):
    trash_dir = ensure_dir(RUNS_DIR / "_trash" / ts_now())
    moved = []
    for p in paths:
        dst = trash_dir / p.name
        # jos nimi olemassa, tee uniikki
        if dst.exists():
            stem, suf = dst.stem, dst.suffix
            k = 1
            while True:
                cand = dst.with_name(f"{stem}_{k}{suf}")
                if not cand.exists():
                    dst = cand
                    break
                k += 1
        shutil.move(str(p), str(dst))
        moved.append((p, dst))
    return trash_dir, moved

def hard_delete(paths):
    deleted = []
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        deleted.append(p)
    return deleted

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Clean us_seasonality_full root outputs from seasonality_reports.")
    ap.add_argument("--hard-delete", action="store_true", help="Poista pysyvästi siirron sijaan.")
    ap.add_argument("--include-aggregates", action="store_true", help="Sisällytä JUUREN aggregates/-kansio poistoon.")
    ap.add_argument("--dry-run", action="store_true", help="Näytä mitä tehtäisiin, muuttamatta mitään.")
    args = ap.parse_args()

    if not REPORTS_DIR.exists():
        raise SystemExit(f"Not found: {REPORTS_DIR.resolve()}")

    # 1) Etsi root-CSV:t (vain juuresta)
    root_csvs = discover_root_files(REPORTS_DIR, FILE_PATTERNS)

    # 2) (valinnainen) lisää rootin aggregates/-kansio
    targets = list(root_csvs)
    agg_dir = REPORTS_DIR / FOLDER_AGGREGATES
    if args.include_aggregates and agg_dir.exists() and agg_dir.is_dir():
        targets.append(agg_dir)

    if not targets:
        print("[INFO] Ei löydetty siivottavia koonti-/universumi-tuloksia seasonality_reports-juuresta.")
        return

    print("[INFO] Siivottavat kohteet:")
    for p in targets:
        rel = p.relative_to(REPORTS_DIR)
        print("  -", rel)

    if args.dry_run:
        print("\n[DRY-RUN] Ei tehty muutoksia.")
        return

    # 3) Tee siirto roskakansioon TAI kova poisto
    if args.hard_delete:
        deleted = hard_delete(targets)
        print(f"[DONE] Kova poisto: {len(deleted)} kohdetta poistettu pysyvästi.")
    else:
        ensure_dir(RUNS_DIR / "_trash")
        trash_dir, moved = move_to_trash(targets)
        print(f"[DONE] Siirretty roskakansioon: {trash_dir}")
        for src, dst in moved:
            print("  ->", src.name, "=>", dst.name)

if __name__ == "__main__":
    main()
