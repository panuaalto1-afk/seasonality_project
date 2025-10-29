# organize_us_run.py
# Moves files produced by us_seasonality_full.py from seasonality_reports root
# into a dedicated run subfolder: seasonality_reports\runs\YYYY-MM-DD_HHMM\
# Optional: also move Seasonality_up_down_week/ and aggregates/ under that run.
#
# Usage examples (run from project root, venv active):
#   py organize_us_run.py
#   py organize_us_run.py --name 2025-10-03_2145
#   py organize_us_run.py --no-move-week --no-move-aggregates --dry-run
#
import os, shutil, glob, datetime as dt
from pathlib import Path

# ---------- config ----------
REPORTS_DIR = Path("seasonality_reports")
RUNS_DIR    = REPORTS_DIR / "runs"

# file patterns created by us_seasonality_full.py in the reports root
FILE_PATTERNS = [
    "best_*_windows_all*.csv",
    "global_top_seasonality_windows*.csv",
    "*_top15_same_direction*.csv",
    "*_top15_anti_direction*.csv",
    "universe_filtered*.csv",
    "universe_with_funda*.csv",
    "constituents_raw*.csv",  # sometimes written with this name
]

FOLDER_SEASONALITY = "Seasonality_up_down_week"  # per-ticker segments/weekly files (large)
FOLDER_AGGREGATES  = "aggregates"                # if already produced in root

# ---------- helpers ----------
def ts_now():
    return dt.datetime.now().strftime("%Y-%m-%d_%H%M")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def unique_dst(dst: Path) -> Path:
    """If dst exists, add numeric suffix (_1, _2, ...)."""
    if not dst.exists():
        return dst
    stem, suf = dst.stem, dst.suffix
    parent = dst.parent
    k = 1
    while True:
        cand = parent / f"{stem}_{k}{suf}"
        if not cand.exists():
            return cand
        k += 1

def safe_move(src: Path, dst: Path, dry: bool = False):
    dst = unique_dst(dst)
    if dry:
        print(f"[DRY] MOVE {src} -> {dst}")
        return
    shutil.move(str(src), str(dst))
    print(f"[OK ] MOVE {src.name} -> {dst.relative_to(dst.parents[2]) if dst.is_file() else dst}")

def discover_files(base: Path, patterns):
    found = []
    for pat in patterns:
        found += [Path(p) for p in glob.glob(str(base / pat))]
    # keep only files
    return [p for p in found if p.is_file()]

def write_manifest(run_dir: Path, moved_files, moved_folders):
    mf = run_dir / "RUN_MANIFEST.txt"
    with mf.open("w", encoding="utf-8") as f:
        f.write(f"run_dir: {run_dir}\ncreated: {dt.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("[files]\n")
        for p in moved_files:
            f.write(f"{p}\n")
        f.write("\n[folders]\n")
        for p in moved_folders:
            f.write(f"{p}\n")
    print(f"[INFO] Manifest written: {mf}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Organize us_seasonality_full outputs into a run subfolder.")
    ap.add_argument("--name", default="", help="Run folder name (default: timestamp YYYY-MM-DD_HHMM)")
    ap.add_argument("--no-move-week", dest="move_week", action="store_false", help="Do not move Seasonality_up_down_week folder")
    ap.add_argument("--no-move-aggregates", dest="move_agg", action="store_false", help="Do not move aggregates folder")
    ap.add_argument("--dry-run", action="store_true", help="List what would be moved without changing anything")
    args = ap.parse_args()

    if not REPORTS_DIR.exists():
        raise SystemExit(f"Not found: {REPORTS_DIR.resolve()}")

    ensure_dir(RUNS_DIR)
    run_name = args.name.strip() or ts_now()
    run_dir  = ensure_dir(RUNS_DIR / run_name)

    moved_files = []
    moved_folders = []

    # 1) move known CSV outputs from reports root
    files = discover_files(REPORTS_DIR, FILE_PATTERNS)
    if not files:
        print("[WARN] No top-level CSVs matched in seasonality_reports. Nothing to move from root.")
    for f in files:
        dst = run_dir / f.name
        safe_move(f, dst, dry=args.dry_run)
        moved_files.append(str(f.name))

    # 2) move Seasonality_up_down_week folder (optional, large)
    if args.move_week:
        src_folder = REPORTS_DIR / FOLDER_SEASONALITY
        if src_folder.exists() and src_folder.is_dir():
            dst_folder = run_dir / FOLDER_SEASONALITY
            dst_folder = unique_dst(dst_folder)
            if args.dry_run:
                print(f"[DRY] MOVE DIR {src_folder} -> {dst_folder}")
            else:
                shutil.move(str(src_folder), str(dst_folder))
                print(f"[OK ] MOVE DIR {src_folder.name} -> {dst_folder.relative_to(RUNS_DIR.parent)}")
            moved_folders.append(f"{FOLDER_SEASONALITY}/")

    # 3) move aggregates folder if it already exists in root (optional)
    if args.move_agg:
        agg_src = REPORTS_DIR / FOLDER_AGGREGATES
        if agg_src.exists() and agg_src.is_dir():
            agg_dst = run_dir / FOLDER_AGGREGATES
            agg_dst = unique_dst(agg_dst)
            if args.dry_run:
                print(f"[DRY] MOVE DIR {agg_src} -> {agg_dst}")
            else:
                shutil.move(str(agg_src), str(agg_dst))
                print(f"[OK ] MOVE DIR {agg_src.name} -> {agg_dst.relative_to(RUNS_DIR.parent)}")
            moved_folders.append(f"{FOLDER_AGGREGATES}/")

    # 4) manifest
    if not args.dry_run:
        write_manifest(run_dir, moved_files, moved_folders)

    print(f"\n[DONE] Organized into: {run_dir}")

if __name__ == "__main__":
    main()
