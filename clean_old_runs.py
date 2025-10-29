# clean_old_runs.py
from pathlib import Path

RUNS = Path(r"C:\Users\OMISTAJA\seasonality_project\seasonality_reports\runs")
KEEP = 14

items = [d for d in RUNS.iterdir() if d.is_dir() and d.name[:4].isdigit()]
items.sort(reverse=True)  # uusimmat ensin
for d in items[KEEP:]:
    # Poista vain, jos sisällä ei ole price_cachea (tai jätä se rauhaan jos haluat)
    if (d / "price_cache").exists():
        continue
    try:
        import shutil
        shutil.rmtree(d)
        print(f"Removed {d}")
    except Exception as e:
        print(f"[WARN] Could not remove {d}: {e}")
