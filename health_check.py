# health_check.py
from pathlib import Path
import pandas as pd

RUN = Path(r"C:\Users\OMISTAJA\seasonality_project\seasonality_reports\runs\2025-10-19_1540")
act = RUN / "actions"
reps = RUN / "reports"

def rows(p: Path) -> int:
    try:
        return 0 if not p.exists() else len(pd.read_csv(p))
    except Exception:
        return -1

def show(label, p):
    p = Path(p)
    n = rows(p)
    print(f"{label:<30} {p}  rows={n}")

print("== ACTIONS ==")
for d in sorted(act.glob("*")):
    if d.is_dir():
        show("portfolio_after_sim", d / "portfolio_after_sim.csv")
        show("trade_candidates",   d / "trade_candidates.csv")
        show("sell_candidates",    d / "sell_candidates.csv")
        show("exit_watchlist",     d / "exit_watchlist.csv")
print("\n== REPORTS ==")
for p in sorted((reps).glob("*")):
    print(p.name)
