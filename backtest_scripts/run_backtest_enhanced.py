# run_backtest_enhanced.py
"""
Run Enhanced Backtest with Sector Analysis
10-year backtest with full reporting

UPDATED: 2025-11-11 15:40 UTC - Verified for 10-year run
"""

import sys
import os
from datetime import date, datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_scripts.backtest_engine import BacktestEngine

def main():
    """
    Run 10-year backtest with sector analysis
    """
    
    print("=" * 80)
    print("ENHANCED BACKTEST - WITH SECTOR ANALYSIS")
    print("=" * 80)
    print()
    
    # Configuration
    constituents_path = "seasonality_reports/constituents_raw.csv"
    
    # 10-year backtest period
    start_date = date(2015, 1, 2)  # 10 years back from 2025
    end_date = date(2025, 11, 8)
    
    print(f"Constituents path: {constituents_path}")
    print(f"Period: {start_date} to {end_date}")
    print()
    
    # Create backtest engine with config overrides
    config_overrides = {
        'start_date': start_date,
        'end_date': end_date,
        'save_plots': True
    }
    
    engine = BacktestEngine(
        config_overrides=config_overrides,
        constituents_path=constituents_path
    )
    
    # Run backtest
    results = engine.run()
    
    print()
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print()
    print(f"Total trades: {len(results['trades_history'])}")
    print(f"Final portfolio value: ${results['equity_curve'].iloc[-1]['total_value']:,.2f}")
    print()
    print("All results saved. Check:")
    print("  - seasonality_reports/backtest_results/[timestamp]/")
    print("    ├── plots/")
    print("    ├── equity_curve.csv")
    print("    ├── trades_history.csv")
    print("    ├── regime_breakdown.csv")
    print("    ├── sector_breakdown.csv")
    print("    └── performance_summary.txt")
    print()

if __name__ == "__main__":
    main()