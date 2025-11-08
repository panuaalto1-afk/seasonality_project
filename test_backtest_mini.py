# test_backtest_mini.py
"""
Mini backtest test (10 stocks, 1 year)
Quick validation before full run
"""

from datetime import date
from backtest_scripts import BacktestEngine

def main():
    print("=" * 80)
    print("MINI BACKTEST TEST")
    print("10 stocks, 1 year (2024-01-01 to 2025-01-01)")
    print("=" * 80)
    print("")
    
    # Override config for mini test
    config_overrides = {
        'start_date': date(2024, 1, 1),
        'end_date': date(2025, 1, 1),
        'initial_cash': 50000.0,
        'max_positions': 5,
        'gate_alpha': 0.05,  # Lowered from 0.10
    }
    
    engine = BacktestEngine(config_overrides=config_overrides)
    
    # Limit universe to 10 stocks (for speed)
    print(f"[TEST] Full universe size: {len(engine.universe)}")
    engine.universe = engine.universe[:10]
    print(f"[TEST] Limited universe to: {engine.universe}")
    
    # Reload prices with limited universe
    # (backtest_engine.py now loads with 1 year lookback automatically)
    from datetime import timedelta
    lookback_start = config_overrides['start_date'] - timedelta(days=365)
    
    engine.stock_prices = engine.data_loader.preload_all_stock_prices(
        engine.universe,
        lookback_start,  # 1 year earlier
        config_overrides['end_date']
    )
    
    print(f"[TEST] Stock prices loaded from {lookback_start} to {config_overrides['end_date']}")
    
    # Run
    results = engine.run()
    
    print("\n✅ Mini test complete!")

if __name__ == "__main__":
    main()
