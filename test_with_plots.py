# test_with_plots.py
"""
Mini backtest with visualizations
"""

from datetime import date
from backtest_scripts import BacktestEngine

def main():
    print("=" * 80)
    print("MINI BACKTEST WITH VISUALIZATIONS")
    print("=" * 80)
    print("")
    
    config_overrides = {
        'start_date': date(2024, 1, 1),
        'end_date': date(2025, 1, 1),
        'initial_cash': 50000.0,
        'max_positions': 5,
        'gate_alpha': 0.05,
        'save_plots': True,  # Enable plots
    }
    
    engine = BacktestEngine(config_overrides=config_overrides)
    
    # Limit to 10 stocks
    engine.universe = engine.universe[:10]
    
    # Reload with lookback
    from datetime import timedelta
    lookback_start = config_overrides['start_date'] - timedelta(days=365)
    engine.stock_prices = engine.data_loader.preload_all_stock_prices(
        engine.universe, lookback_start, config_overrides['end_date']
    )
    
    # Run
    results = engine.run()
    
    print("\n✅ Test complete! Check plots in:")
    print("   seasonality_reports/backtest_results/.../plots/")

if __name__ == "__main__":
    main()
