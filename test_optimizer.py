# test_optimizer.py
"""
Test optimizer with small dataset
"""

from datetime import date
from backtest_scripts import BacktestEngine, WalkForwardOptimizer

def engine_factory(config_overrides=None):
    """Factory function to create engine instances"""
    engine = BacktestEngine(config_overrides=config_overrides)
    
    # Limit to 10 stocks for speed
    engine.universe = engine.universe[:10]
    
    # Reload with lookback
    from datetime import timedelta
    lookback_start = config_overrides['start_date'] - timedelta(days=365)
    engine.stock_prices = engine.data_loader.preload_all_stock_prices(
        engine.universe, lookback_start, config_overrides['end_date']
    )
    
    return engine

def main():
    print("=" * 80)
    print("OPTIMIZER TEST")
    print("=" * 80)
    print("")
    
    # Define parameter space
    param_space = {
        'gate_alpha': (0.05, 0.15),
        'max_positions': [5, 8, 10],
    }
    
    # Optimization config
    opt_config = {
        'method': 'random',  # Use random for speed (not bayesian)
        'max_iterations': 10,  # Only 10 iterations for test
    }
    
    # Create optimizer
    optimizer = WalkForwardOptimizer(
        backtest_engine_factory=engine_factory,
        param_space=param_space,
        optimization_config=opt_config
    )
    
    # Run optimization
    result = optimizer.optimize(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),  # 6 months only
        objective='sharpe'
    )
    
    print("\n✅ Optimization test complete!")
    print(f"\nBest parameters: {result['best_params']}")
    print(f"Best Sharpe: {result['best_score']:.3f}")

if __name__ == "__main__":
    main()
