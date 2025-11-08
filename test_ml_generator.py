# test_ml_generator.py
from datetime import date
from backtest_scripts import BacktestDataLoader, RegimeCalculator, SeasonalityCalculator, MLSignalGenerator

print("Testing ML Signal Generator...")
print("=" * 80)

# Initialize
loader = BacktestDataLoader(
    stock_price_cache="seasonality_reports/runs/2025-10-04_0903/price_cache",
    macro_price_cache="seasonality_reports/price_cache",
    vintage_dir="seasonality_reports/vintage"
)

# Load 10 stocks
universe = loader.load_universe("seasonality_reports/constituents_raw.csv")[:10]
print(f"\nUniverse (10 stocks): {universe}")

stock_prices = loader.preload_all_stock_prices(universe, date(2020, 1, 1), date(2024, 12, 31))
print(f"Stock prices loaded: {len(stock_prices)} tickers")

# Check data availability
for ticker in universe[:3]:
    if ticker in stock_prices:
        df = stock_prices[ticker]
        print(f"\n{ticker}: {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Sample close prices: {df['close'].tail(5).tolist()}")

# Initialize calculators
macro_prices = loader.preload_all_macro_prices(
    ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'HYG', 'LQD'], 
    date(2020, 1, 1), 
    date(2024, 12, 31)
)

regime_calc = RegimeCalculator(macro_prices)
seasonality_calc = SeasonalityCalculator(lookback_years=10)
ml_gen = MLSignalGenerator(seasonality_calc)

# Test generation
test_date = date(2024, 3, 1)
print(f"\n\nGenerating signals for: {test_date}")
print("-" * 80)

regime_data = regime_calc.calculate_regime(test_date)
regime = regime_data['regime']
print(f"Regime: {regime} (score: {regime_data['composite_score']:.3f})")

candidates = ml_gen.generate_signals(
    target_date=test_date,
    stock_prices=stock_prices,
    regime=regime,
    gate_alpha=0.10
)

print(f"\n\nRESULTS:")
print(f"  Candidates shape: {candidates.shape}")
print(f"  Columns: {list(candidates.columns)}")

if not candidates.empty:
    print(f"\n  Top 5 candidates:")
    print(candidates[['ticker', 'mom5', 'mom20', 'score_long', 'entry_price']].head())
else:
    print("\n  ❌ NO CANDIDATES GENERATED!")
    print("\n  Debugging: Check individual stock processing...")
    
    # Debug first stock
    test_ticker = universe[0]
    if test_ticker in stock_prices:
        prices = stock_prices[test_ticker]
        prices_before = prices[prices['date'] < test_date]
        
        print(f"\n  Test ticker: {test_ticker}")
        print(f"    Total rows: {len(prices)}")
        print(f"    Rows before {test_date}: {len(prices_before)}")
        print(f"    Last 5 rows before target date:")
        print(prices_before[['date', 'close']].tail())
