# run_backtest_enhanced.py
"""
Run Enhanced Backtest with Sector Analysis
10-year backtest with full reporting

UPDATED: 2025-11-11 19:39 UTC - Enhanced for comprehensive 10-year analysis
"""

import sys
import os
from datetime import date, datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_scripts.backtest_engine import BacktestEngine

def main():
    """
    Run 10-year backtest with sector analysis and comprehensive reporting
    """
    
    print("=" * 80)
    print("ENHANCED BACKTEST - 10-YEAR ANALYSIS WITH SECTOR DIVERSIFICATION")
    print("=" * 80)
    print()
    
    # Configuration
    constituents_path = "seasonality_reports/constituents_raw.csv"
    
    # 10-year backtest period
    start_date = date(2015, 1, 2)  # 10 years back from 2025
    end_date = date(2025, 11, 8)
    
    print(f"Constituents path: {constituents_path}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Duration: ~{(end_date - start_date).days / 365.25:.1f} years")
    print()
    
    # Create backtest engine with config overrides
    config_overrides = {
        'start_date': start_date,
        'end_date': end_date,
        'save_plots': True,
        'initial_cash': 100000.0,
        'max_positions': 20,
        'position_size': 5000.0,
        'enable_sector_diversification': True,
        'max_positions_per_sector': 4,
    }
    
    print("Configuration:")
    print(f"  Initial Capital: ${config_overrides['initial_cash']:,.2f}")
    print(f"  Max Positions: {config_overrides['max_positions']}")
    print(f"  Position Size: ${config_overrides['position_size']:,.2f}")
    print(f"  Sector Diversification: {config_overrides['enable_sector_diversification']}")
    if config_overrides['enable_sector_diversification']:
        print(f"  Max per Sector: {config_overrides['max_positions_per_sector']}")
    print()
    
    # Initialize engine
    engine = BacktestEngine(
        config_overrides=config_overrides,
        constituents_path=constituents_path
    )
    
    # Run backtest
    print("Starting backtest simulation...")
    print()
    
    results = engine.run()
    
    # Print final summary
    print()
    print("=" * 80)
    print("BACKTEST COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    print()
    
    equity_curve = results['equity_curve']
    trades_history = results['trades_history']
    analysis = results['analysis']
    
    # Quick stats
    initial_value = equity_curve.iloc[0]['total_value']
    final_value = equity_curve.iloc[-1]['total_value']
    total_return = ((final_value - initial_value) / initial_value) * 100
    
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Total Return:            {total_return:.2f}%")
    print()
    
    if not trades_history.empty:
        print(f"Total Trades:            {len(trades_history)}")
        winning = trades_history[trades_history['pl'] > 0]
        print(f"Winning Trades:          {len(winning)}")
        print(f"Win Rate:                {(len(winning) / len(trades_history) * 100):.2f}%")
        print()
    
    # Portfolio metrics
    if 'portfolio_metrics' in analysis:
        pm = analysis['portfolio_metrics']
        print("Key Performance Metrics:")
        print(f"  CAGR:                  {pm.get('cagr', 0):.2f}%")
        print(f"  Sharpe Ratio:          {pm.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio:         {pm.get('sortino_ratio', 0):.3f}")
        print(f"  Calmar Ratio:          {pm.get('calmar_ratio', 0):.3f}")
        print(f"  Max Drawdown:          {pm.get('max_drawdown', 0):.2f}%")
        print(f"  Volatility (Annual):   {pm.get('volatility', 0):.2f}%")
        print()
    
    # Benchmark comparison
    if 'benchmark_comparison' in analysis and analysis['benchmark_comparison']:
        print("Benchmark Comparison:")
        for bench_name, bench_stats in analysis['benchmark_comparison'].items():
            print(f"  vs {bench_name}:")
            print(f"    Outperformance:      {bench_stats.get('outperformance', 0):.2f}%")
            print(f"    Alpha:               {bench_stats.get('alpha', 0):.2f}%")
            print(f"    Beta:                {bench_stats.get('beta', 0):.3f}")
        print()
    
    # Yearly performance summary
    if 'yearly_breakdown' in analysis and not analysis['yearly_breakdown'].empty:
        yb = analysis['yearly_breakdown']
        print("Yearly Performance Summary:")
        print("-" * 80)
        print(f"{'Year':<8} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<12} {'Trades':<10}")
        print("-" * 80)
        for _, row in yb.iterrows():
            print(f"{int(row['year']):<8} {row['return']:>10.2f}% {row['sharpe_ratio']:>8.3f}  {row['max_drawdown']:>10.2f}% {int(row['num_trades']):>8}")
        print()
    
    # Sector breakdown
    if 'sector_breakdown' in analysis and not analysis['sector_breakdown'].empty:
        sb = analysis['sector_breakdown']
        print("Top 5 Sectors by Performance:")
        print("-" * 80)
        top_sectors = sb.nlargest(5, 'total_return')
        for _, row in top_sectors.iterrows():
            print(f"  {row['sector']:<30} Return: {row['total_return']:>8.2f}%  Trades: {int(row['num_trades']):>4}  Win Rate: {row['win_rate']:>6.2f}%")
        print()
    
    # Output location
    print("=" * 80)
    print("All results saved to:")
    print(f"  Directory: seasonality_reports/backtest_results/")
    print(f"  Files:")
    print(f"    - equity_curve.csv")
    print(f"    - trades_history.csv")
    print(f"    - regime_breakdown.csv")
    print(f"    - sector_breakdown.csv")
    print(f"    - yearly_breakdown.csv")
    print(f"    - monthly_returns.csv")
    print(f"    - performance_summary.txt")
    print(f"    - plots/ (all visualizations)")
    print("=" * 80)
    print()
    print("✓ Backtest completed successfully!")
    print()

if __name__ == "__main__":
    main()