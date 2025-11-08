# run_backtest.py
"""
Run Enhanced Backtest
Usage: python run_backtest.py
"""

from backtest_scripts import BacktestEngine

def main():
    print("=" * 80)
    print("STARTING BACKTEST")
    print("=" * 80)
    print("")
    
    # Create engine with default config
    engine = BacktestEngine()
    
    # Run backtest
    results = engine.run()
    
    # Print summary
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    
    equity_curve = results['equity_curve']
    
    if not equity_curve.empty:
        initial = equity_curve['total_value'].iloc[0]
        final = equity_curve['total_value'].iloc[-1]
        total_return = ((final - initial) / initial) * 100
        
        print(f"\nInitial Capital:  ${initial:,.2f}")
        print(f"Final Value:      ${final:,.2f}")
        print(f"Total Return:     {total_return:.2f}%")
        print(f"\nTotal Trades:     {len(results['trades_history'][results['trades_history']['action'] == 'SELL'])}")
        print(f"\nResults saved to: seasonality_reports/backtest_results/")
    
    print("\n✅ Done!\n")

if __name__ == "__main__":
    main()
