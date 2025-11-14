# Enhanced Backtest System - 10-Year Analysis

## Overview

Comprehensive backtesting system with:
- 10-year historical analysis (2015-2025)
- Sector diversification
- Regime-based trading strategies
- ML signal generation
- Advanced performance metrics

## Quick Start

### 1. Run Enhanced Backtest

```bash
python run_backtest_enhanced.py
```

### 2. Check Results

Results are saved to:
```
seasonality_reports/backtest_results/[date]_[time]/
├── equity_curve.csv
├── trades_history.csv
├── yearly_breakdown.csv
├── sector_breakdown.csv
├── regime_breakdown.csv
├── monthly_returns.csv
├── performance_summary.txt
└── plots/
    ├── equity_curve.png
    ├── drawdown.png
    ├── yearly_breakdown.png
    ├── sector_heatmap.png
    ├── regime_performance.png
    └── ... (more visualizations)
```

## Configuration

Edit `backtest_scripts/config.py` to modify:

### Backtest Period
```python
BACKTEST_START = date(2015, 1, 2)
BACKTEST_END = date(2025, 11, 8)
```

### Portfolio Settings
```python
INITIAL_CASH = 100000.0
MAX_POSITIONS = 20
POSITION_SIZE = 5000.0
```

### Sector Diversification
```python
ENABLE_SECTOR_DIVERSIFICATION = True
MAX_POSITIONS_PER_SECTOR = 4
```

### Regime Strategies
```python
REGIME_STRATEGIES = {
    'BULL_STRONG': {
        'position_size_multiplier': 1.3,
        'max_positions': 20,
        'min_hold_days': 14,
        'tp_multiplier': 4.0,
    },
    # ... other regimes
}
```

## Key Features

### 1. Sector Diversification
- Tracks GICS sectors from constituents file
- Limits positions per sector (configurable)
- Prevents sector concentration risk

### 2. Regime-Based Trading
- 7 market regimes: BULL_STRONG → CRISIS
- Dynamic position sizing based on regime
- Minimum hold periods per regime
- Adaptive stop-loss/take-profit levels

### 3. 10-Year Analysis
- Yearly performance breakdown
- Monthly returns heatmap
- Rolling metrics (Sharpe, volatility)
- Drawdown analysis over time

### 4. Advanced Metrics
- CAGR (Compound Annual Growth Rate)
- Sharpe, Sortino, Calmar ratios
- Alpha/Beta vs benchmarks (SPY, QQQ)
- Win rate, profit factor
- Average hold times

### 5. Comprehensive Visualizations
- Equity curve vs benchmarks
- Drawdown chart
- Monthly returns heatmap
- Yearly breakdown
- Sector performance heatmap
- Regime performance
- Trade distribution
- Rolling Sharpe ratio
- Hold time scatter plots

## Output Files

### equity_curve.csv
Daily portfolio values:
- date, cash, position_value, total_value, num_positions

### trades_history.csv
All executed trades:
- ticker, sector, entry_date, exit_date, entry_price, exit_price
- shares, pl, pl_pct, hold_days, reason

### yearly_breakdown.csv
Annual performance:
- year, return, volatility, sharpe_ratio, max_drawdown
- num_trades, win_rate, start_value, end_value

### sector_breakdown.csv
Performance by sector:
- sector, num_trades, avg_return, total_return
- avg_hold_days, win_rate

### regime_breakdown.csv
Performance by market regime:
- regime, trades_count, avg_return_pct, total_return_pct
- avg_hold_days, win_rate_pct, sharpe_ratio

### performance_summary.txt
Human-readable summary with all key metrics

## Architecture

```
backtest_scripts/
├── config.py                    # Configuration parameters
├── backtest_engine.py           # Main orchestrator
├── data_loader.py               # Price data loading
├── portfolio.py                 # Position management
├── regime_calculator.py         # Market regime detection
├── seasonality_calculator.py    # Seasonality features
├── ml_signal_generator.py       # Signal generation
├── auto_decider_simulator.py    # Trading decisions
├── performance_analyzer.py      # Metrics calculation
└── visualizer.py                # Plotting functions
```

## Performance Metrics Explained

### CAGR (Compound Annual Growth Rate)
Annualized return over the period

### Sharpe Ratio
Risk-adjusted return (return / volatility)
- > 1.0: Good
- > 2.0: Excellent

### Sortino Ratio
Like Sharpe but only considers downside volatility

### Calmar Ratio
CAGR / Max Drawdown
- Higher is better

### Max Drawdown
Largest peak-to-trough decline

### Alpha
Excess return vs benchmark (after adjusting for beta)

### Beta
Sensitivity to market movements
- 1.0: Same as market
- > 1.0: More volatile
- < 1.0: Less volatile

## Troubleshooting

### Issue: No sector data
**Solution:** Ensure constituents CSV has 'GICS Sector' or 'Sector' column

### Issue: Missing price data
**Solution:** Check STOCK_PRICE_CACHE and MACRO_PRICE_CACHE paths in config.py

### Issue: Slow execution
**Solution:** Reduce universe size or date range

### Issue: No trades executed
**Solution:** Lower GATE_ALPHA threshold or adjust regime strategies

## Examples

### Conservative Strategy
```python
config_overrides = {
    'max_positions': 10,
    'position_size': 3000.0,
    'gate_alpha': 0.15,  # Higher threshold
}
```

### Aggressive Strategy
```python
config_overrides = {
    'max_positions': 30,
    'position_size': 7000.0,
    'gate_alpha': 0.05,  # Lower threshold
}
```

### Sector Focused
```python
config_overrides = {
    'enable_sector_diversification': True,
    'max_positions_per_sector': 2,  # Max 2 per sector
}
```

## Next Steps

1. Review performance_summary.txt
2. Analyze yearly_breakdown.csv for consistency
3. Check sector_breakdown.csv for concentration
4. Examine regime_breakdown.csv for strategy effectiveness
5. Review plots/ directory for visual insights

## Support

For issues or questions, check:
- config.py for configuration options
- performance_summary.txt for results
- Console output for warnings/errors

---

**Last Updated:** 2025-11-11
**Version:** 2.0 - Enhanced 10-Year Analysis