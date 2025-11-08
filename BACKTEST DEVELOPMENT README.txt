BACKTEST DEVELOPMENT README - Status & Plan

Project: Seasonality Trading System - Backtest Engine
Created: 2025-11-08 09:59 UTC
Status: Planning Phase (Not Implemented Yet)
Next Priority: ml_unified_pipeline.py Enhancement
📊 CURRENT SYSTEM ANALYSIS (Completed)
✅ Live System Components:
Code

seasonality_project/
├── auto_decider.py               ← Decision engine (USES regime_detector.py)
├── regime_detector.py            ← 7 regimes, 5 components (FULL VERSION)
├── regime_strategies.py          ← Strategy per regime
├── ml_unified_pipeline.py        ← ⚠️ SIMPLIFIED (momentum only, NO seasonality)
│
├── seasonality_reports/
│   ├── vintage/                  ← Pre-calculated seasonality data (20 years)
│   │   ├── A_seasonality_week.csv
│   │   ├── A_segments_up.csv
│   │   ├── A_segments_down.csv
│   │   └── {TICKER}_vintage_10y.csv (~500 tickers)
│   │
│   ├── price_cache/              ← Regime ETF prices (SPY, QQQ, IWM, GLD, etc.)
│   │   └── {SYMBOL}.csv (20 years)
│   │
│   └── runs/2025-10-04_0903/price_cache/  ← Stock prices (~500 tickers, 20 years)

🔍 KEY FINDINGS:
1. ml_unified_pipeline.py is SIMPLIFIED (NOT FULL ML)

What it DOES:
Python

# Features:
mom5, mom20, mom60, vol20  # Only momentum + volatility

# Scoring:
score_long = rank(0.6 × mom5 + 0.4 × mom20)  # 0-1 percentile ranking

# Regime:
median_mom20 > 0.02 → "Bull"
median_mom20 < -0.02 → "Bear"
else → "Neutral"

What it DOESN'T DO:

    ❌ No seasonality calculations (despite 20 years of vintage data available!)
    ❌ No ML model (no Random Forest, XGBoost, Neural Net)
    ❌ No integration with regime_detector.py (uses simple 3-state regime)
    ❌ Vintage data exists but is NOT used

2. auto_decider.py USES FULL regime_detector.py

Confirmed:
Python

# auto_decider.py line 42:
from regime_detector import RegimeDetector

# auto_decider.py line 609-613:
detector = RegimeDetector(macro_price_cache_dir="seasonality_reports/price_cache")
regime_data = detector.detect_regime(date=today.strftime("%Y-%m-%d"))
regime = regime_data['regime']  # "BULL_STRONG", "NEUTRAL_BULLISH", "BEAR_WEAK", etc.

Regime States Used:

    BULL_STRONG
    BULL_WEAK
    NEUTRAL_BULLISH
    NEUTRAL_BEARISH
    BEAR_WEAK
    BEAR_STRONG
    CRISIS (implicit)

3. DISCONNECT Between ml_unified_pipeline and auto_decider

Problem:
Code

ml_unified_pipeline.py:
  → Produces signals (momentum-based)
  → Simple regime ("Bull"/"Bear"/"Neutral")
  
auto_decider.py:
  → Reads signals from ml_unified_pipeline
  → BUT calculates OWN regime using regime_detector.py (7 states)
  → Uses regime_strategies.py for position sizing/SL/TP

Result: Signals and regime detection are DECOUPLED
🎯 BACKTEST REQUIREMENTS (Defined)
User Objectives:

    Regime Optimization:
        Optimize threshold values (BULL_STRONG vs BULL_WEAK boundaries)
        Optimize component weights (equity 35%, volatility 20%, etc.)
        Evaluate regime detection accuracy

    Strategy Testing:
        5-year backtest (2020-2025)
        Regime-specific performance breakdown
        Walk-forward optimization

    Parameter Tuning:
        Stop Loss / Take Profit multipliers (per regime)
        Position sizing
        Max positions

    ML Model Comparison:
        Baseline: Current momentum-only system
        Enhanced: ML model + seasonality + regime_detector.py
        Comparison: Which performs better?

📁 PLANNED BACKTEST STRUCTURE
Code

seasonality_project/
├── backtest_scripts/              ← NEW: Backtest code
│   ├── backtest_engine.py         (Main engine)
│   ├── regime_calculator.py       (Historical regime detection)
│   ├── ml_calculator.py           (ML predictions for backtest)
│   ├── data_loader.py             (Load prices + vintage data)
│   ├── portfolio_simulator.py     (Simulate trades)
│   ├── performance_analyzer.py    (Metrics + reports)
│   └── config.py                  (Parameters - easily editable)
│
├── seasonality_reports/
│   └── backtest_results/          ← NEW: Backtest outputs
│       ├── 2020-2025_baseline/    (Momentum-only, current system)
│       │   ├── equity_curve.csv
│       │   ├── trades_history.csv
│       │   ├── regime_breakdown.csv
│       │   ├── regime_optimization_data.csv
│       │   ├── strategy_performance_by_regime.csv
│       │   ├── equity_curve.png
│       │   ├── drawdown.png
│       │   └── performance_report.html
│       │
│       ├── 2020-2025_enhanced/    (ML + Seasonality + Full Regime)
│       └── 2020-2025_optimized/   (Walk-forward optimized)

🔧 TECHNICAL DECISIONS (Confirmed)
Component	Decision	Rationale
ML Predictions	Calculate walk-forward	Realistic simulation, no future leak
Regime Detection	Use regime_detector.py logic (7 states)	Same as auto_decider.py
Stock Universe	constituents_raw.csv (current)	Accept survivorship bias (known limitation)
Reporting	Regime-specific + overall	Optimize per regime
Entry Price	T open + random gap (±1-2%)	Realistic slippage
SL/TP	Regime-based (optimizable)	Per regime_strategies.py
Timeframe	5 years (2020-2025)	Covers COVID, 2022 bear, 2023-25 recovery
All Reports	Yes (equity curve, trades, regime breakdown, HTML)	Full analysis
⚙️ WALK-FORWARD OPTIMIZATION CONFIG
Python

# config.py (to be created)

BACKTEST_CONFIG = {
    # Walk-forward settings
    'walk_forward': {
        'train_window': 180,    # 6 months training
        'test_window': 30,      # 1 month testing
        'step_size': 30,        # Re-optimize monthly
    },
    
    # Optimization settings
    'optimization': {
        'method': 'grid',       # or 'bayesian' for faster convergence
        'max_iterations': 100,
    },
    
    # Regime parameters to optimize
    'regime_params': {
        'optimize_thresholds': True,
        'optimize_weights': True,
        'threshold_range': [0.1, 0.8],
        'weight_range': [0.0, 0.5],
    },
    
    # Strategy parameters to optimize
    'strategy_params': {
        'sl_multiplier': [0.8, 1.0, 1.2, 1.5],
        'tp_multiplier': [1.5, 2.0, 2.5, 3.0],
        'max_positions': [5, 8, 10, 12],
    }
}

📊 EXPECTED OUTPUT STRUCTURE
regime_optimization_data.csv:
CSV

date,regime_detected,composite_score,equity_signal,volatility_signal,credit_signal,safe_haven_signal,breadth_signal,portfolio_return_next_day
2020-01-02,NEUTRAL_BULLISH,0.128,0.005,0.176,0.009,0.039,0.836,0.012
2020-01-03,NEUTRAL_BULLISH,0.142,0.008,0.165,0.011,0.042,0.851,0.008
...

strategy_performance_by_regime.csv:
CSV

regime,trades_count,win_rate,avg_return,sharpe,max_drawdown
BULL_STRONG,45,68%,+5.2%,1.8,-8.5%
NEUTRAL_BULLISH,120,58%,+2.1%,1.2,-12.3%
BEAR_STRONG,15,40%,-1.5%,0.3,-18.7%
...

🚨 RISKS & LIMITATIONS (Identified)
Risk	Severity	Mitigation
Survivorship Bias	Medium	Document limitation; compare to SPY benchmark
Overfitting	Medium	Walk-forward validation; out-of-sample testing
Regime Lag	Low	Accept (same as live system)
Parameter Explosion	Medium	Prioritize most impactful parameters; Bayesian optimization
Data Size	Low	~300 KB per backtest (954 GB available)
⏱️ ESTIMATED TIMELINE (Not Started)
Phase 1: Baseline Backtest (Current System)

    Time: 3-4 hours development
    Compute: 5-10 minutes
    Output: Baseline performance metrics

Phase 2: Enhanced Backtest (ML + Seasonality + Full Regime)

    Time: +6-8 hours development
    Compute: 30-60 minutes
    Output: Enhanced performance, comparison to baseline

Phase 3: Optimization

    Time: +4-6 hours development
    Compute: 2-4 hours (walk-forward)
    Output: Optimized parameters, best Sharpe ratio

Total Estimate:

    Development: 15-20 hours
    Compute: 3-5 hours
    Deliverable: Full backtest system with optimization

🎯 CURRENT PRIORITY: ml_unified_pipeline.py Enhancement
Issue:

ml_unified_pipeline.py is simplified (momentum-only), but 20 years of seasonality data (vintage/) exists unused.
Goal:

Enhance ml_unified_pipeline.py to become a proper ML system:

    Add seasonality features (use vintage/ data)
    Add ML model (Random Forest / XGBoost)
    Integrate regime_detector.py (7-state regime)
    Make it production-ready for live trading

Why First:

    Backtest will test the ENHANCED system
    No point backtesting the simplified version if we're upgrading it anyway
    Enhanced ml_unified_pipeline can be used BOTH for backtest AND live trading

📝 NEXT STEPS (When Returning to Backtest):

    ✅ Complete ml_unified_pipeline.py enhancement
    ✅ Test enhanced pipeline on recent data (1-2 weeks)
    ✅ Begin backtest development (baseline + enhanced + optimization)
    ✅ Analyze results (regime breakdown, parameter optimization)
    ✅ Deploy optimized parameters to live system

💾 FILES TO PRESERVE (When Resuming):

    This README.md
    Current system analysis (regime usage findings)
    Planned backtest structure
    Config templates
    Risk assessment

🔗 RELATED DOCUMENTATION:

    regime_detector.py - Full regime detection (7 states, 5 components)
    regime_strategies.py - Strategy per regime
    auto_decider.py - Decision engine (uses regime_detector.py)
    ml_unified_pipeline.py - Simplified signal generator (TO BE ENHANCED)

Status: ⏸️ PAUSED - Prioritizing ml_unified_pipeline.py enhancement
Resume: After ml_unified_pipeline.py is production-ready
Contact: @panuaalto1-afk