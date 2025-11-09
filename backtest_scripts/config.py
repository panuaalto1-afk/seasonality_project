# backtest_scripts/config.py
"""
Backtest Configuration
Easily editable parameters for backtesting
"""

from datetime import date

# =====================================================================
# BACKTEST PERIOD
# =====================================================================

BACKTEST_START = date(2020, 1, 1)
BACKTEST_END = date(2025, 11, 8)

# =====================================================================
# UNIVERSE
# =====================================================================

UNIVERSE_CSV = "seasonality_reports/constituents_raw.csv"
MAX_UNIVERSE_SIZE = 500  # Limit stocks for speed

# =====================================================================
# DATA SOURCES
# =====================================================================

STOCK_PRICE_CACHE = "seasonality_reports/runs/2025-10-04_0903/price_cache"
MACRO_PRICE_CACHE = "seasonality_reports/price_cache"
VINTAGE_DIR = "seasonality_reports/vintage"

# =====================================================================
# INITIAL CAPITAL
# =====================================================================

INITIAL_CASH = 100000.0  # $100k starting capital

# =====================================================================
# PORTFOLIO RULES
# =====================================================================

MAX_POSITIONS = 8           # Maximum concurrent positions
POSITION_SIZE = 5000.0      # $ per position (base)
GATE_ALPHA = 0.10           # Minimum score_long threshold

# Entry/Exit rules
ENTRY_METHOD = "T_open_with_gap"  # "T-1_close", "T_open", "T_open_with_gap"
GAP_DISTRIBUTION = "historical"    # "historical", "fixed", "none"
SLIPPAGE_PCT = 0.001              # 0.1% slippage

# Stop Loss / Take Profit
USE_STOP_LOSS = True
USE_TAKE_PROFIT = True
TRAILING_STOP = False  # Future feature

# =====================================================================
# REGIME STRATEGIES (per regime_strategies.py)
# =====================================================================

REGIME_STRATEGIES = {
    'BULL_STRONG': {
        'position_size_multiplier': 1.2,
        'stop_multiplier': 0.8,
        'tp_multiplier': 2.5,
        'max_positions': 10,
    },
    'BULL_WEAK': {
        'position_size_multiplier': 1.0,
        'stop_multiplier': 1.0,
        'tp_multiplier': 2.0,
        'max_positions': 8,
    },
    'NEUTRAL_BULLISH': {
        'position_size_multiplier': 0.9,
        'stop_multiplier': 1.0,
        'tp_multiplier': 1.5,
        'max_positions': 8,
    },
    'NEUTRAL_BEARISH': {
        'position_size_multiplier': 0.7,
        'stop_multiplier': 1.2,
        'tp_multiplier': 1.2,
        'max_positions': 6,
    },
    'BEAR_WEAK': {
        'position_size_multiplier': 0.5,
        'stop_multiplier': 1.5,
        'tp_multiplier': 1.0,
        'max_positions': 4,
    },
    'BEAR_STRONG': {
        'position_size_multiplier': 0.3,
        'stop_multiplier': 1.8,
        'tp_multiplier': 0.8,
        'max_positions': 2,
    },
    'CRISIS': {
        'position_size_multiplier': 0.0,  # Exit all
        'stop_multiplier': 2.0,
        'tp_multiplier': 0.5,
        'max_positions': 0,
    },
}

# =====================================================================
# ML / SEASONALITY PARAMETERS
# =====================================================================

SEASONALITY_LOOKBACK_YEARS = 10
MOMENTUM_PERIODS = {
    'short': 5,
    'medium': 20,
    'long': 60,
}
VOLATILITY_PERIOD = 20
ATR_PERIOD = 14

# ML Model (future)
USE_ML_MODEL = False  # For now: momentum + seasonality blend
ML_MODEL_TYPE = "lightgbm"  # "lightgbm", "xgboost", "random_forest"

# =====================================================================
# REGIME DETECTION PARAMETERS
# =====================================================================

REGIME_COMPONENT_WEIGHTS = {
    'equity': 0.35,
    'volatility': 0.20,
    'credit': 0.20,
    'safe_haven': 0.15,
    'breadth': 0.10,
}

REGIME_THRESHOLDS = {
    'CRISIS': -1.0,
    'BEAR_STRONG': -0.50,
    'BEAR_WEAK': -0.25,
    'NEUTRAL_BEARISH': -0.05,
    'NEUTRAL_BULLISH': 0.05,
    'BULL_WEAK': 0.25,
    'BULL_STRONG': 0.50,
}

# =====================================================================
# OPTIMIZATION PARAMETERS (Vaihe 3)
# =====================================================================

OPTIMIZATION_CONFIG = {
    'method': 'bayesian',  # 'grid', 'random', 'bayesian'
    'max_iterations': 100,
    'n_jobs': -1,  # Use all CPU cores
    
    # Walk-forward settings
    'walk_forward': {
        'train_window': 180,  # 6 months training
        'test_window': 30,    # 1 month testing
        'step_size': 30,      # Re-optimize monthly
    },
    
    # Parameter search spaces
    'param_space': {
        'stop_multiplier': (0.5, 2.0),
        'tp_multiplier': (1.0, 3.0),
        'max_positions': [5, 8, 10, 12, 15],
        'gate_alpha': (0.05, 0.20),
        'position_size_multiplier': (0.5, 1.5),
        
        # Regime thresholds (optional)
        'regime_threshold_bull_strong': (0.3, 0.7),
        'regime_threshold_neutral': (-0.1, 0.1),
        
        # Component weights (optional)
        'weight_equity': (0.2, 0.5),
        'weight_volatility': (0.1, 0.3),
    },
}

# =====================================================================
# BENCHMARKS
# =====================================================================

BENCHMARKS = ['SPY', 'QQQ']  # S&P 500, NASDAQ

# =====================================================================
# PERFORMANCE METRICS
# =====================================================================

METRICS_TO_CALCULATE = [
    'total_return',
    'annual_return',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'win_rate',
    'profit_factor',
    'avg_win',
    'avg_loss',
    'avg_hold_time',
    'total_trades',
]

# =====================================================================
# OUTPUT
# =====================================================================

OUTPUT_DIR = "seasonality_reports/backtest_results"
SAVE_PLOTS = True
SAVE_HTML_REPORT = True
SAVE_TRADES_CSV = True

# Plotting style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # matplotlib style
PLOT_DPI = 150

# =====================================================================
# DEBUGGING / LOGGING
# =====================================================================

VERBOSE = True
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
