# backtest_scripts/config.py
"""
Backtest Configuration - VERSION 2.0
Dynamic Position Sizing + Sector-Specific Strategies

UPDATED: 2025-11-12 15:16 UTC
CHANGES:
  - Added dynamic position sizing (percentage-based)
  - Increased gate_alpha for better trade quality
  - Enhanced Bull regime strategies
  - Sector-specific TP/SL parameters
  - Adaptive position sizing in drawdowns
"""

from datetime import date

# =====================================================================
# VERSION INFO
# =====================================================================

CONFIG_VERSION = "2.0"
CONFIG_DATE = "2025-11-12"
CONFIG_DESCRIPTION = "Dynamic Position Sizing + Sector Optimization"

# =====================================================================
# BACKTEST PERIOD - 10 YEARS
# =====================================================================

BACKTEST_START = date(2015, 1, 2)
BACKTEST_END = date(2025, 11, 8)

# =====================================================================
# UNIVERSE
# =====================================================================

UNIVERSE_CSV = "seasonality_reports/constituents_raw.csv"
MAX_UNIVERSE_SIZE = 500

# =====================================================================
# DATA SOURCES
# =====================================================================

STOCK_PRICE_CACHE = "seasonality_reports/runs/2025-10-04_0903/price_cache"
MACRO_PRICE_CACHE = "seasonality_reports/price_cache"
VINTAGE_DIR = "seasonality_reports/vintage"

# =====================================================================
# INITIAL CAPITAL
# =====================================================================

INITIAL_CASH = 100000.0

# =====================================================================
# PORTFOLIO RULES - DYNAMIC POSITION SIZING (NEW!)
# =====================================================================

MAX_POSITIONS = 20

# CRITICAL CHANGE: Dynamic position sizing
POSITION_SIZE_METHOD = 'percentage'  # 'fixed' or 'percentage'
POSITION_SIZE_PCT = 0.05  # 5% of portfolio value per position
POSITION_SIZE_FIXED = 5000.0  # Legacy: fixed dollar amount (backup)

# Position size limits
MIN_POSITION_SIZE = 1000.0   # Minimum $1k per position
MAX_POSITION_SIZE = 50000.0  # Maximum $50k per position
MAX_POSITION_PCT = 0.10      # Never exceed 10% in single position

# QUALITY GATE (INCREASED for better trades)
GATE_ALPHA = 0.15  # Raised from 0.10 → 0.15 (expect ~30% fewer but better trades)

# Entry/Exit rules
ENTRY_METHOD = "T_open_with_gap"
GAP_DISTRIBUTION = "historical"
SLIPPAGE_PCT = 0.001

# Stop Loss / Take Profit
USE_STOP_LOSS = True
USE_TAKE_PROFIT = True
TRAILING_STOP = False

# =====================================================================
# SECTOR DIVERSIFICATION - ENHANCED
# =====================================================================

ENABLE_SECTOR_DIVERSIFICATION = True
SECTOR_ROTATION_ENABLED = True

# Sector-specific max positions (NEW!)
SECTOR_MAX_POSITIONS = {
    'Energy': 6,  # Increased from 4 (best performer)
    'Consumer Discretionary': 5,  # Increased from 4
    'Materials': 4,
    'Technology': 3,  # Reduced (high volatility)
    'Information Technology': 3,  # Reduced
    'Health Care': 4,
    'Financials': 3,
    'Industrials': 4,
    'Consumer Staples': 3,
    'Communication Services': 3,
    'Utilities': 2,
    'Real Estate': 3,
    'Default': 2,  # All others
}

# =====================================================================
# SECTOR-SPECIFIC STRATEGIES (NEW!)
# =====================================================================

SECTOR_STRATEGIES = {
    'Energy': {
        'tp_multiplier': 5.0,      # Large moves
        'sl_multiplier': 0.9,      # Tight stop
        'min_hold_days': 30,       # Long-term trends
        'position_size_boost': 1.2, # 20% larger positions
        'volatility_tolerance': 0.04,  # High vol OK
    },
    'Consumer Discretionary': {
        'tp_multiplier': 4.0,
        'sl_multiplier': 1.0,
        'min_hold_days': 21,
        'position_size_boost': 1.1,
        'volatility_tolerance': 0.03,
    },
    'Materials': {
        'tp_multiplier': 4.0,
        'sl_multiplier': 1.0,
        'min_hold_days': 21,
        'position_size_boost': 1.0,
        'volatility_tolerance': 0.035,
    },
    'Technology': {
        'tp_multiplier': 3.0,      # Volatile, take profits faster
        'sl_multiplier': 1.2,      # Wider stop
        'min_hold_days': 14,
        'position_size_boost': 0.9, # Smaller positions (risk)
        'volatility_tolerance': 0.05,
    },
    'Information Technology': {
        'tp_multiplier': 3.0,
        'sl_multiplier': 1.2,
        'min_hold_days': 14,
        'position_size_boost': 0.9,
        'volatility_tolerance': 0.045,
    },
    'Health Care': {
        'tp_multiplier': 3.5,
        'sl_multiplier': 1.1,
        'min_hold_days': 21,
        'position_size_boost': 1.0,
        'volatility_tolerance': 0.03,
    },
    'Default': {
        'tp_multiplier': 2.5,
        'sl_multiplier': 1.0,
        'min_hold_days': 14,
        'position_size_boost': 1.0,
        'volatility_tolerance': 0.03,
    }
}

# =====================================================================
# REGIME STRATEGIES - ENHANCED BULL REGIMES
# =====================================================================

REGIME_STRATEGIES = {
    'BULL_STRONG': {
        'position_size_multiplier': 1.3,
        'stop_multiplier': 0.8,
        'tp_multiplier': 5.0,  # INCREASED: 4.0 → 5.0 (let winners run!)
        'max_positions': 20,
        'min_hold_days': 30,   # INCREASED: 14 → 30 (hold longer)
    },
    'BULL_WEAK': {
        'position_size_multiplier': 1.1,
        'stop_multiplier': 1.0,
        'tp_multiplier': 4.0,  # INCREASED: 3.0 → 4.0
        'max_positions': 18,
        'min_hold_days': 35,   # INCREASED: 21 → 35
    },
    'NEUTRAL_BULLISH': {
        'position_size_multiplier': 0.9,
        'stop_multiplier': 1.0,
        'tp_multiplier': 2.5,  # INCREASED: 2.0 → 2.5
        'max_positions': 15,
        'min_hold_days': 14,
    },
    'NEUTRAL_BEARISH': {
        'position_size_multiplier': 0.7,
        'stop_multiplier': 1.2,
        'tp_multiplier': 1.5,  # INCREASED: 1.2 → 1.5
        'max_positions': 10,
        'min_hold_days': 5,
    },
    'BEAR_WEAK': {
        'position_size_multiplier': 0.5,
        'stop_multiplier': 1.5,
        'tp_multiplier': 1.2,  # INCREASED: 1.0 → 1.2
        'max_positions': 6,
        'min_hold_days': 3,
    },
    'BEAR_STRONG': {
        'position_size_multiplier': 0.3,
        'stop_multiplier': 1.8,
        'tp_multiplier': 1.0,  # INCREASED: 0.8 → 1.0
        'max_positions': 3,
        'min_hold_days': 1,
    },
    'CRISIS': {
        'position_size_multiplier': 0.0,
        'stop_multiplier': 2.0,
        'tp_multiplier': 0.5,
        'max_positions': 0,
        'min_hold_days': 0,
    },
}

# =====================================================================
# ADAPTIVE POSITION SIZING (NEW!)
# =====================================================================

ADAPTIVE_POSITION_SIZING = {
    'enabled': True,
    
    # Reduce position size during drawdowns
    'drawdown_reduction': {
        'enabled': True,
        'thresholds': {
            0.03: 0.9,   # -3% DD: reduce to 90%
            0.05: 0.8,   # -5% DD: reduce to 80%
            0.07: 0.7,   # -7% DD: reduce to 70%
        }
    },
    
    # Increase position size when Sharpe is high
    'sharpe_boost': {
        'enabled': True,
        'rolling_window': 60,  # 60-day rolling Sharpe
        'thresholds': {
            2.0: 1.1,   # Sharpe > 2.0: boost 10%
            2.5: 1.2,   # Sharpe > 2.5: boost 20%
        }
    },
    
    # Reduce position size when volatility spikes
    'volatility_reduction': {
        'enabled': True,
        'threshold': 0.04,  # Daily vol > 4%
        'multiplier': 0.8,  # Reduce to 80%
    }
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

USE_ML_MODEL = False
ML_MODEL_TYPE = "lightgbm"

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
    'NEUTRAL_BEARISH': -0.10,
    'NEUTRAL_BULLISH': 0.05,
    'BULL_WEAK': 0.15,
    'BULL_STRONG': 0.30,
}

# =====================================================================
# 10-YEAR ANALYSIS CONFIGURATION
# =====================================================================

ANALYSIS_CONFIG = {
    'yearly_breakdown': True,
    'monthly_breakdown': True,
    'sector_analysis': True,
    'regime_analysis': True,
    'rolling_metrics': True,
    'rolling_window_days': 252,
    
    'analyze_by_year': True,
    'analyze_by_quarter': True,
    'analyze_by_month': True,
    
    'calculate_rolling_sharpe': True,
    'calculate_rolling_volatility': True,
    'calculate_rolling_drawdown': True,
    'calculate_alpha_beta': True,
    
    # NEW: Position sizing analysis
    'analyze_position_sizing': True,
    'analyze_sector_allocation': True,
}

# =====================================================================
# BENCHMARKS
# =====================================================================

BENCHMARKS = ['SPY', 'QQQ']

# =====================================================================
# PERFORMANCE METRICS
# =====================================================================

METRICS_TO_CALCULATE = [
    'total_return',
    'annual_return',
    'cagr',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'max_drawdown',
    'avg_drawdown',
    'max_drawdown_duration',
    'win_rate',
    'profit_factor',
    'avg_win',
    'avg_loss',
    'avg_hold_time',
    'total_trades',
    'winning_trades',
    'losing_trades',
    'alpha',
    'beta',
    'information_ratio',
    # NEW metrics
    'avg_position_size',
    'position_size_std',
    'sector_concentration',
]

# =====================================================================
# OUTPUT
# =====================================================================

OUTPUT_DIR = "seasonality_reports/backtest_results"
SAVE_PLOTS = True
SAVE_HTML_REPORT = True
SAVE_TRADES_CSV = True

SAVE_YEARLY_BREAKDOWN = True
SAVE_MONTHLY_BREAKDOWN = True
SAVE_SECTOR_ANALYSIS = True
SAVE_REGIME_ANALYSIS = True

# NEW outputs
SAVE_POSITION_SIZING_ANALYSIS = True
SAVE_SECTOR_ALLOCATION_HISTORY = True

PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_DPI = 150

# =====================================================================
# DEBUGGING / LOGGING
# =====================================================================

VERBOSE = True
LOG_LEVEL = 'INFO'

SHOW_PROGRESS_BAR = True
PROGRESS_UPDATE_FREQUENCY = 100