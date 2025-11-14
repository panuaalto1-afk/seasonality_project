# backtest_scripts/config.py
"""
Backtest Configuration - Version 2.0
Centralized configuration for all backtest parameters

UPDATED: 2025-11-13 20:16 UTC
FIXES:
  - Adjusted REGIME_THRESHOLDS for -1 to +1 score range
  - Previous thresholds (0.05-0.75) caused 7/10 years to be CRISIS
  - New thresholds properly handle momentum-based scoring
  - volatility_tolerance: 0.40 (fixed from 0.03)
"""

from datetime import date

# ============================================================================
# VERSION INFO
# ============================================================================
CONFIG_VERSION = "2.0"
CONFIG_DATE = "2025-11-13"

# ============================================================================
# BACKTEST PERIOD
# ============================================================================
BACKTEST_START = date(2015, 1, 1)
BACKTEST_END = date(2025, 11, 8)

# ============================================================================
# CAPITAL & POSITION SIZING
# ============================================================================
INITIAL_CASH = 100000  # $100k starting capital

# Position sizing method: 'percentage' or 'fixed'
POSITION_SIZE_METHOD = 'percentage'

# Percentage-based sizing
POSITION_SIZE_PCT = 0.05  # 5% of portfolio per position

# Fixed dollar sizing (if using 'fixed' method)
POSITION_SIZE_FIXED = 5000  # $5k per position

# Position size limits
MIN_POSITION_SIZE = 1000   # Minimum $1k position
MAX_POSITION_SIZE = 20000  # Maximum $20k position
MAX_POSITION_PCT = 0.10    # Max 10% of portfolio in one position

# Portfolio limits
MAX_POSITIONS = 20  # Maximum concurrent positions

# ============================================================================
# ENTRY FILTERS
# ============================================================================
GATE_ALPHA = 0.15  # Minimum score to enter (0.0-1.0)

# ============================================================================
# EXIT RULES
# ============================================================================
USE_STOP_LOSS = True
USE_TAKE_PROFIT = True

# These are base values, overridden by regime/sector strategies
BASE_STOP_LOSS_PCT = 0.08    # 8% stop loss
BASE_TAKE_PROFIT_PCT = 0.15  # 15% take profit
BASE_MIN_HOLD_DAYS = 14      # Minimum hold period

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
# Slippage/costs
SLIPPAGE_PCT = 0.001  # 0.1% slippage per trade

# Drawdown protection
MAX_DRAWDOWN_STOP = 0.25  # Stop trading if DD > 25%

# ============================================================================
# REGIME CALCULATION
# ============================================================================
# Component weights for regime calculation
REGIME_COMPONENT_WEIGHTS = {
    'equity': 0.30,         # 30% - Equity market trend (SPY/QQQ)
    'trend': 0.25,          # 25% - Momentum indicators
    'volatility': 0.20,     # 20% - VIX levels
    'credit': 0.15,         # 15% - Credit spreads (HYG vs TLT)
    'safe_haven': 0.10,     # 10% - Safe haven flows (TLT, GLD)
}

# Regime classification thresholds
# UPDATED: Adjusted for -1 to +1 score range from momentum calculations
# Previous thresholds (0.05-0.75) were too high, causing most regimes to be classified as CRISIS
REGIME_THRESHOLDS = {
    'BULL_STRONG': 0.60,       # Score >= 0.60 (strong uptrend)
    'BULL_WEAK': 0.30,         # Score 0.30-0.60 (moderate uptrend)
    'NEUTRAL_BULLISH': 0.10,   # Score 0.10-0.30 (sideways bullish)
    'NEUTRAL_BEARISH': -0.10,  # Score -0.10-0.10 (sideways bearish)
    'BEAR_WEAK': -0.30,        # Score -0.30 to -0.10 (moderate downtrend)
    'BEAR_STRONG': -0.50,      # Score -0.50 to -0.30 (strong downtrend)
    'CRISIS': -1.00,           # Score < -0.50 (panic/crisis)
}

# ============================================================================
# ADAPTIVE POSITION SIZING
# ============================================================================
ADAPTIVE_POSITION_SIZING = {
    'enabled': True,
    'base_size_pct': 0.05,  # 5% base
    
    # Volatility adjustments
    'volatility': {
        'enabled': True,
        'low_vol_threshold': 0.15,   # < 15% vol
        'high_vol_threshold': 0.30,  # > 30% vol
        'low_vol_multiplier': 1.2,   # Increase size by 20%
        'high_vol_multiplier': 0.8,  # Reduce size by 20%
    },
    
    # Drawdown adjustments
    'drawdown': {
        'enabled': True,
        'threshold_minor': 0.05,   # -5% DD
        'threshold_major': 0.10,   # -10% DD
        'reduction_minor': 0.9,    # Reduce to 90%
        'reduction_major': 0.7,    # Reduce to 70%
    },
    
    # Winning streak adjustments
    'streak': {
        'enabled': True,
        'win_threshold': 3,        # 3+ wins
        'loss_threshold': 3,       # 3+ losses
        'win_multiplier': 1.1,     # Increase by 10%
        'loss_multiplier': 0.9,    # Reduce by 10%
    },
}

# ============================================================================
# REGIME-BASED STRATEGIES
# ============================================================================
REGIME_STRATEGIES = {
    'BULL_STRONG': {
        'max_positions': 20,
        'position_size_multiplier': 1.2,  # Increase size by 20%
        'stop_multiplier': 0.8,           # Tighter stops
        'tp_multiplier': 2.5,             # Wider targets
        'min_hold_days': 10,
        'gate_adjustment': 0.95,          # Lower gate slightly
    },
    'BULL_WEAK': {
        'max_positions': 18,
        'position_size_multiplier': 1.0,
        'stop_multiplier': 1.0,
        'tp_multiplier': 2.0,
        'min_hold_days': 14,
        'gate_adjustment': 1.0,
    },
    'NEUTRAL_BULLISH': {
        'max_positions': 15,
        'position_size_multiplier': 0.9,
        'stop_multiplier': 1.0,
        'tp_multiplier': 1.5,
        'min_hold_days': 14,
        'gate_adjustment': 1.0,
    },
    'NEUTRAL_BEARISH': {
        'max_positions': 12,
        'position_size_multiplier': 0.7,
        'stop_multiplier': 1.2,
        'tp_multiplier': 1.2,
        'min_hold_days': 14,
        'gate_adjustment': 1.05,  # Raise gate slightly
    },
    'BEAR_WEAK': {
        'max_positions': 8,
        'position_size_multiplier': 0.5,
        'stop_multiplier': 1.5,
        'tp_multiplier': 1.0,
        'min_hold_days': 7,
        'gate_adjustment': 1.10,
    },
    'BEAR_STRONG': {
        'max_positions': 5,
        'position_size_multiplier': 0.3,
        'stop_multiplier': 1.8,
        'tp_multiplier': 0.8,
        'min_hold_days': 5,
        'gate_adjustment': 1.15,
    },
    'CRISIS': {
        'max_positions': 0,           # No new positions
        'position_size_multiplier': 0.0,
        'stop_multiplier': 2.0,
        'tp_multiplier': 0.5,
        'min_hold_days': 0,
        'gate_adjustment': 1.20,
    },
}

# ============================================================================
# SECTOR-SPECIFIC STRATEGIES
# ============================================================================
SECTOR_STRATEGIES = {
    'Technology': {
        'max_positions': 5,
        'tp_multiplier': 2.5,
        'sl_multiplier': 1.2,
        'min_hold_days': 10,
        'volatility_tolerance': 0.40,  # 40% volatility allowed (FIXED from 0.03)
        'score_threshold_adjustment': 0.95,
    },
    'Health Care': {
        'max_positions': 4,
        'tp_multiplier': 2.0,
        'sl_multiplier': 1.0,
        'min_hold_days': 14,
        'volatility_tolerance': 0.35,
        'score_threshold_adjustment': 1.0,
    },
    'Financials': {
        'max_positions': 4,
        'tp_multiplier': 1.8,
        'sl_multiplier': 1.1,
        'min_hold_days': 14,
        'volatility_tolerance': 0.40,
        'score_threshold_adjustment': 1.0,
    },
    'Consumer Discretionary': {
        'max_positions': 4,
        'tp_multiplier': 2.2,
        'sl_multiplier': 1.1,
        'min_hold_days': 12,
        'volatility_tolerance': 0.38,
        'score_threshold_adjustment': 0.98,
    },
    'Industrials': {
        'max_positions': 3,
        'tp_multiplier': 1.8,
        'sl_multiplier': 1.0,
        'min_hold_days': 14,
        'volatility_tolerance': 0.35,
        'score_threshold_adjustment': 1.0,
    },
    'Communication Services': {
        'max_positions': 3,
        'tp_multiplier': 2.0,
        'sl_multiplier': 1.2,
        'min_hold_days': 12,
        'volatility_tolerance': 0.38,
        'score_threshold_adjustment': 1.0,
    },
    'Consumer Staples': {
        'max_positions': 3,
        'tp_multiplier': 1.5,
        'sl_multiplier': 0.9,
        'min_hold_days': 21,
        'volatility_tolerance': 0.25,
        'score_threshold_adjustment': 1.05,
    },
    'Energy': {
        'max_positions': 2,
        'tp_multiplier': 2.5,
        'sl_multiplier': 1.5,
        'min_hold_days': 10,
        'volatility_tolerance': 0.50,  # Energy is volatile
        'score_threshold_adjustment': 1.05,
    },
    'Utilities': {
        'max_positions': 2,
        'tp_multiplier': 1.3,
        'sl_multiplier': 0.8,
        'min_hold_days': 21,
        'volatility_tolerance': 0.22,
        'score_threshold_adjustment': 1.08,
    },
    'Real Estate': {
        'max_positions': 2,
        'tp_multiplier': 1.6,
        'sl_multiplier': 1.0,
        'min_hold_days': 21,
        'volatility_tolerance': 0.30,
        'score_threshold_adjustment': 1.05,
    },
    'Materials': {
        'max_positions': 2,
        'tp_multiplier': 2.0,
        'sl_multiplier': 1.3,
        'min_hold_days': 14,
        'volatility_tolerance': 0.40,
        'score_threshold_adjustment': 1.02,
    },
    'Default': {
        'max_positions': 3,
        'tp_multiplier': 2.0,
        'sl_multiplier': 1.0,
        'min_hold_days': 14,
        'volatility_tolerance': 0.40,
        'score_threshold_adjustment': 1.0,
    },
}

# Sector position limits (cross-check)
SECTOR_MAX_POSITIONS = {
    'Technology': 5,
    'Health Care': 4,
    'Financials': 4,
    'Consumer Discretionary': 4,
    'Industrials': 3,
    'Communication Services': 3,
    'Consumer Staples': 3,
    'Energy': 2,
    'Utilities': 2,
    'Real Estate': 2,
    'Materials': 2,
}

ENABLE_SECTOR_DIVERSIFICATION = True

# ============================================================================
# SEASONALITY PARAMETERS
# ============================================================================
SEASONALITY_LOOKBACK_YEARS = 10  # Use 10 years of history

# ============================================================================
# DATA PATHS
# ============================================================================
# Stock price cache (from your runs)
STOCK_PRICE_CACHE = "seasonality_reports/runs/2025-10-04_0903/price_cache"

# Macro price cache
MACRO_PRICE_CACHE = "seasonality_reports/price_cache"

# Vintage data (for historical S&P 500 membership)
VINTAGE_DIR = "seasonality_reports/vintage"

# Universe CSV
UNIVERSE_CSV = "seasonality_reports/constituents_raw.csv"

# Output directory
OUTPUT_DIR = "seasonality_reports/backtest_results"

# ============================================================================
# BENCHMARK SYMBOLS
# ============================================================================
BENCHMARKS = ['SPY', 'QQQ']  # S&P 500 and Nasdaq-100

# ============================================================================
# VISUALIZATION
# ============================================================================
SAVE_PLOTS = True
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
SHOW_PROGRESS_BAR = True

# ============================================================================
# PERFORMANCE THRESHOLDS (for alerts/monitoring)
# ============================================================================
MIN_SHARPE_RATIO = 1.0
MAX_ACCEPTABLE_DRAWDOWN = 0.20  # 20%
MIN_WIN_RATE = 0.45  # 45%

# ============================================================================
# DEBUG OPTIONS
# ============================================================================
DEBUG_MODE = False
SAVE_DAILY_SNAPSHOTS = False  # Save portfolio state each day (slow!)