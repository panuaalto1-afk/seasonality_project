"""
Backtest Configuration - Aggressive Setup v3.0
Date: 2025-11-14
Target: Beat SPY/QQQ with 2-year test (2023-2025)
"""

from datetime import date

# ============================================================================
# BACKTEST PERIOD
# ============================================================================
BACKTEST_START = date(2023, 1, 1)
BACKTEST_END = date(2025, 1, 1)
INITIAL_CASH = 100000.0

# ============================================================================
# POSITION SIZING - AGGRESSIVE (6% Dynamic)
# ============================================================================
POSITION_SIZE_METHOD = 'percentage'  # 'percentage' or 'fixed'
POSITION_SIZE_PCT = 0.06  # 6% of portfolio per position
POSITION_SIZE_FIXED = 5000  # Only used if method='fixed'

MIN_POSITION_SIZE = 1000.0  # Minimum $1k per position
MAX_POSITION_SIZE = 50000.0  # Maximum $50k per position
MAX_POSITION_PCT = 0.10  # Never exceed 10% of portfolio

# ============================================================================
# PORTFOLIO LIMITS - REGIME-BASED (AGGRESSIVE)
# ============================================================================
REGIME_MAX_POSITIONS = {
    'BULL_STRONG': 10,
    'BULL_WEAK': 8,
    'NEUTRAL_BULLISH': 6,
    'NEUTRAL_BEARISH': 4,
    'BEAR_WEAK': 3,
    'BEAR_STRONG': 2,
    'CRISIS': 0,
}

DEFAULT_MAX_POSITIONS = 8  # Fallback if regime unknown

# ============================================================================
# SIGNAL FILTERING - AGGRESSIVE (Lower gate = more trades)
# ============================================================================
GATE_ALPHA = 0.10  # Lower than v2.0's 0.15 = more trades
MIN_SCORE_LONG = 0.10  # Minimum ML score for entry

# ============================================================================
# SECTOR FILTERING - EXCLUDE WEAK PERFORMERS
# ============================================================================
SECTOR_BLACKLIST = [
    'Biopharmaceutical',      # -8.99% in 10y test
    'Materials',              # -3.07% in 10y test
    'Health Care',            # -0.28% in 10y test
    'Communication Services',  # +0.06% in 10y test
]

# Sector-specific position limits (focus on winners)
SECTOR_MAX_POSITIONS = {
    'Financials': 4,              # +9.94% winner
    'Utilities': 3,               # +9.54% winner
    'Information Technology': 4,   # +8.53% winner
    'Energy': 3,                  # Historically strong
    'Real Estate': 2,             # +6.74% winner
    'Consumer Discretionary': 2,
    'Industrials': 2,
    'Consumer Staples': 1,
    'Default': 1,
}

ENABLE_SECTOR_DIVERSIFICATION = True

# ============================================================================
# RISK MANAGEMENT - AGGRESSIVE (Tight SL, High TP)
# ============================================================================
USE_STOP_LOSS = True
USE_TAKE_PROFIT = True

# Regime-specific Stop Loss multipliers (ATR-based)
REGIME_STOP_LOSS_MULTIPLIER = {
    'BULL_STRONG': 0.5,      # Very tight (50% ATR)
    'BULL_WEAK': 0.6,        # Tight
    'NEUTRAL_BULLISH': 0.7,
    'NEUTRAL_BEARISH': 0.8,
    'BEAR_WEAK': 0.9,
    'BEAR_STRONG': 1.0,
    'CRISIS': 1.0,
}

# Regime-specific Take Profit multipliers (ATR-based)
REGIME_TAKE_PROFIT_MULTIPLIER = {
    'BULL_STRONG': 4.0,      # High TP (let winners run)
    'BULL_WEAK': 3.5,
    'NEUTRAL_BULLISH': 3.0,
    'NEUTRAL_BEARISH': 2.5,
    'BEAR_WEAK': 2.0,
    'BEAR_STRONG': 1.5,
    'CRISIS': 1.0,
}

# Default SL/TP if regime unknown
DEFAULT_STOP_MULTIPLIER = 0.8
DEFAULT_TP_MULTIPLIER = 2.5

# Minimum hold days per regime
REGIME_MIN_HOLD_DAYS = {
    'BULL_STRONG': 5,
    'BULL_WEAK': 7,
    'NEUTRAL_BULLISH': 10,
    'NEUTRAL_BEARISH': 10,
    'BEAR_WEAK': 5,
    'BEAR_STRONG': 3,
    'CRISIS': 0,
}

DEFAULT_MIN_HOLD_DAYS = 7

# ============================================================================
# TRANSACTION COSTS - REALISTIC
# ============================================================================
SLIPPAGE_PCT = 0.005  # 0.5% slippage per trade
COMMISSION_PCT = 0.001  # 0.1% commission per trade
TOTAL_COST_PER_SIDE = SLIPPAGE_PCT + COMMISSION_PCT  # 0.6% per side

# ============================================================================
# ADAPTIVE SIZING - DISABLED (Keep it simple)
# ============================================================================
ADAPTIVE_POSITION_SIZING = {
    'enabled': False,  # Disabled for aggressive setup
}

# ============================================================================
# REGIME STRATEGIES - FROM LIVE SYSTEM
# ============================================================================
REGIME_STRATEGIES = {
    'BULL_STRONG': {
        'max_positions': 10,
        'position_size_multiplier': 1.0,  # Use base 6%
        'stop_multiplier': 0.5,
        'tp_multiplier': 4.0,
        'min_hold_days': 5,
        'gate_adjustment': 0.95,  # Slightly lower threshold
    },
    'BULL_WEAK': {
        'max_positions': 8,
        'position_size_multiplier': 1.0,
        'stop_multiplier': 0.6,
        'tp_multiplier': 3.5,
        'min_hold_days': 7,
        'gate_adjustment': 1.0,
    },
    'NEUTRAL_BULLISH': {
        'max_positions': 6,
        'position_size_multiplier': 1.0,
        'stop_multiplier': 0.7,
        'tp_multiplier': 3.0,
        'min_hold_days': 10,
        'gate_adjustment': 1.0,
    },
    'NEUTRAL_BEARISH': {
        'max_positions': 4,
        'position_size_multiplier': 0.9,  # Slightly smaller
        'stop_multiplier': 0.8,
        'tp_multiplier': 2.5,
        'min_hold_days': 10,
        'gate_adjustment': 1.05,  # Higher threshold
    },
    'BEAR_WEAK': {
        'max_positions': 3,
        'position_size_multiplier': 0.8,
        'stop_multiplier': 0.9,
        'tp_multiplier': 2.0,
        'min_hold_days': 5,
        'gate_adjustment': 1.1,
    },
    'BEAR_STRONG': {
        'max_positions': 2,
        'position_size_multiplier': 0.7,
        'stop_multiplier': 1.0,
        'tp_multiplier': 1.5,
        'min_hold_days': 3,
        'gate_adjustment': 1.15,
    },
    'CRISIS': {
        'max_positions': 0,
        'position_size_multiplier': 0.0,
        'stop_multiplier': 1.0,
        'tp_multiplier': 1.0,
        'min_hold_days': 0,
        'gate_adjustment': 2.0,  # Effectively no entries
    },
}

# ============================================================================
# DATA PATHS
# ============================================================================
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEASONALITY_REPORTS = PROJECT_ROOT / "seasonality_reports"

# Price data
PRICE_CACHE_DIR = SEASONALITY_REPORTS / "runs" / "2025-10-04_0903" / "price_cache"
REGIME_PRICE_CACHE = SEASONALITY_REPORTS / "price_cache"

# Vintage seasonality data
VINTAGE_DIR = SEASONALITY_REPORTS / "vintage"

# Universe
CONSTITUENTS_CSV = SEASONALITY_REPORTS / "constituents_raw.csv"

# Output
BACKTEST_RESULTS_DIR = SEASONALITY_REPORTS / "backtest_results"

# ============================================================================
# REGIME DETECTION SETTINGS
# ============================================================================
REGIME_TICKERS = [
    'SPY', 'QQQ', 'IWM',  # Equity indices
    '^SPX', '^VIX',       # Market indicators
    'GLD', 'TLT',         # Safe havens
    'HYG', 'LQD',         # Credit spreads
]

# ============================================================================
# PERFORMANCE & VISUALIZATION
# ============================================================================
SAVE_PLOTS = True
SAVE_TRADE_HISTORY = True
SAVE_REGIME_HISTORY = True

# Benchmarks for comparison
BENCHMARKS = ['SPY', 'QQQ']

# ============================================================================
# UNIVERSE SETTINGS
# ============================================================================
MAX_UNIVERSE_SIZE = 500  # Maximum number of tickers
MIN_TRADING_DAYS = 252   # Require at least 1 year of data

# ============================================================================
# EXECUTION SETTINGS
# ============================================================================
SHOW_PROGRESS_BAR = True
VERBOSE = True
DEBUG = False

# Entry timing
ENTRY_METHOD = 'open_with_gap'  # 'open', 'open_with_gap', or 'close'
GAP_MIN = -0.02  # -2% gap
GAP_MAX = 0.02   # +2% gap

# Exit timing
EXIT_METHOD = 'intraday_check'  # Check SL/TP against high/low

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================
REQUIRE_ATR = True  # Require ATR calculation for SL/TP
REQUIRE_SECTOR = False  # Don't skip if sector missing
REQUIRE_REGIME = True  # Require regime detection

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_TO_FILE = True
LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# OPTIMIZATION SETTINGS (Not used in this run)
# ============================================================================
OPTIMIZATION_MODE = False
WALK_FORWARD_SPLITS = 4
TRAIN_TEST_RATIO = 0.75

# ============================================================================
# NOTES
# ============================================================================
"""
AGGRESSIVE SETUP v3.0 - Configuration Notes:

TARGET: Beat SPY (+50% 2y) and QQQ (+70% 2y)
EXPECTED: +80-120% over 2 years

KEY CHANGES FROM v2.0:
1. Position size: 6% (vs 5%)
2. Gate alpha: 0.10 (vs 0.15) = more trades
3. Max positions: 10 in BULL_STRONG (vs 20)
4. Stop loss: 0.5-1.0x ATR (vs 0.8-1.8x) = tighter
5. Take profit: 4.0-1.5x ATR (vs 2.5-0.8x) = higher
6. Sector blacklist: 4 weak sectors removed
7. Costs: 0.6% per side (realistic)
8. Adaptive sizing: OFF (simpler)

RISK PROFILE:
- Expected Max DD: 12-15% (vs 14.7% in v2.0)
- Expected Sharpe: 1.5-1.8
- Expected Win Rate: 45-50%
- Expected Trades: ~400-500 (2 years)

COMPATIBILITY WITH LIVE SYSTEM:
- Same regime detection logic (regime_detector.py)
- Same signal generation (ml_unified_pipeline.py)
- Different parameters (more aggressive)
- Live system should use more conservative settings after validation
"""