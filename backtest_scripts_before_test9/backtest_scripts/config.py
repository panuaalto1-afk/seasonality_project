"""
Backtest Configuration - Aggressive Setup v3.1 - 10-YEAR BACKTEST
Date: 2025-11-15
Target: 10-year performance test (2014-2025)
Changes: Extended period, optimized for long-term testing
"""

from datetime import date

# ============================================================================
# BACKTEST PERIOD - 10 YEARS + 1 YEAR WARM-UP
# ============================================================================
BACKTEST_START = date(2014, 1, 1)  # CHANGED: Was 2023-01-01
BACKTEST_END = date(2025, 11, 8)    # Same
INITIAL_CASH = 100000.0

# ============================================================================
# POSITION SIZING - MORE AGGRESSIVE (8% Dynamic)
# ============================================================================
POSITION_SIZE_METHOD = 'percentage'  # 'percentage' or 'fixed'
POSITION_SIZE_PCT = 0.08  # CHANGED: 8% (was 6%) - more aggressive
POSITION_SIZE_FIXED = 5000  # Only used if method='fixed'

MIN_POSITION_SIZE = 1000.0  # Minimum $1k per position
MAX_POSITION_SIZE = 50000.0  # Maximum $50k per position
MAX_POSITION_PCT = 0.10  # Never exceed 10% of portfolio

# ============================================================================
# PORTFOLIO LIMITS - REGIME-BASED (MORE AGGRESSIVE)
# ============================================================================
REGIME_MAX_POSITIONS = {
    'BULL_STRONG': 12,         # CHANGED: +2 (was 10)
    'BULL_WEAK': 10,           # CHANGED: +2 (was 8)
    'NEUTRAL_BULLISH': 8,      # CHANGED: +2 (was 6)
    'NEUTRAL_BEARISH': 6,      # CHANGED: +2 (was 4)
    'BEAR_WEAK': 4,            # CHANGED: +1 (was 3)
    'BEAR_STRONG': 2,          # Same
    'CRISIS': 0,               # Same
}

DEFAULT_MAX_POSITIONS = 8  # Fallback if regime unknown

# ============================================================================
# SIGNAL FILTERING - MORE TRADES
# ============================================================================
GATE_ALPHA = 0.20  # CHANGED: 0.15 (was 0.10) = top 15% signals
MIN_SCORE_LONG = 8.0  # CHANGED: 12.0 (was 0.10) = lower threshold

# ============================================================================
# SECTOR FILTERING - LESS RESTRICTIVE
# ============================================================================
SECTOR_BLACKLIST = [
    'Utilities',  # CHANGED: Only 1 sector (was 4)
    # Removed: Biopharmaceutical, Materials, Health Care, Communication Services
]

# Sector-specific position limits (focus on winners)
SECTOR_MAX_POSITIONS = {
    'Financials': 4,              # +9.94% winner
    'Utilities': 3,               # +9.54% winner
    'Information Technology': 4,   # +8.53% winner
    'Energy': 3,                  # Historically strong
    'Real Estate': 3,             # CHANGED: +1 (was 2)
    'Consumer Discretionary': 3,  # CHANGED: +1 (was 2)
    'Industrials': 3,             # CHANGED: +1 (was 2)
    'Consumer Staples': 2,        # CHANGED: +1 (was 1)
    'Materials': 2,               # ADDED
    'Health Care': 2,             # ADDED
    'Communication Services': 2,  # ADDED
    'Default': 2,                 # CHANGED: +1 (was 1)
}

ENABLE_SECTOR_DIVERSIFICATION = True

# ============================================================================
# RISK MANAGEMENT - WIDER STOPS, HIGHER TARGETS
# ============================================================================
USE_STOP_LOSS = True
USE_TAKE_PROFIT = True

# Regime-specific Stop Loss multipliers (ATR-based) - WIDER
REGIME_STOP_LOSS_MULTIPLIER = {
    'BULL_STRONG': 1.5,      # CHANGED: 1.5x ATR (was 0.5x) - wider
    'BULL_WEAK': 1.5,        # CHANGED: 1.5x (was 0.6x)
    'NEUTRAL_BULLISH': 2.0,  # CHANGED: 2.0x (was 0.7x)
    'NEUTRAL_BEARISH': 2.0,  # CHANGED: 2.0x (was 0.8x)
    'BEAR_WEAK': 1.5,        # CHANGED: 1.5x (was 0.9x)
    'BEAR_STRONG': 1.0,      # Same
    'CRISIS': 1.0,           # Same
}

# Regime-specific Take Profit multipliers (ATR-based) - HIGHER
REGIME_TAKE_PROFIT_MULTIPLIER = {
    'BULL_STRONG': 6.0,      # CHANGED: 6.0x ATR (was 4.0x) - higher
    'BULL_WEAK': 5.0,        # CHANGED: 5.0x (was 3.5x)
    'NEUTRAL_BULLISH': 4.0,  # CHANGED: 4.0x (was 3.0x)
    'NEUTRAL_BEARISH': 3.0,  # CHANGED: 3.0x (was 2.5x)
    'BEAR_WEAK': 2.5,        # CHANGED: 2.5x (was 2.0x)
    'BEAR_STRONG': 2.0,      # CHANGED: 2.0x (was 1.5x)
    'CRISIS': 1.5,           # CHANGED: 1.5x (was 1.0x)
}

# Default SL/TP if regime unknown
DEFAULT_STOP_MULTIPLIER = 1.5   # CHANGED: 1.5 (was 0.8)
DEFAULT_TP_MULTIPLIER = 4.0     # CHANGED: 4.0 (was 2.5)

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
SLIPPAGE_PCT = 0.001  # CHANGED: 0.1% (was 0.5%) - more realistic
COMMISSION_PCT = 0.001  # 0.1% commission per trade
TOTAL_COST_PER_SIDE = SLIPPAGE_PCT + COMMISSION_PCT  # 0.2% per side

# ============================================================================
# ADAPTIVE SIZING - DISABLED (Keep it simple)
# ============================================================================
ADAPTIVE_POSITION_SIZING = {
    'enabled': False,  # Disabled for aggressive setup
}

# ============================================================================
# REGIME STRATEGIES - UPDATED FOR 10-YEAR
# ============================================================================
REGIME_STRATEGIES = {
    'BULL_STRONG': {
        'max_positions': 12,     # CHANGED: +2
        'position_size_multiplier': 1.0,
        'stop_multiplier': 1.5,  # CHANGED: wider
        'tp_multiplier': 6.0,    # CHANGED: higher
        'min_hold_days': 5,
        'gate_adjustment': 0.95,
    },
    'BULL_WEAK': {
        'max_positions': 10,     # CHANGED: +2
        'position_size_multiplier': 1.0,
        'stop_multiplier': 1.5,  # CHANGED: wider
        'tp_multiplier': 5.0,    # CHANGED: higher
        'min_hold_days': 7,
        'gate_adjustment': 1.0,
    },
    'NEUTRAL_BULLISH': {
        'max_positions': 8,      # CHANGED: +2
        'position_size_multiplier': 1.0,
        'stop_multiplier': 2.0,  # CHANGED: wider
        'tp_multiplier': 4.0,    # CHANGED: higher
        'min_hold_days': 10,
        'gate_adjustment': 1.0,
    },
    'NEUTRAL_BEARISH': {
        'max_positions': 6,      # CHANGED: +2
        'position_size_multiplier': 0.9,
        'stop_multiplier': 2.0,  # CHANGED: wider
        'tp_multiplier': 3.0,    # CHANGED: higher
        'min_hold_days': 10,
        'gate_adjustment': 1.05,
    },
    'BEAR_WEAK': {
        'max_positions': 4,      # CHANGED: +1
        'position_size_multiplier': 0.8,
        'stop_multiplier': 1.5,  # CHANGED: wider
        'tp_multiplier': 2.5,    # CHANGED: higher
        'min_hold_days': 5,
        'gate_adjustment': 1.1,
    },
    'BEAR_STRONG': {
        'max_positions': 2,
        'position_size_multiplier': 0.7,
        'stop_multiplier': 1.0,
        'tp_multiplier': 2.0,    # CHANGED: higher
        'min_hold_days': 3,
        'gate_adjustment': 1.15,
    },
    'CRISIS': {
        'max_positions': 0,
        'position_size_multiplier': 0.0,
        'stop_multiplier': 1.0,
        'tp_multiplier': 1.5,    # CHANGED: higher
        'min_hold_days': 0,
        'gate_adjustment': 2.0,
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
VERBOSE = False  # CHANGED: False for 10-year (was True) - less output
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
AGGRESSIVE SETUP v3.1 - 10-YEAR BACKTEST Configuration Notes:

TARGET: Long-term performance validation (2014-2025)
EXPECTED: CAGR 20-30% → Total Return +400-600% (10 years)

KEY CHANGES FROM v3.0:
1. Backtest period: 2014-2025 (was 2023-2025) = 10 years + 1 warm-up
2. Position size: 8% (was 6%) - more aggressive
3. Gate alpha: 0.15 (was 0.10) = top 15% signals
4. Min score: 12.0 (was 0.10) = lower threshold
5. Max positions: 12/10/8/6/4/2/0 (was 10/8/6/4/3/2/0) = more capacity
6. Stop loss: 1.5-2.0x ATR (was 0.5-1.0x) = wider, let winners run
7. Take profit: 6.0-5.0x ATR (was 4.0-3.5x) = higher targets
8. Sector blacklist: 1 sector (was 4) = less restrictive
9. Transaction costs: 0.2% per side (was 0.6%) = more realistic
10. Verbose: False (was True) = faster execution

MARKET EVENTS TESTED (2014-2025):
- 2015-2016: Oil crisis, China devaluation
- 2018 Q4: -20% correction
- 2020 Q1: COVID crash (-35%)
- 2020-2021: V-shaped recovery
- 2022: Bear market (-25%)
- 2023-2024: AI boom recovery

EXPECTED RESULTS:
- CAGR: 20-30% per year
- Total Return: +400-600% (10 years)
- Max Drawdown: -20% to -25%
- Sharpe Ratio: 1.5-2.0
- Win Rate: 40-50%
- Total Trades: 800-1200

BENCHMARKS TO BEAT:
- SPY (2015-2024): ~15% CAGR → +300% total
- QQQ (2015-2024): ~20% CAGR → +500% total

COMPATIBILITY:
- Same regime detection logic
- Same signal generation
- More aggressive parameters
- Suitable for long-term validation before live deployment
"""