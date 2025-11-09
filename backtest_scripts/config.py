from datetime import date

# Configurations for backtesting
BACKTEST_START = date(2015, 1, 1)  # Updated for 10-year backtest

REGIME_THRESHOLDS = {
    'BULL_STRONG': 0.30,  # Updated
    'BULL_WEAK': 0.15,    # Updated
    'NEUTRAL_BEARISH': -0.10  # Updated
}

REGIME_STRATEGIES = {
    'BULL_STRONG': {'min_hold_days': 5},  # Added min_hold_days
    'BULL_WEAK': {'min_hold_days': 5},    # Added min_hold_days
    'NEUTRAL_BULLISH': {'min_hold_days': 5},  # Added min_hold_days
    'NEUTRAL_BEARISH': {'min_hold_days': 5}, # Added min_hold_days
}

TP_MULTIPLIERS = {
    'BULL_STRONG': 4.0,  # Updated
    'BULL_WEAK': 3.0,    # Updated
    'NEUTRAL_BULLISH': 2.0   # Updated
}

POSITION_SIZE_MULTIPLIER = {
    'BULL_STRONG': 1.3,  # Updated
    'BULL_WEAK': 1.1     # Updated
}

MAX_POSITIONS = {
    'BULL_WEAK': 9  # Updated
}

"""
UPDATED: 2025-11-09 16:53 UTC - 10-year backtest + Optimized regime thresholds
"""

PARAM_SPACE = {
    'tp_multiplier': {'max': 5.0}  # Updated
}