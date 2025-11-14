# backtest_scripts/config_inverse.py
"""
Configuration for Enhanced Backtest with Inverse ETFs
Extends base config with inverse ETF support
"""

from datetime import date
from .config import *  # Import base config

# ✅ INVERSE ETF MAPPING
INVERSE_ETF_MAPPING = {
    'SPY': 'SH',    # ProShares Short S&P 500 (-1x SPY)
    'QQQ': 'PSQ',   # ProShares Short QQQ (-1x QQQ)
    'IWM': 'RWM',   # ProShares Short Russell 2000 (-1x IWM)
}

# ✅ ENHANCED REGIME STRATEGIES WITH INVERSE ETFs
REGIME_STRATEGIES_INVERSE = {
    'BULL_STRONG': {
        'position_size_multiplier': 1.5,  # Increased from 1.2
        'stop_multiplier': 0.8,
        'tp_multiplier': 3.0,              # Increased from 2.5
        'max_positions': 12,               # Increased from 10
        'allow_inverse_etfs': False,
    },
    'BULL_WEAK': {
        'position_size_multiplier': 1.0,
        'stop_multiplier': 1.0,
        'tp_multiplier': 2.0,
        'max_positions': 8,
        'allow_inverse_etfs': False,
    },
    'NEUTRAL_BULLISH': {
        'position_size_multiplier': 0.9,
        'stop_multiplier': 1.0,
        'tp_multiplier': 1.5,
        'max_positions': 8,
        'allow_inverse_etfs': False,
    },
    'NEUTRAL_BEARISH': {
        'position_size_multiplier': 0.7,
        'stop_multiplier': 1.2,
        'tp_multiplier': 1.2,
        'max_positions': 6,
        'allow_inverse_etfs': True,        # ✅ Allow inverse ETFs
        'inverse_etf_allocation': 0.3,     # 30% of capital to inverse
    },
    'BEAR_WEAK': {
        'position_size_multiplier': 0.5,
        'stop_multiplier': 1.5,
        'tp_multiplier': 1.5,              # Increased from 1.0
        'max_positions': 4,
        'allow_inverse_etfs': True,        # ✅ Allow inverse ETFs
        'inverse_etf_allocation': 0.5,     # 50% to inverse
    },
    'BEAR_STRONG': {
        'position_size_multiplier': 0.3,
        'stop_multiplier': 1.8,
        'tp_multiplier': 2.0,              # Increased from 0.8
        'max_positions': 3,                # Increased from 2
        'allow_inverse_etfs': True,        # ✅ Allow inverse ETFs
        'inverse_etf_allocation': 0.7,     # 70% to inverse
    },
    'CRISIS': {
        'position_size_multiplier': 0.3,   # ✅ Changed from 0.0
        'stop_multiplier': 2.0,
        'tp_multiplier': 2.5,              # Increased from 0.5
        'max_positions': 2,                # ✅ Changed from 0
        'allow_inverse_etfs': True,        # ✅ Allow inverse ETFs
        'inverse_etf_allocation': 1.0,     # 100% to inverse (full defensive)
    },
}

# Inverse ETF trading parameters
INVERSE_ETF_PARAMS = {
    'stop_loss_pct': 0.08,     # 8% stop loss
    'take_profit_pct': 0.15,   # 15% take profit
    'max_hold_days': 30,       # Max 30 days hold
}