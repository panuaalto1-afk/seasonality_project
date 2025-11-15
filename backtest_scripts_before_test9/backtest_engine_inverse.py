# backtest_scripts/backtest_engine_inverse.py
"""
Enhanced Backtest Engine with Inverse ETF Support
Extends BacktestEngine with inverse ETF capabilities
"""

import os
from typing import Dict, Optional
from .backtest_engine import BacktestEngine
from .config_inverse import (
    REGIME_STRATEGIES_INVERSE,
    INVERSE_ETF_MAPPING,
    INVERSE_ETF_PARAMS
)
from .auto_decider_inverse import AutoDeciderInverse

class BacktestEngineInverse(BacktestEngine):
    """
    Enhanced backtest engine with inverse ETF support
    """
    
    def __init__(self, config_overrides: Optional[Dict] = None):
        """
        Initialize enhanced backtest engine
        
        Args:
            config_overrides: Optional config overrides
        """
        # ✅ SET INVERSE ATTRIBUTES BEFORE super().__init__()
        self.inverse_etf_mapping = INVERSE_ETF_MAPPING
        self.inverse_etf_params = INVERSE_ETF_PARAMS
        
        # Override regime strategies with inverse version
        if config_overrides is None:
            config_overrides = {}
        
        config_overrides['regime_strategies'] = REGIME_STRATEGIES_INVERSE
        
        # Call parent init (this will call _init_components)
        super().__init__(config_overrides)
        
        print("=" * 80)
        print("✅ INVERSE ETF BACKTEST ENGINE ACTIVE")
        print("=" * 80)
        print(f"Inverse ETFs: {list(INVERSE_ETF_MAPPING.values())}")
        print(f"CRISIS regime: max_positions={REGIME_STRATEGIES_INVERSE['CRISIS']['max_positions']}")
        print("=" * 80)
        print("")
    
    def _init_components(self):
        """Override to add inverse ETFs to universe and auto_decider"""
        # Call parent init first
        super()._init_components()
        
        # Add inverse ETFs to stock prices
        print("[INVERSE] Loading inverse ETF prices...")
        inverse_etfs = list(self.inverse_etf_mapping.values())
        
        for etf in inverse_etfs:
            if etf not in self.stock_prices:
                # Load from macro cache (same as SPY, QQQ)
                df = self.data_loader.load_macro_prices(
                    etf,
                    self.config['start_date'],
                    self.config['end_date']
                )
                
                if df is not None:
                    # Add OHLC columns (use close as proxy)
                    df['open'] = df['close']
                    df['high'] = df['close'] * 1.01  # Assume 1% intraday range
                    df['low'] = df['close'] * 0.99
                    df['volume'] = 0
                    
                    self.stock_prices[etf] = df
                    print(f"  ✓ Loaded {etf}: {len(df)} days")
                else:
                    print(f"  ✗ Failed to load {etf}")
        
        # Replace auto_decider with enhanced version
        print("[INVERSE] Initializing enhanced auto_decider...")
        self.auto_decider = AutoDeciderInverse(
            regime_strategies=self.config['regime_strategies'],
            inverse_etf_mapping=self.inverse_etf_mapping,
            inverse_etf_params=self.inverse_etf_params,
            stock_prices=self.stock_prices
        )
        
        print("[OK] Inverse ETF components initialized\n")


# Main entry point
if __name__ == "__main__":
    print("Running Enhanced Backtest with Inverse ETFs...")
    engine = BacktestEngineInverse()
    results = engine.run()
    print("\n✅ Backtest complete with inverse ETF support!")