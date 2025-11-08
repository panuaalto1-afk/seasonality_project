#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_strategies.py - Regime-specific Trading Strategies

Määrittää eri kaupankäyntistrategiat eri markkinaregimeille:

BULL_STRONG / BULL_WEAK:
  → Momentum strategy
  → Osta vahvimmat nousijat
  → Pitkät stop-tappiot (anna trendin jatkua)
  
NEUTRAL_BULLISH / NEUTRAL_BEARISH:
  → Balanced strategy
  → Momentum + laatu
  → Normaalit stop/TP
  
BEAR_WEAK / BEAR_STRONG:
  → Mean reversion / Counter-trend
  → Osta ylilyötyjä laatuosakkeita
  → Tiukat stop-tappiot
  
CRISIS:
  → Capital preservation
  → Ei uusia positioita
  → Sulje kaikki avoimet
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class RegimeStrategy:
    """
    Määrittää strategian parametrit regimen perusteella
    """
    
    def __init__(self, regime: str):
        self.regime = regime
        self.config = self._get_strategy_config()
    
    def _get_strategy_config(self) -> Dict:
        """
        Palauta strategian konfiguraatio regimen mukaan
        """
        configs = {
            'BULL_STRONG': {
                'strategy_type': 'momentum',
                'signal_weights': {
                    'momentum': 0.70,    # Painota momentumia
                    'quality': 0.20,
                    'value': 0.10
                },
                'min_momentum_pct': 0.10,  # Vähintään +10% momentum
                'max_volatility': 0.50,    # Max 50% vuosivol
                'position_sizing': 1.3,     # 130% normaalista
                'stop_multiplier': 1.5,     # Väljät stopit (anna trendin jatkua)
                'tp_multiplier': 2.0,       # Korkeat targetit
                'max_positions': 12,
                'min_ml_score': 0.70,
                'entry_style': 'aggressive'
            },
            
            'BULL_WEAK': {
                'strategy_type': 'momentum',
                'signal_weights': {
                    'momentum': 0.60,
                    'quality': 0.30,
                    'value': 0.10
                },
                'min_momentum_pct': 0.05,
                'max_volatility': 0.40,
                'position_sizing': 1.0,
                'stop_multiplier': 1.2,
                'tp_multiplier': 1.5,
                'max_positions': 10,
                'min_ml_score': 0.75,
                'entry_style': 'selective'
            },
            
            'NEUTRAL_BULLISH': {
                'strategy_type': 'balanced',
                'signal_weights': {
                    'momentum': 0.50,
                    'quality': 0.35,
                    'value': 0.15
                },
                'min_momentum_pct': 0.0,   # Salli negatiiviset jos laatu hyvä
                'max_volatility': 0.35,
                'position_sizing': 0.9,
                'stop_multiplier': 1.0,
                'tp_multiplier': 1.2,
                'max_positions': 8,
                'min_ml_score': 0.75,
                'entry_style': 'selective'
            },
            
            'NEUTRAL_BEARISH': {
                'strategy_type': 'defensive_quality',
                'signal_weights': {
                    'momentum': 0.30,
                    'quality': 0.50,       # Painota laatua
                    'value': 0.20
                },
                'min_momentum_pct': -0.05,  # Salli lievä negatiivinen
                'max_volatility': 0.30,
                'position_sizing': 0.7,
                'stop_multiplier': 0.9,
                'tp_multiplier': 1.0,
                'max_positions': 6,
                'min_ml_score': 0.80,
                'entry_style': 'conservative'
            },
            
            'BEAR_WEAK': {
                'strategy_type': 'mean_reversion',
                'signal_weights': {
                    'momentum': 0.20,
                    'quality': 0.60,       # Laatu kriittistä
                    'value': 0.20          # Arvo-osakkeet
                },
                'min_momentum_pct': -0.15,  # Salli ylilyönnit
                'max_volatility': 0.30,
                'position_sizing': 0.5,
                'stop_multiplier': 0.8,     # Tiukat stopit
                'tp_multiplier': 0.8,       # Pienet targetit
                'max_positions': 4,
                'min_ml_score': 0.85,
                'entry_style': 'very_conservative'
            },
            
            'BEAR_STRONG': {
                'strategy_type': 'defensive',
                'signal_weights': {
                    'momentum': 0.10,
                    'quality': 0.70,
                    'value': 0.20
                },
                'min_momentum_pct': -0.20,
                'max_volatility': 0.25,
                'position_sizing': 0.3,
                'stop_multiplier': 0.7,
                'tp_multiplier': 0.6,
                'max_positions': 2,
                'min_ml_score': 0.90,
                'entry_style': 'extreme_conservative'
            },
            
            'CRISIS': {
                'strategy_type': 'capital_preservation',
                'signal_weights': {
                    'momentum': 0.0,
                    'quality': 0.0,
                    'value': 0.0
                },
                'min_momentum_pct': 0.0,
                'max_volatility': 0.0,
                'position_sizing': 0.0,     # Ei uusia positioita
                'stop_multiplier': 0.5,     # Exit kaikki nopeasti
                'tp_multiplier': 0.5,
                'max_positions': 0,
                'min_ml_score': 1.0,        # Ei uusia entryjä
                'entry_style': 'no_entries'
            }
        }
        
        return configs.get(self.regime, configs['NEUTRAL_BULLISH'])
    
    def filter_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtteriä signaalit regimen mukaan
        
        Args:
            signals_df: DataFrame jossa mom20, vol20, ml_score, jne.
        
        Returns:
            Filtteröity DataFrame
        """
        if signals_df.empty:
            return signals_df
        
        df = signals_df.copy()
        
        # 1. Momentum-filtteri
        if 'mom20' in df.columns:
            min_mom = self.config['min_momentum_pct']
            df = df[df['mom20'] >= min_mom]
        
        # 2. Volatiliteetti-filtteri
        if 'vol20' in df.columns:
            max_vol = self.config['max_volatility']
            df = df[df['vol20'] <= max_vol]
        
        # 3. ML score -filtteri
        if 'ml_score' in df.columns:
            min_score = self.config['min_ml_score']
            df = df[df['ml_score'] >= min_score]
        
        return df
    
    def calculate_composite_score(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Laske composite score regimen signal weights mukaan
        
        Combines:
        - Momentum (mom20)
        - Quality (ml_score proxy)
        - Value (mean reversion signal)
        """
        if signals_df.empty:
            return signals_df
        
        df = signals_df.copy()
        weights = self.config['signal_weights']
        
        # Normalisoi komponentit 0-1
        if 'mom20' in df.columns:
            mom_norm = (df['mom20'] - df['mom20'].min()) / (df['mom20'].max() - df['mom20'].min() + 0.001)
        else:
            mom_norm = 0.5
        
        if 'ml_score' in df.columns:
            quality_norm = df['ml_score']
        else:
            quality_norm = 0.5
        
        # Value = inverse momentum (mean reversion)
        if 'mom20' in df.columns:
            value_norm = 1.0 - mom_norm
        else:
            value_norm = 0.5
        
        # Composite
        df['composite_score'] = (
            weights['momentum'] * mom_norm +
            weights['quality'] * quality_norm +
            weights['value'] * value_norm
        )
        
        return df
    
    def rank_signals(self, signals_df: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
        """
        Järjestä signaalit composite scoren mukaan
        
        Returns top N signals
        """
        if signals_df.empty:
            return signals_df
        
        # Laske composite score
        df = self.calculate_composite_score(signals_df)
        
        # Filtteriä
        df = self.filter_signals(df)
        
        # Järjestä
        df = df.sort_values('composite_score', ascending=False)
        
        # Rajoita määrää
        if top_n is None:
            top_n = self.config['max_positions']
        
        return df.head(top_n)
    
    def get_position_parameters(self, base_position_size: float = 1000.0) -> Dict:
        """
        Palauta position sizing parametrit
        
        Args:
            base_position_size: Perus position size ($)
        
        Returns:
            {
                'position_size': 1300,  # Adjusted
                'stop_multiplier': 1.5,
                'tp_multiplier': 2.0,
                'max_positions': 12
            }
        """
        return {
            'position_size': base_position_size * self.config['position_sizing'],
            'stop_multiplier': self.config['stop_multiplier'],
            'tp_multiplier': self.config['tp_multiplier'],
            'max_positions': self.config['max_positions'],
            'min_ml_score': self.config['min_ml_score'],
            'entry_style': self.config['entry_style']
        }
    
    def print_strategy(self):
        """Tulosta strategian tiedot"""
        print("\n" + "="*80)
        print(f"📊 REGIME STRATEGY: {self.regime}")
        print("="*80)
        print(f"Strategy Type:    {self.config['strategy_type']}")
        print(f"Entry Style:      {self.config['entry_style']}")
        print(f"\nSignal Weights:")
        for signal, weight in self.config['signal_weights'].items():
            print(f"  {signal:12} {weight:.1%}")
        print(f"\nFilters:")
        print(f"  Min Momentum:   {self.config['min_momentum_pct']:+.1%}")
        print(f"  Max Volatility: {self.config['max_volatility']:.1%}")
        print(f"  Min ML Score:   {self.config['min_ml_score']:.2f}")
        print(f"\nPosition Sizing:")
        print(f"  Size Multiplier: {self.config['position_sizing']:.1f}x")
        print(f"  Stop Multiplier: {self.config['stop_multiplier']:.1f}x")
        print(f"  TP Multiplier:   {self.config['tp_multiplier']:.1f}x")
        print(f"  Max Positions:   {self.config['max_positions']}")
        print("="*80 + "\n")


# ==================== TESTAUSFUNKTIO ====================

def test_strategies():
    """Testaa eri regime-strategioita"""
    print("\n" + "="*80)
    print("🧪 TESTING REGIME STRATEGIES")
    print("="*80 + "\n")
    
    # Testaa kaikki regimes
    regimes = [
        'BULL_STRONG', 'BULL_WEAK', 
        'NEUTRAL_BULLISH', 'NEUTRAL_BEARISH',
        'BEAR_WEAK', 'BEAR_STRONG', 'CRISIS'
    ]
    
    for regime in regimes:
        strategy = RegimeStrategy(regime)
        strategy.print_strategy()
    
    # Testaa signaalien filtteröintiä
    print("\n" + "="*80)
    print("🧪 TESTING SIGNAL FILTERING")
    print("="*80 + "\n")
    
    # Mock data
    test_signals = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMD'],
        'mom20': [0.15, 0.08, 0.05, 0.25, -0.03],
        'vol20': [0.25, 0.20, 0.18, 0.60, 0.35],
        'ml_score': [0.85, 0.78, 0.92, 0.65, 0.88]
    })
    
    print("Original signals:")
    print(test_signals.to_string(index=False))
    
    # BULL_STRONG strategy
    print("\n" + "-"*80)
    print("BULL_STRONG strategy:")
    strategy = RegimeStrategy('BULL_STRONG')
    filtered = strategy.rank_signals(test_signals, top_n=3)
    print(filtered[['ticker', 'mom20', 'vol20', 'ml_score', 'composite_score']].to_string(index=False))
    
    # BEAR_WEAK strategy
    print("\n" + "-"*80)
    print("BEAR_WEAK strategy:")
    strategy = RegimeStrategy('BEAR_WEAK')
    filtered = strategy.rank_signals(test_signals, top_n=3)
    print(filtered[['ticker', 'mom20', 'vol20', 'ml_score', 'composite_score']].to_string(index=False))
    
    print("\n✅ STRATEGY TESTS COMPLETE!")


if __name__ == "__main__":
    test_strategies()