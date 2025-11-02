#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sector_rotation.py - Sector Rotation Strategy by Market Regime

Määrittää optimaalisen sektori-allokaation markkinaregimenin mukaan:

BULL Regimes (BULL_STRONG, BULL_WEAK):
  → Growth sectors: Technology, Consumer Discretionary, Communication Services
  → Cyclicals: Industrials, Materials
  
NEUTRAL Regimes (NEUTRAL_BULLISH, NEUTRAL_BEARISH):
  → Balanced: Financials, Healthcare, Mix of growth & defensive
  
BEAR Regimes (BEAR_WEAK, BEAR_STRONG):
  → Defensive sectors: Utilities, Consumer Staples, Healthcare
  → Quality focus: Low debt, stable earnings
  
CRISIS:
  → Safe havens: Gold (GLD), Treasuries (TLT), Cash
  → Minimal equity exposure
"""

from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np


class SectorRotation:
    """
    Hallitsee sektori-rotaatiota regime-perusteisesti
    """
    
    # GICS Sector mappings (11 standard sectors)
    SECTORS = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'ORCL', 'ACN', 'AMD', 'INTC', 'QCOM', 'TXN'],
        'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY', 'AMGN', 'CVS', 'MDT', 'GILD', 'CI'],
        'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG', 'MAR', 'F', 'GM', 'ROST', 'YUM'],
        'Consumer Staples': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'CL', 'MDLZ', 'KMB', 'GIS', 'KHC', 'HSY', 'SYY', 'STZ'],
        'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'ATVI', 'TTWO', 'PARA', 'OMC', 'FOX'],
        'Industrials': ['BA', 'HON', 'UNP', 'UPS', 'RTX', 'CAT', 'LMT', 'GE', 'DE', 'MMM', 'FDX', 'NSC', 'EMR', 'ETN', 'CSX'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI', 'WMB', 'HAL', 'DVN', 'HES', 'BKR'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED', 'ES', 'WEC', 'DTE', 'PPL', 'EIX'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'WELL', 'DLR', 'O', 'AVB', 'EQR', 'VTR', 'ARE', 'SBAC', 'INVH'],
        'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'DOW', 'PPG', 'CTVA', 'VMC', 'MLM', 'ALB', 'CF']
    }
    
    def __init__(self):
        # Build reverse lookup: ticker -> sector
        self.ticker_to_sector = {}
        for sector, tickers in self.SECTORS.items():
            for ticker in tickers:
                self.ticker_to_sector[ticker] = sector
        
        # Regime-specific sector preferences
        self.regime_sector_weights = self._define_regime_sector_weights()
    
    def _define_regime_sector_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Määrittää sektori-painotukset kullekin regimelle
        
        Returns:
            {
                'BULL_STRONG': {
                    'Technology': 0.30,
                    'Consumer Discretionary': 0.25,
                    ...
                },
                ...
            }
        """
        return {
            'BULL_STRONG': {
                'Technology': 0.30,
                'Consumer Discretionary': 0.25,
                'Communication Services': 0.15,
                'Industrials': 0.10,
                'Financials': 0.10,
                'Materials': 0.05,
                'Healthcare': 0.05,
                'Consumer Staples': 0.0,
                'Utilities': 0.0,
                'Energy': 0.0,
                'Real Estate': 0.0
            },
            
            'BULL_WEAK': {
                'Technology': 0.25,
                'Consumer Discretionary': 0.20,
                'Financials': 0.15,
                'Healthcare': 0.15,
                'Communication Services': 0.10,
                'Industrials': 0.10,
                'Materials': 0.05,
                'Consumer Staples': 0.0,
                'Utilities': 0.0,
                'Energy': 0.0,
                'Real Estate': 0.0
            },
            
            'NEUTRAL_BULLISH': {
                'Technology': 0.20,
                'Healthcare': 0.20,
                'Financials': 0.15,
                'Consumer Discretionary': 0.15,
                'Industrials': 0.10,
                'Consumer Staples': 0.10,
                'Communication Services': 0.05,
                'Utilities': 0.05,
                'Materials': 0.0,
                'Energy': 0.0,
                'Real Estate': 0.0
            },
            
            'NEUTRAL_BEARISH': {
                'Healthcare': 0.25,
                'Consumer Staples': 0.20,
                'Utilities': 0.15,
                'Financials': 0.15,
                'Technology': 0.10,
                'Consumer Discretionary': 0.10,
                'Industrials': 0.05,
                'Communication Services': 0.0,
                'Materials': 0.0,
                'Energy': 0.0,
                'Real Estate': 0.0
            },
            
            'BEAR_WEAK': {
                'Healthcare': 0.30,
                'Consumer Staples': 0.25,
                'Utilities': 0.20,
                'Financials': 0.10,
                'Technology': 0.10,
                'Consumer Discretionary': 0.05,
                'Industrials': 0.0,
                'Communication Services': 0.0,
                'Materials': 0.0,
                'Energy': 0.0,
                'Real Estate': 0.0
            },
            
            'BEAR_STRONG': {
                'Consumer Staples': 0.35,
                'Healthcare': 0.30,
                'Utilities': 0.30,
                'Financials': 0.05,
                'Technology': 0.0,
                'Consumer Discretionary': 0.0,
                'Industrials': 0.0,
                'Communication Services': 0.0,
                'Materials': 0.0,
                'Energy': 0.0,
                'Real Estate': 0.0
            },
            
            'CRISIS': {
                # In crisis, prefer safe havens (handled separately)
                # Minimal equity exposure
                'Consumer Staples': 0.40,
                'Healthcare': 0.30,
                'Utilities': 0.30,
                'Financials': 0.0,
                'Technology': 0.0,
                'Consumer Discretionary': 0.0,
                'Industrials': 0.0,
                'Communication Services': 0.0,
                'Materials': 0.0,
                'Energy': 0.0,
                'Real Estate': 0.0
            }
        }
    
    def get_sector(self, ticker: str) -> Optional[str]:
        """Palauta tickerin sektori"""
        return self.ticker_to_sector.get(ticker.upper())
    
    def get_sector_weight(self, sector: str, regime: str) -> float:
        """
        Palauta sektorin paino tietyssä regimessa
        
        Args:
            sector: Sektorin nimi
            regime: Regime (esim. 'BULL_STRONG')
        
        Returns:
            Weight 0.0-1.0
        """
        weights = self.regime_sector_weights.get(regime, {})
        return weights.get(sector, 0.0)
    
    def calculate_sector_bias(self, ticker: str, regime: str) -> float:
        """
        Laske sektori-bias multiplier tickerille
        
        Returns:
            Multiplier: 0.0 (avoid) - 2.0 (highly preferred)
        """
        sector = self.get_sector(ticker)
        
        if sector is None:
            # Unknown sector -> neutral
            return 1.0
        
        weight = self.get_sector_weight(sector, regime)
        
        # Convert weight to multiplier
        # weight 0.0 -> multiplier 0.0 (avoid)
        # weight 0.10 -> multiplier 1.0 (neutral)
        # weight 0.30 -> multiplier 2.0 (strong preference)
        
        if weight == 0.0:
            return 0.0  # Avoid this sector
        elif weight < 0.10:
            return 0.5  # Low preference
        elif weight < 0.20:
            return 1.0  # Neutral
        elif weight < 0.30:
            return 1.5  # Preference
        else:
            return 2.0  # Strong preference
    
    def apply_sector_rotation(self, 
                              signals_df: pd.DataFrame, 
                              regime: str,
                              top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Soveltaa sektori-rotaatiota signaaleihin
        
        Args:
            signals_df: DataFrame jossa ticker, score, jne.
            regime: Current regime
            top_n: Palauta top N (oletus: kaikki)
        
        Returns:
            Filtteröity ja uudelleen rankingoitu DataFrame
        """
        if signals_df.empty:
            return signals_df
        
        df = signals_df.copy()
        
        # Lisää sector info
        df['sector'] = df['ticker'].apply(self.get_sector)
        df['sector_bias'] = df['ticker'].apply(lambda t: self.calculate_sector_bias(t, regime))
        
        # Laske adjusted score
        if 'composite_score' in df.columns:
            base_score = df['composite_score']
        elif 'score_long' in df.columns:
            base_score = df['score_long']
        else:
            base_score = 0.5  # Fallback
        
        df['sector_adjusted_score'] = base_score * df['sector_bias']
        
        # Filtteriä pois 0.0 bias (avoid sectors)
        df = df[df['sector_bias'] > 0.0]
        
        # Järjestä adjusted scoren mukaan
        df = df.sort_values('sector_adjusted_score', ascending=False)
        
        if top_n:
            df = df.head(top_n)
        
        return df
    
    def get_sector_summary(self, regime: str) -> str:
        """
        Palauta sektori-allokaation yhteenveto tekstinä
        """
        weights = self.regime_sector_weights.get(regime, {})
        
        # Järjestä suurimmasta pienimpään
        sorted_sectors = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        lines = [f"\n📊 SECTOR ALLOCATION - {regime}"]
        lines.append("=" * 60)
        
        for sector, weight in sorted_sectors:
            if weight > 0.0:
                bars = '█' * int(weight * 50)
                lines.append(f"{sector:25} {weight:5.1%} {bars}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def print_sector_rotation_strategy(self, regime: str):
        """Tulosta sektori-rotaatiostrategia regimelle"""
        print(self.get_sector_summary(regime))
        
        # Preferred sectors
        weights = self.regime_sector_weights.get(regime, {})
        preferred = [s for s, w in weights.items() if w >= 0.20]
        neutral = [s for s, w in weights.items() if 0.10 <= w < 0.20]
        avoid = [s for s, w in weights.items() if w == 0.0]
        
        print(f"\n✅ PREFERRED SECTORS (≥20%):")
        for sector in preferred:
            print(f"   - {sector}")
        
        if neutral:
            print(f"\n⚖️  NEUTRAL SECTORS (10-20%):")
            for sector in neutral:
                print(f"   - {sector}")
        
        if avoid:
            print(f"\n❌ AVOID SECTORS (0%):")
            for sector in avoid:
                print(f"   - {sector}")


# ==================== TESTAUSFUNKTIO ====================

def test_sector_rotation():
    """Testaa sector rotation"""
    print("\n" + "="*80)
    print("🧪 TESTING SECTOR ROTATION")
    print("="*80 + "\n")
    
    # Luo rotator
    rotator = SectorRotation()
    
    # Testaa eri regimes
    regimes = ['BULL_STRONG', 'NEUTRAL_BULLISH', 'BEAR_WEAK', 'CRISIS']
    
    for regime in regimes:
        rotator.print_sector_rotation_strategy(regime)
        print("\n")
    
    # Testaa signaalien filtteröintiä
    print("="*80)
    print("🧪 TESTING SIGNAL FILTERING")
    print("="*80 + "\n")
    
    # Mock signals
    test_signals = pd.DataFrame({
        'ticker': ['AAPL', 'WMT', 'NEE', 'JPM', 'XOM', 'BA', 'UNH', 'GOOGL'],
        'composite_score': [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    })
    
    print("Original signals (by composite_score):")
    print(test_signals[['ticker', 'composite_score']].to_string(index=False))
    
    # Test BULL_STRONG (prefer Tech)
    print("\n" + "-"*80)
    print("BULL_STRONG regime (prefer Tech/Consumer Discretionary):")
    filtered = rotator.apply_sector_rotation(test_signals, 'BULL_STRONG', top_n=5)
    print(filtered[['ticker', 'sector', 'sector_bias', 'composite_score', 'sector_adjusted_score']].to_string(index=False))
    
    # Test BEAR_WEAK (prefer Defensive)
    print("\n" + "-"*80)
    print("BEAR_WEAK regime (prefer Healthcare/Staples/Utilities):")
    filtered = rotator.apply_sector_rotation(test_signals, 'BEAR_WEAK', top_n=5)
    print(filtered[['ticker', 'sector', 'sector_bias', 'composite_score', 'sector_adjusted_score']].to_string(index=False))
    
    print("\n✅ SECTOR ROTATION TEST COMPLETE!")


if __name__ == "__main__":
    test_sector_rotation()