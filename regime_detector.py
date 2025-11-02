#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_detector.py - Market Regime Detection System

Analysoi markkinaregimen 5 komponentin perusteella:
1. Equity momentum (SPY, QQQ, IWM)
2. Volatility (realized vol)
3. Credit spreads (HYG vs LQD)
4. Safe haven flows (GLD, TLT)
5. Market breadth (correlation)

Returns regime classification:
- BULL_STRONG, BULL_WEAK
- NEUTRAL_BULLISH, NEUTRAL_BEARISH
- BEAR_WEAK, BEAR_STRONG, CRISIS
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


class RegimeDetector:
    """
    Tunnistaa markkinaregimen monipuolisella analyysillä
    
    Käyttää 5 komponenttia:
    - Equity: SPY, QQQ, IWM momentum
    - Volatility: Realized volatility
    - Credit: HYG/LQD spread
    - Safe Haven: GLD, TLT flows
    - Breadth: Market correlation
    """
    
    def __init__(self, 
                 macro_price_cache_dir: str = "seasonality_reports/price_cache",
                 equity_price_cache_dir: Optional[str] = None):
        """
        Args:
            macro_price_cache_dir: Kansio jossa SPY, QQQ, IWM, GLD, TLT, HYG, LQD
            equity_price_cache_dir: Kansio jossa yksittäiset osakkeet (breadth-laskentaa varten)
        """
        self.macro_cache = Path(macro_price_cache_dir)
        self.equity_cache = Path(equity_price_cache_dir) if equity_price_cache_dir else None
        
        # Tarkista että makro-cache löytyy
        if not self.macro_cache.exists():
            print(f"[RegimeDetector] Warning: macro_price_cache_dir not found: {self.macro_cache}")
        
        # Regime thresholds
        self.thresholds = {
            'BULL_STRONG': 0.50,
            'BULL_WEAK': 0.25,
            'NEUTRAL_BULLISH': 0.0,
            'NEUTRAL_BEARISH': -0.25,
            'BEAR_WEAK': -0.50,
            'BEAR_STRONG': -0.75,
            'CRISIS': -1.0
        }
    
    def _load_prices(self, ticker: str, lookback_days: int = 60) -> Optional[pd.DataFrame]:
        """
        Lataa hinnat makro-cache:sta
        
        Returns:
            DataFrame with columns: date, close
        """
        csv_path = self.macro_cache / f"{ticker}.csv"
        
        if not csv_path.exists():
            print(f"[RegimeDetector] Warning: {ticker}.csv not found in {self.macro_cache}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            
            # Normalisoi sarakkeet
            date_col = None
            close_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'date' in col_lower or 'time' in col_lower:
                    date_col = col
                if 'close' in col_lower or 'adj' in col_lower:
                    close_col = col
            
            if date_col is None or close_col is None:
                print(f"[RegimeDetector] Warning: Could not find date/close columns in {ticker}.csv")
                return None
            
            df = df[[date_col, close_col]].copy()
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Rajoita lookback_days:iin
            if lookback_days:
                df = df.tail(lookback_days)
            
            return df
            
        except Exception as e:
            print(f"[RegimeDetector] Error loading {ticker}: {e}")
            return None
    
    def _calculate_momentum(self, df: pd.DataFrame, periods: List[int] = [5, 20, 60]) -> Dict[str, float]:
        """Laske momentum eri periodeilla"""
        if df is None or len(df) < max(periods):
            return {f'mom{p}': 0.0 for p in periods}
        
        results = {}
        for period in periods:
            if len(df) >= period:
                pct_change = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1)
                results[f'mom{period}'] = float(pct_change)
            else:
                results[f'mom{period}'] = 0.0
        
        return results
    
    def _calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Laske realized volatility"""
        if df is None or len(df) < window:
            return 0.0
        
        returns = df['close'].pct_change().dropna()
        vol = returns.tail(window).std() * np.sqrt(252)  # Annualized
        return float(vol)
    
    # ==================== COMPONENT 1: EQUITY MOMENTUM ====================
    def _analyze_equity_momentum(self, date: str) -> Dict:
        """
        Analysoi equity-markkinoiden momentum
        
        Käyttää SPY, QQQ, IWM:ää
        """
        tickers = ['SPY', 'QQQ', 'IWM']
        weights = [0.5, 0.3, 0.2]  # SPY painotettu eniten
        
        composite_momentum = 0.0
        details = {}
        
        for ticker, weight in zip(tickers, weights):
            df = self._load_prices(ticker, lookback_days=120)
            
            if df is not None:
                mom = self._calculate_momentum(df)
                details[ticker] = mom
                
                # Painotettu keskiarvo (20-päivän momentum pääpaino)
                composite_momentum += weight * mom.get('mom20', 0.0)
            else:
                details[ticker] = {'mom5': 0.0, 'mom20': 0.0, 'mom60': 0.0}
        
        # Normalisoi -1 to +1
        signal = np.tanh(composite_momentum * 5)  # Scale to [-1, 1]
        
        return {
            'signal': float(signal),
            'raw_momentum': float(composite_momentum),
            'details': details
        }
    
    # ==================== COMPONENT 2: VOLATILITY ====================
    def _analyze_volatility(self, date: str) -> Dict:
        """
        Analysoi volatiliteetti (SPY)
        
        Korkea vol = negatiivinen (risk-off)
        Matala vol = positiivinen (risk-on)
        """
        df = self._load_prices('SPY', lookback_days=60)
        
        if df is None:
            return {'signal': 0.0, 'realized_vol': 0.0}
        
        vol = self._calculate_volatility(df, window=20)
        
        # Normalisoi: 10% vol = neutraali, >30% = -1, <5% = +1
        if vol < 0.05:
            signal = 1.0
        elif vol > 0.30:
            signal = -1.0
        else:
            # Linear interpolation
            signal = 1.0 - ((vol - 0.05) / 0.25) * 2.0
        
        return {
            'signal': float(signal),
            'realized_vol': float(vol)
        }
    
    # ==================== COMPONENT 3: CREDIT SPREADS ====================
    def _analyze_credit_spreads(self, date: str) -> Dict:
        """
        Analysoi credit spreadit (HYG vs LQD)
        
        Kapeneva spread = risk-on
        Levenevä spread = risk-off
        """
        hyg = self._load_prices('HYG', lookback_days=60)
        lqd = self._load_prices('LQD', lookback_days=60)
        
        if hyg is None or lqd is None:
            return {'signal': 0.0, 'spread_change': 0.0}
        
        # Laske spread (HYG/LQD ratio)
        merged = pd.merge(hyg, lqd, on='date', suffixes=('_hyg', '_lqd'))
        merged['spread'] = merged['close_hyg'] / merged['close_lqd']
        
        if len(merged) < 20:
            return {'signal': 0.0, 'spread_change': 0.0}
        
        # Spread change (20-päivän)
        spread_change = (merged['spread'].iloc[-1] / merged['spread'].iloc[-20] - 1)
        
        # Positiivinen spread change = risk-on (HYG nousee enemmän)
        # Negatiivinen = risk-off
        signal = np.tanh(spread_change * 20)  # Scale to [-1, 1]
        
        return {
            'signal': float(signal),
            'spread_change': float(spread_change)
        }
    
    # ==================== COMPONENT 4: SAFE HAVEN FLOWS ====================
    def _analyze_safe_haven(self, date: str) -> Dict:
        """
        Analysoi safe haven -virtoja (GLD, TLT)
        
        Vahva safe haven flow = risk-off
        Heikko flow = risk-on
        """
        gld = self._load_prices('GLD', lookback_days=60)
        tlt = self._load_prices('TLT', lookback_days=60)
        
        safe_haven_momentum = 0.0
        details = {}
        
        for ticker, df in [('GLD', gld), ('TLT', tlt)]:
            if df is not None:
                mom = self._calculate_momentum(df)
                details[ticker] = mom
                safe_haven_momentum += mom.get('mom20', 0.0)
            else:
                details[ticker] = {'mom5': 0.0, 'mom20': 0.0, 'mom60': 0.0}
        
        # Average
        safe_haven_momentum /= 2.0
        
        # Inverse: vahva safe haven = negatiivinen signaali
        signal = -np.tanh(safe_haven_momentum * 5)
        
        return {
            'signal': float(signal),
            'safe_haven_momentum': float(safe_haven_momentum),
            'details': details
        }
    
    # ==================== COMPONENT 5: MARKET BREADTH ====================
    def _analyze_breadth(self, date: str) -> Dict:
        """
        Analysoi markkinoiden breadth
        
        Käyttää equity_cache:a jos saatavilla
        Muuten fallback: SPY vs IWM correlation
        """
        # Jos ei equity cache, käytä SPY vs IWM
        spy = self._load_prices('SPY', lookback_days=60)
        iwm = self._load_prices('IWM', lookback_days=60)
        
        if spy is None or iwm is None:
            return {'signal': 0.0, 'correlation': 0.0}
        
        # Laske korrelaatio
        merged = pd.merge(spy, iwm, on='date', suffixes=('_spy', '_iwm'))
        
        if len(merged) < 20:
            return {'signal': 0.0, 'correlation': 0.0}
        
        spy_ret = merged['close_spy'].pct_change()
        iwm_ret = merged['close_iwm'].pct_change()
        
        corr = spy_ret.tail(20).corr(iwm_ret.tail(20))
        
        # Korkea korrelaatio = hyvä breadth = positiivinen
        signal = float(corr) if not np.isnan(corr) else 0.0
        
        return {
            'signal': signal,
            'correlation': float(corr) if not np.isnan(corr) else 0.0
        }
    
    # ==================== COMPOSITE REGIME ====================
    def _calculate_composite_score(self, components: Dict) -> Tuple[float, float]:
        """
        Laske kokonaispistemäärä ja luottamus
        
        Returns:
            (composite_score, confidence)
        """
        # Painot (summa = 1.0)
        weights = {
            'equity': 0.35,
            'volatility': 0.20,
            'credit': 0.20,
            'safe_haven': 0.15,
            'breadth': 0.10
        }
        
        composite = 0.0
        confidence_sum = 0.0
        
        for key, weight in weights.items():
            signal = components[key]['signal']
            composite += weight * signal
            
            # Confidence = kuinka kaukana nollasta (vahva signaali)
            confidence_sum += weight * abs(signal)
        
        return composite, confidence_sum
    
    def _classify_regime(self, composite_score: float) -> str:
        """Luokittele regime composite scoren perusteella"""
        if composite_score >= self.thresholds['BULL_STRONG']:
            return 'BULL_STRONG'
        elif composite_score >= self.thresholds['BULL_WEAK']:
            return 'BULL_WEAK'
        elif composite_score >= self.thresholds['NEUTRAL_BULLISH']:
            return 'NEUTRAL_BULLISH'
        elif composite_score >= self.thresholds['NEUTRAL_BEARISH']:
            return 'NEUTRAL_BEARISH'
        elif composite_score >= self.thresholds['BEAR_WEAK']:
            return 'BEAR_WEAK'
        elif composite_score >= self.thresholds['BEAR_STRONG']:
            return 'BEAR_STRONG'
        else:
            return 'CRISIS'
    
    def detect_regime(self, date: Optional[str] = None) -> Dict:
        """
        Tunnista markkinaregime
        
        Args:
            date: YYYY-MM-DD (oletus: tänään)
        
        Returns:
            {
                'date': '2025-11-02',
                'regime': 'BULL_STRONG',
                'composite_score': 0.65,
                'confidence': 0.85,
                'components': {
                    'equity': {...},
                    'volatility': {...},
                    ...
                }
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"[RegimeDetector] Detecting regime for {date}...")
        
        # Analysoi komponentit
        components = {
            'equity': self._analyze_equity_momentum(date),
            'volatility': self._analyze_volatility(date),
            'credit': self._analyze_credit_spreads(date),
            'safe_haven': self._analyze_safe_haven(date),
            'breadth': self._analyze_breadth(date)
        }
        
        # Laske composite
        composite_score, confidence = self._calculate_composite_score(components)
        
        # Luokittele
        regime = self._classify_regime(composite_score)
        
        result = {
            'date': date,
            'regime': regime,
            'composite_score': composite_score,
            'confidence': confidence,
            'components': components
        }
        
        print(f"[RegimeDetector] Regime: {regime} (score: {composite_score:.3f}, confidence: {confidence:.2%})")
        
        # Tallenna historiaan
        self._save_to_history(result)
        
        return result
    
    def _save_to_history(self, result: Dict):
        """Tallenna regime historiaan"""
        history_path = Path("seasonality_reports/regime_history.csv")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Luo rivi
        row = {
            'date': result['date'],
            'regime': result['regime'],
            'composite_score': result['composite_score'],
            'confidence': result['confidence'],
            'equity_signal': result['components']['equity']['signal'],
            'volatility_signal': result['components']['volatility']['signal'],
            'credit_signal': result['components']['credit']['signal'],
            'safe_haven_signal': result['components']['safe_haven']['signal'],
            'breadth_signal': result['components']['breadth']['signal']
        }
        
        # Append tai luo uusi
        if history_path.exists():
            df = pd.read_csv(history_path)
            # Päivitä jos sama päivä on jo olemassa
            df = df[df['date'] != result['date']]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        
        df = df.sort_values('date')
        df.to_csv(history_path, index=False)


# ==================== TESTAUSFUNKTIO ====================

def test_detector():
    """Testaa RegimeDetector"""
    print("\n" + "="*80)
    print("🧪 TESTING RegimeDetector")
    print("="*80 + "\n")
    
    # Luo detector
    detector = RegimeDetector(
        macro_price_cache_dir="seasonality_reports/price_cache",
        equity_price_cache_dir=None  # Ei tarvita tässä testissä
    )
    
    # Testaa detection
    print("1️⃣ Testing regime detection...")
    result = detector.detect_regime()
    
    print("\n" + "="*80)
    print("REGIME ANALYSIS RESULTS")
    print("="*80)
    print(f"\nDate:       {result['date']}")
    print(f"Regime:     {result['regime']}")
    print(f"Score:      {result['composite_score']:.4f}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    print("\n--- COMPONENTS ---")
    for name, data in result['components'].items():
        print(f"{name.upper():15} signal: {data['signal']:+.3f}")
    
    print("="*80 + "\n")
    print("✅ DETECTOR TEST COMPLETE!")


if __name__ == "__main__":
    test_detector()