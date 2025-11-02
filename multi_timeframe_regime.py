#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_timeframe_regime.py - Multi-Timeframe Regime Analysis

Analysoi markkinaregimen kolmella eri aikajänteellä:
- Daily: 5-60 päivän momentum (lyhyt aikaväli)
- Weekly: 4-13 viikon trendi (keskipitkä)
- Monthly: 3-12 kuukauden sykli (pitkä)

Palauttaa hierarkisen regime-rakenteen joka kertoo:
- Onko lyhyen aikavälin liike trendiin vai vastakkaissuuntaan
- Onko kyseessä korjausliike vai trendin muutos
- Miten vahva trend on (kaikki timeframet samaan suuntaan)
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from regime_detector import RegimeDetector


class MultiTimeframeRegime:
    """
    Analysoi regime usealla aikajänteellä
    """
    
    def __init__(self, 
                 macro_price_cache_dir: str = "seasonality_reports/price_cache",
                 equity_price_cache_dir: Optional[str] = None):
        self.detector = RegimeDetector(
            macro_price_cache_dir=macro_price_cache_dir,
            equity_price_cache_dir=equity_price_cache_dir
        )
        
        # Timeframe parametrit (päivissä lookback)
        self.timeframes = {
            'daily': {
                'lookback': 60,      # 3 kuukautta
                'label': 'Daily',
                'weight': 0.50       # Painotetaan nykyhetkeä eniten
            },
            'weekly': {
                'lookback': 180,     # 6 kuukautta (26 viikkoa)
                'label': 'Weekly',
                'weight': 0.30
            },
            'monthly': {
                'lookback': 365,     # 1 vuosi (12 kuukautta)
                'label': 'Monthly',
                'weight': 0.20
            }
        }
    
    def _aggregate_regime_score(self, 
                                 start_date: str, 
                                 end_date: str,
                                 sample_interval: int = 5) -> Dict:
        """
        Laske aggregoitu regime score aikavälille
        
        Args:
            start_date: Aloituspäivä (YYYY-MM-DD)
            end_date: Lopetuspäivä (YYYY-MM-DD)
            sample_interval: Näytteen ottotiheys päivissä
        
        Returns:
            {
                'avg_score': 0.25,
                'regime': 'NEUTRAL_BULLISH',
                'confidence': 0.65,
                'trend_strength': 0.70  # Kuinka yhdenmukainen periodi
            }
        """
        # Parse päivämäärät
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Kerää näytteitä
        samples = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            try:
                # Hae regime tälle päivälle
                result = self.detector.detect_regime(date=current_dt.strftime("%Y-%m-%d"))
                samples.append({
                    'date': current_dt,
                    'score': result['composite_score'],
                    'regime': result['regime'],
                    'confidence': result['confidence']
                })
            except:
                pass
            
            # Seuraava sample
            current_dt += timedelta(days=sample_interval)
        
        if not samples:
            return {
                'avg_score': 0.0,
                'regime': 'NEUTRAL_BULLISH',
                'confidence': 0.0,
                'trend_strength': 0.0
            }
        
        # Laske statistiikat
        df = pd.DataFrame(samples)
        
        avg_score = df['score'].mean()
        avg_confidence = df['confidence'].mean()
        
        # Trend strength = kuinka yhdensuuntainen (std pienempi = vahvempi)
        score_std = df['score'].std()
        trend_strength = 1.0 - min(score_std, 1.0)  # 0-1
        
        # Määritä regime avg_scoren perusteella
        regime = self.detector._classify_regime(avg_score)
        
        return {
            'avg_score': float(avg_score),
            'regime': regime,
            'confidence': float(avg_confidence),
            'trend_strength': float(trend_strength),
            'samples': len(samples)
        }
    
    def detect_multi_timeframe(self, date: Optional[str] = None) -> Dict:
        """
        Tunnista regime kaikilla timeframeilla
        
        Returns:
            {
                'date': '2025-11-02',
                'daily': {...},
                'weekly': {...},
                'monthly': {...},
                'composite': {
                    'regime': 'BULL_STRONG',
                    'alignment': 'strong',  # 'strong', 'partial', 'weak', 'conflicted'
                    'confidence': 0.85
                },
                'interpretation': 'Strong bull trend across all timeframes'
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\n[MultiTimeframe] Analyzing regimes for {date}...")
        
        end_dt = datetime.strptime(date, "%Y-%m-%d")
        
        results = {}
        
        # Analysoi jokainen timeframe
        for tf_name, tf_config in self.timeframes.items():
            start_dt = end_dt - timedelta(days=tf_config['lookback'])
            
            print(f"  {tf_config['label']:8} ({tf_config['lookback']} days)...", end=' ')
            
            result = self._aggregate_regime_score(
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
                sample_interval=5 if tf_name == 'daily' else 10
            )
            
            result['timeframe'] = tf_name
            result['weight'] = tf_config['weight']
            results[tf_name] = result
            
            print(f"{result['regime']:17} (score: {result['avg_score']:+.3f}, strength: {result['trend_strength']:.2f})")
        
        # Laske composite regime
        composite = self._calculate_composite_regime(results)
        
        # Tulkinta
        interpretation = self._interpret_multi_timeframe(results, composite)
        
        output = {
            'date': date,
            'daily': results['daily'],
            'weekly': results['weekly'],
            'monthly': results['monthly'],
            'composite': composite,
            'interpretation': interpretation
        }
        
        print(f"\n  Composite: {composite['regime']} ({composite['alignment']} alignment, confidence: {composite['confidence']:.1%})")
        print(f"  → {interpretation}")
        
        return output
    
    def _calculate_composite_regime(self, results: Dict) -> Dict:
        """
        Yhdistä eri timeframet yhteen composite regimeen
        
        Logiikat:
        - Jos kaikki samaa → Strong alignment
        - Jos 2/3 samaa → Partial alignment
        - Jos kaikki eri suuntaan → Conflicted
        """
        # Laske painotettu score
        weighted_score = 0.0
        weighted_confidence = 0.0
        
        for tf_name, result in results.items():
            weighted_score += result['avg_score'] * result['weight']
            weighted_confidence += result['confidence'] * result['weight']
        
        # Määritä composite regime
        composite_regime = self.detector._classify_regime(weighted_score)
        
        # Analysoi alignment
        regimes = [r['regime'] for r in results.values()]
        
        # Simplified regime grouping (BULL / NEUTRAL / BEAR / CRISIS)
        def group_regime(regime: str) -> str:
            if 'BULL' in regime:
                return 'BULL'
            elif 'BEAR' in regime:
                return 'BEAR'
            elif 'CRISIS' in regime:
                return 'CRISIS'
            else:
                return 'NEUTRAL'
        
        grouped = [group_regime(r) for r in regimes]
        
        # Count alignment
        if grouped[0] == grouped[1] == grouped[2]:
            alignment = 'strong'
            confidence_mult = 1.2
        elif grouped[0] == grouped[1] or grouped[0] == grouped[2] or grouped[1] == grouped[2]:
            alignment = 'partial'
            confidence_mult = 1.0
        else:
            alignment = 'conflicted'
            confidence_mult = 0.7
        
        # Special case: Any CRISIS
        if 'CRISIS' in regimes:
            alignment = 'crisis'
            confidence_mult = 1.5
        
        final_confidence = min(weighted_confidence * confidence_mult, 1.0)
        
        return {
            'regime': composite_regime,
            'weighted_score': float(weighted_score),
            'alignment': alignment,
            'confidence': float(final_confidence)
        }
    
    def _interpret_multi_timeframe(self, results: Dict, composite: Dict) -> str:
        """
        Anna inhimillinen tulkinta tilanteesta
        """
        daily_regime = results['daily']['regime']
        weekly_regime = results['weekly']['regime']
        monthly_regime = results['monthly']['regime']
        
        alignment = composite['alignment']
        
        # Strong alignment
        if alignment == 'strong':
            if 'BULL' in composite['regime']:
                return "Strong bull trend across all timeframes - high conviction longs"
            elif 'BEAR' in composite['regime']:
                return "Strong bear trend across all timeframes - defensive positioning"
            else:
                return "Neutral across all timeframes - range-bound market"
        
        # Crisis
        elif alignment == 'crisis':
            return "CRISIS detected on one or more timeframes - CAPITAL PRESERVATION MODE"
        
        # Partial alignment
        elif alignment == 'partial':
            # Daily vs longer-term mismatch?
            daily_group = 'BULL' if 'BULL' in daily_regime else ('BEAR' if 'BEAR' in daily_regime else 'NEUTRAL')
            monthly_group = 'BULL' if 'BULL' in monthly_regime else ('BEAR' if 'BEAR' in monthly_regime else 'NEUTRAL')
            
            if daily_group != monthly_group:
                if monthly_group == 'BULL' and daily_group == 'BEAR':
                    return "Short-term pullback in long-term bull trend - buying opportunity"
                elif monthly_group == 'BEAR' and daily_group == 'BULL':
                    return "Dead cat bounce in long-term bear trend - avoid new longs"
            
            return "Partial alignment - mixed signals, selective positioning"
        
        # Conflicted
        else:
            return "Conflicted signals across timeframes - high uncertainty, reduce exposure"
    
    def get_trading_bias(self, multi_tf_result: Dict) -> Dict:
        """
        Palauta kaupankäynti-bias multi-timeframe analyysin perusteella
        
        Returns:
            {
                'bias': 'bullish' | 'bearish' | 'neutral' | 'defensive',
                'strength': 0.0-1.0,
                'recommendation': 'aggressive' | 'moderate' | 'conservative' | 'exit'
            }
        """
        alignment = multi_tf_result['composite']['alignment']
        regime = multi_tf_result['composite']['regime']
        confidence = multi_tf_result['composite']['confidence']
        
        # Crisis → Exit
        if alignment == 'crisis' or 'CRISIS' in regime:
            return {
                'bias': 'defensive',
                'strength': 1.0,
                'recommendation': 'exit'
            }
        
        # Strong alignment
        if alignment == 'strong':
            if 'BULL' in regime:
                return {
                    'bias': 'bullish',
                    'strength': confidence,
                    'recommendation': 'aggressive' if confidence > 0.7 else 'moderate'
                }
            elif 'BEAR' in regime:
                return {
                    'bias': 'bearish',
                    'strength': confidence,
                    'recommendation': 'defensive'
                }
            else:
                return {
                    'bias': 'neutral',
                    'strength': confidence,
                    'recommendation': 'conservative'
                }
        
        # Partial / Conflicted
        else:
            # Käytä monthly (pitkä aikaväli) pääsuuntana
            monthly_regime = multi_tf_result['monthly']['regime']
            
            if 'BULL' in monthly_regime:
                bias = 'bullish'
            elif 'BEAR' in monthly_regime:
                bias = 'bearish'
            else:
                bias = 'neutral'
            
            return {
                'bias': bias,
                'strength': confidence * 0.6,  # Alenna vahvuutta epäselvässä tilanteessa
                'recommendation': 'conservative'
            }


# ==================== TESTAUSFUNKTIO ====================

def test_multi_timeframe():
    """Testaa multi-timeframe regime"""
    print("\n" + "="*80)
    print("🧪 TESTING MULTI-TIMEFRAME REGIME")
    print("="*80)
    
    # Luo analyzer
    mtf = MultiTimeframeRegime(
        macro_price_cache_dir="seasonality_reports/price_cache"
    )
    
    # Testaa detection
    result = mtf.detect_multi_timeframe()
    
    # Näytä tulokset
    print("\n" + "="*80)
    print("MULTI-TIMEFRAME REGIME ANALYSIS")
    print("="*80)
    print(f"\nDate: {result['date']}")
    
    print(f"\n{'Timeframe':<10} {'Regime':<20} {'Score':>8} {'Strength':>10} {'Confidence':>12}")
    print("-" * 80)
    
    for tf_name in ['daily', 'weekly', 'monthly']:
        tf = result[tf_name]
        print(f"{tf_name.upper():<10} {tf['regime']:<20} {tf['avg_score']:>+8.3f} {tf['trend_strength']:>9.1%} {tf['confidence']:>11.1%}")
    
    print("\n" + "-" * 80)
    comp = result['composite']
    print(f"{'COMPOSITE':<10} {comp['regime']:<20} {comp['weighted_score']:>+8.3f} {comp['alignment']:>10} {comp['confidence']:>11.1%}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print(f"{result['interpretation']}")
    
    # Trading bias
    print("\n" + "="*80)
    print("TRADING BIAS:")
    print("="*80)
    bias = mtf.get_trading_bias(result)
    print(f"Bias:           {bias['bias'].upper()}")
    print(f"Strength:       {bias['strength']:.1%}")
    print(f"Recommendation: {bias['recommendation'].upper()}")
    print("="*80 + "\n")
    
    print("✅ MULTI-TIMEFRAME TEST COMPLETE!")


if __name__ == "__main__":
    test_multi_timeframe()