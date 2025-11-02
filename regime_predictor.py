#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_predictor.py - ML-based Regime Transition Prediction

Ennustaa seuraavan markkinaregimen 1-5 päivää etukäteen käyttäen:
- Historiallista transitio-matriisia
- Komponentti-trendejä (equity, vol, credit, safe haven)
- LightGBM-mallia (jos asennettu)

Output:
- Todennäköisyys jokaiselle regimelle
- Varoitukset tulevista regime-muutoksista
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from collections import defaultdict

# LightGBM (asennetaan tarvittaessa: pip install lightgbm)
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[RegimePredictor] Warning: LightGBM not installed. Using fallback method.")


class RegimePredictor:
    """
    ML-pohjainen regime-ennustaja
    
    Käyttää kahta metodia:
    1. Historiallinen transitio-matriisi (yksinkertainen)
    2. LightGBM (edistynyt, jos asennettu)
    """
    
    def __init__(self, regime_history_path: str = "seasonality_reports/regime_history.csv"):
        self.history_path = Path(regime_history_path)
        self.regime_order = [
            'BULL_STRONG', 'BULL_WEAK', 'NEUTRAL_BULLISH', 
            'NEUTRAL_BEARISH', 'BEAR_WEAK', 'BEAR_STRONG', 'CRISIS'
        ]
        
        # Lataa historia
        self.history_df = self._load_history()
        
        # Laske transitio-matriisi
        self.transition_matrix = self._calculate_transition_matrix()
        
        # ML-malli (koulutetaan tarvittaessa)
        self.model = None
        self.feature_names = None
    
    def _load_history(self) -> pd.DataFrame:
        """Lataa regime-historia"""
        if not self.history_path.exists():
            print(f"[RegimePredictor] Warning: {self.history_path} not found")
            return pd.DataFrame()
        
        df = pd.read_csv(self.history_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df
    
    def _get_default_transition_matrix(self) -> pd.DataFrame:
        """
        Realistinen transitio-matriisi perustuen:
        - Taloustieteelliseen teoriaan
        - Historiallisiin markkina-sykleihin  
        - Regime-persistenssiin (regimes pysyvät melko stabiileina)
        
        Logiikka:
        - Regimes pysyvät yleensä samassa (45-60%)
        - Siirtymät yleensä viereisiin regimeihin
        - Harvoin hyppää yli 2+ tasoa (esim. BULL_STRONG → BEAR_STRONG)
        """
        
        # Manuaalisesti määritelty matriisi
        matrix_data = {
            #                 BS    BW    NB    NBE   BWE   BST   CRI
            'BULL_STRONG':      [0.60, 0.25, 0.10, 0.03, 0.01, 0.00, 0.01],
            'BULL_WEAK':        [0.15, 0.50, 0.25, 0.07, 0.02, 0.00, 0.01],
            'NEUTRAL_BULLISH':  [0.08, 0.20, 0.45, 0.20, 0.05, 0.01, 0.01],
            'NEUTRAL_BEARISH':  [0.02, 0.08, 0.20, 0.45, 0.20, 0.04, 0.01],
            'BEAR_WEAK':        [0.01, 0.02, 0.08, 0.20, 0.50, 0.15, 0.04],
            'BEAR_STRONG':      [0.00, 0.01, 0.03, 0.10, 0.25, 0.55, 0.06],
            'CRISIS':           [0.00, 0.01, 0.02, 0.05, 0.15, 0.35, 0.42]
        }
        
        matrix = pd.DataFrame(
            [matrix_data[regime] for regime in self.regime_order],
            index=self.regime_order,
            columns=self.regime_order
        )
        
        return matrix
    
    def _calculate_transition_matrix(self) -> pd.DataFrame:
        """
        Laske transitio-matriisi historiasta TAI käytä oletusmatriisia
        """
        # Jos ei tarpeeksi historiaa (< 30 päivää)
        if self.history_df.empty or len(self.history_df) < 30:
            print("[RegimePredictor] Using default transition matrix (insufficient history)")
            return self._get_default_transition_matrix()
        
        # Tarkista onko vaihtelua
        unique_regimes = self.history_df['regime'].nunique()
        if unique_regimes < 3:
            print(f"[RegimePredictor] Using default matrix (only {unique_regimes} unique regimes)")
            return self._get_default_transition_matrix()
        
        print(f"[RegimePredictor] Calculating transition matrix from {len(self.history_df)} days of history")
        
        # Laske transitiot historiasta
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.history_df) - 1):
            current = self.history_df.iloc[i]['regime']
            next_regime = self.history_df.iloc[i + 1]['regime']
            transitions[current][next_regime] += 1
        
        # Muunna todennäköisyyksiksi
        matrix_data = []
        for from_regime in self.regime_order:
            row = []
            total = sum(transitions[from_regime].values())
            
            for to_regime in self.regime_order:
                if total > 0:
                    prob = transitions[from_regime][to_regime] / total
                else:
                    # Fallback: käytä oletusmatriisia tälle regimelle
                    default_matrix = self._get_default_transition_matrix()
                    prob = default_matrix.loc[from_regime, to_regime]
                row.append(prob)
            
            matrix_data.append(row)
        
        matrix = pd.DataFrame(
            matrix_data,
            index=self.regime_order,
            columns=self.regime_order
        )
        
        return matrix
    
    def _calculate_regime_duration(self, current_date: str) -> int:
        """Laske kuinka monta päivää ollaan ollut nykyisessä regimessä"""
        if self.history_df.empty:
            return 0
        
        current_date_dt = pd.to_datetime(current_date)
        recent = self.history_df[self.history_df['date'] <= current_date_dt]
        
        if recent.empty:
            return 0
        
        current_regime = recent.iloc[-1]['regime']
        
        # Laske päivät taaksepäin samassa regimessä
        duration = 1
        for i in range(len(recent) - 2, -1, -1):
            if recent.iloc[i]['regime'] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_component_changes(self, lookback_days: int = 5) -> Dict[str, float]:
        """
        Laske komponenttien muutokset viimeisen N päivän ajalta
        
        Returns:
            Dict: {
                'equity_change_1d': +0.015,
                'equity_change_5d': +0.045,
                'volatility_change_1d': -0.02,
                ...
            }
        """
        if self.history_df.empty or len(self.history_df) < lookback_days:
            return {}
        
        changes = {}
        
        # Viimeisimmät rivit
        recent = self.history_df.tail(lookback_days + 1)
        
        # Laske muutokset (jos sarakkeet löytyvät)
        for component in ['equity_signal', 'volatility_signal', 'credit_signal', 
                         'safe_haven_signal', 'breadth_signal']:
            if component in recent.columns:
                # 1-päivän muutos
                if len(recent) >= 2:
                    change_1d = recent[component].iloc[-1] - recent[component].iloc[-2]
                    changes[f'{component}_change_1d'] = float(change_1d)
                
                # N-päivän muutos
                change_nd = recent[component].iloc[-1] - recent[component].iloc[0]
                changes[f'{component}_change_{lookback_days}d'] = float(change_nd)
        
        # Composite score muutos
        if 'composite_score' in recent.columns:
            changes['composite_score_change_1d'] = float(
                recent['composite_score'].iloc[-1] - recent['composite_score'].iloc[-2]
            ) if len(recent) >= 2 else 0.0
            
            changes['composite_score_change_5d'] = float(
                recent['composite_score'].iloc[-1] - recent['composite_score'].iloc[0]
            )
        
        return changes
    
    def predict_simple(self, current_regime: str, horizon_days: int = 5) -> Dict:
        """
        Yksinkertainen ennuste transitio-matriisin perusteella
        
        Args:
            current_regime: Nykyinen regime
            horizon_days: Montako päivää eteenpäin (1-5)
        
        Returns:
            {
                'current_regime': 'NEUTRAL_BULLISH',
                'horizon_days': 5,
                'predictions': {
                    'BULL_STRONG': 0.12,
                    'BULL_WEAK': 0.35,
                    'NEUTRAL_BULLISH': 0.40,
                    ...
                },
                'most_likely': 'NEUTRAL_BULLISH',
                'transition_probability': 0.60
            }
        """
        if current_regime not in self.regime_order:
            current_regime = 'NEUTRAL_BULLISH'
        
        # Aloitetaan nykyisestä
        probs = pd.Series(0.0, index=self.regime_order)
        probs[current_regime] = 1.0
        
        # Moninkertainen transitio (päivä kerrallaan)
        for _ in range(horizon_days):
            # Matrix multiplication: current_probs @ transition_matrix
            probs = probs @ self.transition_matrix
        
        # Muunna dict:ksi
        predictions = probs.to_dict()
        most_likely = probs.idxmax()
        transition_prob = 1.0 - probs[current_regime]  # Todennäköisyys että muuttuu
        
        return {
            'current_regime': current_regime,
            'horizon_days': horizon_days,
            'predictions': predictions,
            'most_likely': most_likely,
            'transition_probability': float(transition_prob),
            'method': 'transition_matrix'
        }
    
    def predict_with_ml(self, current_date: str, horizon_days: int = 5) -> Dict:
        """
        Edistynyt ennuste LightGBM:llä (jos malli koulutettu)
        
        Returns:
            Dict: Sama rakenne kuin predict_simple + feature importance
        """
        if not HAS_LIGHTGBM or self.model is None:
            # Fallback yksinkertaiseen
            current_regime = self.history_df.iloc[-1]['regime'] if not self.history_df.empty else 'NEUTRAL_BULLISH'
            result = self.predict_simple(current_regime, horizon_days)
            result['method'] = 'fallback_to_simple'
            return result
        
        # TODO: Implement ML prediction (Step 3)
        # Nyt placeholder
        current_regime = self.history_df.iloc[-1]['regime'] if not self.history_df.empty else 'NEUTRAL_BULLISH'
        return self.predict_simple(current_regime, horizon_days)
    
    def predict(self, current_date: Optional[str] = None, horizon_days: int = 5) -> Dict:
        """
        Pääfunktio: ennusta seuraava regime
        
        Käyttää ML:ää jos saatavilla, muuten transitio-matriisia
        """
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Hae nykyinen regime
        if not self.history_df.empty:
            recent = self.history_df[self.history_df['date'] <= current_date]
            if not recent.empty:
                current_regime = recent.iloc[-1]['regime']
            else:
                current_regime = 'NEUTRAL_BULLISH'
        else:
            current_regime = 'NEUTRAL_BULLISH'
        
        # Laske lisätietoja
        duration = self._calculate_regime_duration(current_date)
        component_changes = self._calculate_component_changes()
        
        # Tee ennuste
        if HAS_LIGHTGBM and self.model is not None:
            prediction = self.predict_with_ml(current_date, horizon_days)
        else:
            prediction = self.predict_simple(current_regime, horizon_days)
        
        # Lisää metadata
        prediction['regime_duration_days'] = duration
        prediction['component_changes'] = component_changes
        
        # Transitio-signaalit
        prediction['transition_signals'] = self._detect_transition_signals(component_changes)
        
        return prediction
    
    def _detect_transition_signals(self, component_changes: Dict) -> Dict[str, bool]:
        """
        Tunnista signaaleja regimen muutoksesta
        
        Returns:
            {
                'equity_accelerating': True/False,
                'volatility_declining': True/False,
                'credit_improving': True/False,
                'safe_haven_declining': True/False
            }
        """
        signals = {}
        
        # Equity kiihtyvä?
        if 'equity_signal_change_5d' in component_changes:
            signals['equity_accelerating'] = component_changes['equity_signal_change_5d'] > 0.1
        
        # Volatiliteetti laskeva?
        if 'volatility_signal_change_5d' in component_changes:
            signals['volatility_declining'] = component_changes['volatility_signal_change_5d'] > 0.1
        
        # Credit paraneva?
        if 'credit_signal_change_5d' in component_changes:
            signals['credit_improving'] = component_changes['credit_signal_change_5d'] > 0.1
        
        # Safe haven vähenevä?
        if 'safe_haven_signal_change_5d' in component_changes:
            signals['safe_haven_declining'] = component_changes['safe_haven_signal_change_5d'] > 0.1
        
        return signals
    
    def print_prediction(self, prediction: Dict):
        """Tulosta ennuste kauniisti"""
        print("\n" + "="*80)
        print("🔮 REGIME PREDICTION")
        print("="*80)
        
        print(f"\nCurrent Regime: {prediction['current_regime']}")
        print(f"Duration:       {prediction.get('regime_duration_days', 0)} days")
        print(f"Horizon:        {prediction['horizon_days']} days")
        print(f"Method:         {prediction.get('method', 'unknown')}")
        
        print(f"\n📊 PREDICTIONS (in {prediction['horizon_days']} days):")
        print("-" * 80)
        
        # Järjestä todennäköisyyden mukaan
        preds = sorted(
            prediction['predictions'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for regime, prob in preds:
            bar = "█" * int(prob * 50)
            marker = " ← MOST LIKELY" if regime == prediction['most_likely'] else ""
            print(f"{regime:<18} {prob:>6.1%} {bar}{marker}")
        
        print(f"\nTransition Probability: {prediction['transition_probability']:.1%}")
        
        # Transitio-signaalit
        if 'transition_signals' in prediction and prediction['transition_signals']:
            print(f"\n⚠️  TRANSITION SIGNALS:")
            for signal, active in prediction['transition_signals'].items():
                status = "✅ YES" if active else "❌ NO"
                print(f"  {signal}: {status}")
        
        print("="*80 + "\n")
    
    def print_transition_matrix(self):
        """Tulosta transitio-matriisi"""
        print("\n" + "="*80)
        print("📈 HISTORICAL TRANSITION MATRIX")
        print("="*80)
        print("\nProbability of transitioning FROM (rows) TO (columns):\n")
        
        # Format taulukko
        print(self.transition_matrix.to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n" + "="*80 + "\n")


# ==================== TESTAUSFUNKTIO ====================

def test_predictor():
    """Testaa RegimePredictor"""
    print("\n" + "="*80)
    print("🧪 TESTING RegimePredictor")
    print("="*80 + "\n")
    
    # Luo predictor
    predictor = RegimePredictor()
    
    # Testaa transitio-matriisi
    print("1️⃣ Testing transition matrix...")
    predictor.print_transition_matrix()
    
    # Testaa ennuste (1 päivä)
    print("\n2️⃣ Testing 1-day prediction...")
    pred_1d = predictor.predict(horizon_days=1)
    predictor.print_prediction(pred_1d)
    
    # Testaa ennuste (5 päivää)
    print("\n3️⃣ Testing 5-day prediction...")
    pred_5d = predictor.predict(horizon_days=5)
    predictor.print_prediction(pred_5d)
    
    print("\n✅ PREDICTOR TESTS COMPLETE!")


if __name__ == "__main__":
    test_predictor()