#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_detector.py - Adaptive Multi-Factor Market Regime Detection

Tunnistaa markkinatilanteen käyttäen:
1. Equity momentum (SPY, QQQ, IWM)
2. Volatility (realized vol)
3. Credit spreads (HYG vs LQD)
4. Safe haven flows (GLD, TLT)
5. Breadth (osakkeiden korrelaatiot)

Regimes:
- BULL_STRONG: Vahva nousu, matala volatiliteetti
- BULL_WEAK: Nousu, korkea volatiliteetti
- NEUTRAL_BULLISH: Sivuttain, positiivinen bias
- NEUTRAL_BEARISH: Sivuttain, negatiivinen bias
- BEAR_WEAK: Lasku, kohtalainen volatiliteetti
- BEAR_STRONG: Vahva lasku, korkea volatiliteetti
- CRISIS: Paniikki, äärimmäinen volatiliteetti
"""

import os
import sys

# UTF-8 encoding fix for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import json

class RegimeDetector:
    """
    Adaptiivinen regime-tunnistin
    
    Features:
    - Multi-factor analyysi (5 komponenttia)
    - Dynaaminen kalibrointi (oppii historiasta)
    - Confidence score (luottamus tunnistukseen)
    - Transition detection (regime-muutosten tunnistus)
    """
    
    def __init__(self, 
                 macro_price_cache_dir: str,
                 equity_price_cache_dir: Optional[str] = None,
                 config_path: str = "config/regime_config.json"):
        """
        Args:
            macro_price_cache_dir: Makro-indikaattorit (GLD, TLT, HYG, LQD)
            equity_price_cache_dir: Osakeindeksit (SPY, QQQ, IWM) - jos eri kuin macro
            config_path: Konfiguraatiotiedoston polku
        """
        self.macro_cache_dir = Path(macro_price_cache_dir)
        
        # Jos equity_cache ei määritelty, käytä samaa kuin macro
        if equity_price_cache_dir:
            self.equity_cache_dir = Path(equity_price_cache_dir)
        else:
            self.equity_cache_dir = self.macro_cache_dir
        
        self.config_path = Path(config_path)
        
        # Lataa tai luo konfiguraatio
        self.config = self._load_config()
        
        # Vaaditut ticker-indikaattorit
        self.required_tickers = {
            'equity': ['SPY', 'QQQ', 'IWM'],      # Osakeindeksit
            'credit': ['HYG', 'LQD'],              # Luotto (high yield vs investment grade)
            'safe_haven': ['GLD', 'TLT'],          # Turvasatamat
            'commodity': ['USO'],                   # Öljy (valinnainen)
            'fx': ['UUP']                          # Dollari (valinnainen)
        }
        
        # Määritä mistä cachesta mikäkin ticker ladataan
        self.ticker_cache_map = {
            'SPY': self.equity_cache_dir,
            'QQQ': self.equity_cache_dir,
            'IWM': self.equity_cache_dir,
            'HYG': self.macro_cache_dir,
            'LQD': self.macro_cache_dir,
            'GLD': self.macro_cache_dir,
            'TLT': self.macro_cache_dir,
            'USO': self.macro_cache_dir,
            'UUP': self.macro_cache_dir
        }
    
    def _load_config(self) -> Dict:
        """Lataa tai luo regime detection konfiguraatio"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        else:
            # Default konfiguraatio
            return {
                'weights': {
                    'equity_momentum': 0.35,
                    'volatility': 0.25,
                    'credit_spread': 0.20,
                    'safe_haven': 0.10,
                    'breadth': 0.10
                },
                'thresholds': {
                    'bull_strong': 0.50,
                    'bull_weak': 0.10,
                    'neutral_bearish': -0.10,
                    'bear_weak': -0.50,
                    'crisis_vol': 0.40,      # Realized vol > 40% = crisis
                    'high_vol': 0.25,        # Vol > 25% = high
                    'low_vol': 0.15          # Vol < 15% = low
                },
                'lookback_periods': {
                    'momentum': 20,           # 20 päivää momentum
                    'volatility': 20,         # 20 päivää realized vol
                    'correlation': 60         # 60 päivää korrelaatiot
                },
                'last_calibrated': None,
                'performance_metrics': {}
            }
    
    def _save_config(self):
        """Tallenna konfiguraatio"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _load_ticker(self, ticker: str, days: int = 120) -> Optional[pd.DataFrame]:
        """Lataa ticker-data oikeasta price_cachesta"""
        # Määritä oikea cache tälle tickerille
        cache_dir = self.ticker_cache_map.get(ticker, self.macro_cache_dir)
        csv_path = cache_dir / f"{ticker}.csv"
        
        if not csv_path.exists():
            print(f"[RegimeDetector] Warning: {ticker}.csv not found in {cache_dir}")
            return None
        
        try:
            # ============= KORJAUS: Skipaa ticker-rivi (rivi 2) =============
            df = pd.read_csv(csv_path, parse_dates=['Date'], skiprows=[1])
            # ================================================================
            df = df.sort_values('Date')
            
            # Ota vain viimeisimmät N päivää
            if len(df) > days:
                df = df.tail(days)
            
            return df
        except Exception as e:
            print(f"[RegimeDetector] Error loading {ticker}: {e}")
            return None
    
    def _calculate_momentum(self, df: pd.DataFrame, period: int = 20) -> float:
        """Laske momentum (prosentuaalinen muutos)"""
        if df is None or len(df) < period:
            return 0.0
        
        try:
            close = df['Close'].astype(float)
            momentum = (close.iloc[-1] / close.iloc[-period] - 1.0)
            return float(momentum) if np.isfinite(momentum) else 0.0
        except:
            return 0.0
    
    def _calculate_realized_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """Laske realized volatility (annualisoitu)"""
        if df is None or len(df) < period + 1:
            return 0.0
        
        try:
            close = df['Close'].astype(float)
            returns = close.pct_change().dropna()
            vol = returns.tail(period).std() * np.sqrt(252)  # Annualisoitu
            return float(vol) if np.isfinite(vol) else 0.0
        except:
            return 0.0
    
    def _equity_signal(self) -> Tuple[float, Dict]:
        """
        KOMPONENTTI 1: Equity Momentum
        
        Käyttää: SPY, QQQ, IWM
        Output: -1 to +1 (bear to bull)
        """
        lookback = self.config['lookback_periods']['momentum']
        momentums = []
        details = {}
        
        for ticker in self.required_tickers['equity']:
            df = self._load_ticker(ticker)
            if df is not None:
                mom = self._calculate_momentum(df, lookback)
                momentums.append(mom)
                details[ticker] = mom
        
        if not momentums:
            return 0.0, details
        
        # Mediaani momentum
        median_mom = np.median(momentums)
        
        # Normalisoi -1 to +1
        # -10% = -1, +10% = +1
        signal = np.clip(median_mom / 0.10, -1.0, 1.0)
        
        details['median'] = median_mom
        details['signal'] = signal
        
        return signal, details
    
    def _volatility_signal(self) -> Tuple[float, Dict]:
        """
        KOMPONENTTI 2: Volatility
        
        Käyttää: SPY realized vol
        Output: -1 (high vol/bad) to +1 (low vol/good)
        """
        lookback = self.config['lookback_periods']['volatility']
        thresholds = self.config['thresholds']
        
        df = self._load_ticker('SPY')
        if df is None:
            return 0.0, {'error': 'SPY not found'}
        
        vol = self._calculate_realized_volatility(df, lookback)
        
        # Normalisoi: low vol = +1, high vol = -1
        if vol > thresholds['crisis_vol']:
            signal = -1.0  # Paniikki
        elif vol > thresholds['high_vol']:
            signal = -0.5  # Korkea vol
        elif vol < thresholds['low_vol']:
            signal = 1.0   # Matala vol
        else:
            signal = 0.0   # Normaali
        
        details = {
            'realized_vol': vol,
            'signal': signal,
            'classification': 'crisis' if vol > thresholds['crisis_vol'] else 
                            ('high' if vol > thresholds['high_vol'] else 
                            ('low' if vol < thresholds['low_vol'] else 'normal'))
        }
        
        return signal, details
    
    def _credit_signal(self) -> Tuple[float, Dict]:
        """
        KOMPONENTTI 3: Credit Spread
        
        Käyttää: HYG (high yield) vs LQD (investment grade)
        Output: -1 (widening spreads) to +1 (narrowing spreads)
        """
        lookback = self.config['lookback_periods']['momentum']
        
        hyg_df = self._load_ticker('HYG')
        lqd_df = self._load_ticker('LQD')
        
        if hyg_df is None or lqd_df is None:
            return 0.0, {'error': 'HYG or LQD not found'}
        
        hyg_mom = self._calculate_momentum(hyg_df, lookback)
        lqd_mom = self._calculate_momentum(lqd_df, lookback)
        
        # HYG > LQD = riskinotto = hyvä
        # LQD > HYG = turvasatama = huono
        spread_signal = hyg_mom - lqd_mom
        
        # Normalisoi
        signal = np.clip(spread_signal / 0.05, -1.0, 1.0)
        
        details = {
            'HYG_momentum': hyg_mom,
            'LQD_momentum': lqd_mom,
            'spread_difference': spread_signal,
            'signal': signal
        }
        
        return signal, details
    
    def _safe_haven_signal(self) -> Tuple[float, Dict]:
        """
        KOMPONENTTI 4: Safe Haven Flow
        
        Käyttää: GLD (kulta), TLT (treasuries)
        Output: -1 (flight to safety) to +1 (risk on)
        """
        lookback = self.config['lookback_periods']['momentum']
        
        gld_df = self._load_ticker('GLD')
        tlt_df = self._load_ticker('TLT')
        
        if gld_df is None or tlt_df is None:
            return 0.0, {'error': 'GLD or TLT not found'}
        
        gld_mom = self._calculate_momentum(gld_df, lookback)
        tlt_mom = self._calculate_momentum(tlt_df, lookback)
        
        # Kulta/obligaatiot nousee = turvasatamahaku = huono
        safe_haven_flow = (gld_mom + tlt_mom) / 2
        
        # Käänteinen: safe haven nousee = negatiivinen signaali
        signal = -np.clip(safe_haven_flow / 0.05, -1.0, 1.0)
        
        details = {
            'GLD_momentum': gld_mom,
            'TLT_momentum': tlt_mom,
            'safe_haven_flow': safe_haven_flow,
            'signal': signal
        }
        
        return signal, details
    
    def _breadth_signal(self, equity_details: Dict) -> Tuple[float, Dict]:
        """
        KOMPONENTTI 5: Market Breadth
        
        Käyttää: SPY, QQQ, IWM korrelaatio
        Output: -1 (divergence) to +1 (convergence)
        """
        # Tarkista että kaikkien momentumit samaan suuntaan
        momentums = [
            equity_details.get('SPY', 0),
            equity_details.get('QQQ', 0),
            equity_details.get('IWM', 0)
        ]
        
        if not momentums:
            return 0.0, {'error': 'No equity data'}
        
        # Laske hajonta (pieni = hyvä breadth)
        std = np.std(momentums)
        mean = np.mean(momentums)
        
        # Jos kaikki samaan suuntaan = hyvä breadth
        if mean > 0 and std < 0.02:  # Kaikki nousee
            signal = 1.0
        elif mean < 0 and std < 0.02:  # Kaikki laskee
            signal = -1.0
        elif std > 0.05:  # Suuri hajonta = huono breadth
            signal = -0.5
        else:
            signal = 0.0
        
        details = {
            'momentums': momentums,
            'mean': mean,
            'std': std,
            'signal': signal
        }
        
        return signal, details
    
    def detect_regime(self, date: Optional[str] = None) -> Dict:
        """
        PÄÄFUNKTIO: Tunnista regime
        
        Returns:
            {
                'regime': 'BULL_STRONG' | 'BULL_WEAK' | 'NEUTRAL_BULLISH' | ...
                'composite_score': -1.0 to 1.0
                'confidence': 0.0 to 1.0
                'components': {...}
                'date': '2025-11-02'
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\n[RegimeDetector] Detecting regime for {date}...")
        
        # Laske komponentit
        weights = self.config['weights']
        
        equity_sig, equity_det = self._equity_signal()
        vol_sig, vol_det = self._volatility_signal()
        credit_sig, credit_det = self._credit_signal()
        safe_sig, safe_det = self._safe_haven_signal()
        breadth_sig, breadth_det = self._breadth_signal(equity_det)
        
        # Yhdistä painotetusti
        composite_score = (
            weights['equity_momentum'] * equity_sig +
            weights['volatility'] * vol_sig +
            weights['credit_spread'] * credit_sig +
            weights['safe_haven'] * safe_sig +
            weights['breadth'] * breadth_sig
        )
        
        # Määritä regime thresholdien perusteella
        thresholds = self.config['thresholds']
        realized_vol = vol_det.get('realized_vol', 0.20)
        
        if realized_vol > thresholds['crisis_vol']:
            regime = 'CRISIS'
        elif composite_score > thresholds['bull_strong']:
            regime = 'BULL_STRONG' if realized_vol < thresholds['low_vol'] else 'BULL_WEAK'
        elif composite_score > thresholds['bull_weak']:
            regime = 'NEUTRAL_BULLISH'
        elif composite_score > thresholds['neutral_bearish']:
            regime = 'NEUTRAL_BEARISH'
        elif composite_score > thresholds['bear_weak']:
            regime = 'BEAR_WEAK'
        else:
            regime = 'BEAR_STRONG'
        
        # Laske confidence (kuinka selkeä regime on)
        confidence = min(1.0, abs(composite_score) * 2)
        
        result = {
            'date': date,
            'regime': regime,
            'composite_score': round(composite_score, 4),
            'confidence': round(confidence, 4),
            'components': {
                'equity': {'signal': equity_sig, 'details': equity_det},
                'volatility': {'signal': vol_sig, 'details': vol_det},
                'credit': {'signal': credit_sig, 'details': credit_det},
                'safe_haven': {'signal': safe_sig, 'details': safe_det},
                'breadth': {'signal': breadth_sig, 'details': breadth_det}
            },
            'weights': weights
        }
        
        print(f"[RegimeDetector] Regime: {regime} (score: {composite_score:.3f}, confidence: {confidence:.2%})")
        
        return result
    
    def save_regime_history(self, regime_data: Dict, output_dir: str):
        """Tallenna regime-historia analysointia varten"""
        output_path = Path(output_dir) / "regime_history.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Yksinkertaistettu rivi CSV:hen
        row = {
            'date': regime_data['date'],
            'regime': regime_data['regime'],
            'composite_score': regime_data['composite_score'],
            'confidence': regime_data['confidence'],
            'equity_signal': regime_data['components']['equity']['signal'],
            'volatility_signal': regime_data['components']['volatility']['signal'],
            'credit_signal': regime_data['components']['credit']['signal'],
            'safe_haven_signal': regime_data['components']['safe_haven']['signal'],
            'breadth_signal': regime_data['components']['breadth']['signal']
        }
        
        df = pd.DataFrame([row])
        
        # Append jos tiedosto on olemassa
        if output_path.exists():
            existing = pd.read_csv(output_path)
            # Poista duplikaatit (jos sama päivä)
            existing = existing[existing['date'] != regime_data['date']]
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(output_path, index=False)
        print(f"[RegimeDetector] Saved to {output_path}")


# ==================== TESTAUSFUNKTIO ====================

def test_regime_detector():
    """Testaa regime detectoria"""
    print("\n" + "="*80)
    print("🧪 TESTING REGIME DETECTOR")
    print("="*80 + "\n")
    
    # ============= SAMA CACHE MOLEMMILLE =============
    price_cache = Path("seasonality_reports/price_cache")
    # =================================================
    
    if not price_cache.exists():
        print(f"❌ Price cache not found: {price_cache}")
        return
    
    # Tarkista että SPY löytyy
    if not (price_cache / "SPY.csv").exists():
        print(f"❌ SPY.csv not found in {price_cache}")
        print(f"💡 Run 'python update_regime_prices.py' first!")
        return
    
    print(f"✅ Using price cache: {price_cache}")
    
    # Luo detector (sama cache molemmille)
    detector = RegimeDetector(
        macro_price_cache_dir=str(price_cache),
        equity_price_cache_dir=str(price_cache)
    )
    
    # Tunnista regime
    regime_data = detector.detect_regime()
    
    # Tulosta tulokset
    print("\n" + "="*80)
    print("📊 REGIME DETECTION RESULTS:")
    print("="*80)
    print(f"Date:            {regime_data['date']}")
    print(f"Regime:          {regime_data['regime']}")
    print(f"Composite Score: {regime_data['composite_score']:.4f}")
    print(f"Confidence:      {regime_data['confidence']:.2%}")
    print("\n" + "-"*80)
    print("COMPONENT SIGNALS:")
    print("-"*80)
    for component, data in regime_data['components'].items():
        print(f"{component:15s}: {data['signal']:+.3f}")
        
    # Tulosta yksityiskohtia
    print("\n" + "-"*80)
    print("COMPONENT DETAILS:")
    print("-"*80)
    
    # Equity details
    eq_det = regime_data['components']['equity']['details']
    print(f"\nEquity Momentum:")
    if 'SPY' in eq_det:
        print(f"  SPY:    {eq_det['SPY']:+.2%}")
    if 'QQQ' in eq_det:
        print(f"  QQQ:    {eq_det['QQQ']:+.2%}")
    if 'IWM' in eq_det:
        print(f"  IWM:    {eq_det['IWM']:+.2%}")
    if 'median' in eq_det:
        print(f"  Median: {eq_det['median']:+.2%}")
    
    # Volatility details
    vol_det = regime_data['components']['volatility']['details']
    if 'realized_vol' in vol_det:
        print(f"\nVolatility:")
        print(f"  Realized Vol: {vol_det['realized_vol']:.2%}")
        print(f"  Class:        {vol_det.get('classification', 'N/A')}")
    
    # Credit details
    cred_det = regime_data['components']['credit']['details']
    if 'HYG_momentum' in cred_det:
        print(f"\nCredit Spread:")
        print(f"  HYG: {cred_det['HYG_momentum']:+.2%}")
        print(f"  LQD: {cred_det['LQD_momentum']:+.2%}")
        print(f"  Diff: {cred_det['spread_difference']:+.2%}")
    
    # Safe haven details
    safe_det = regime_data['components']['safe_haven']['details']
    if 'GLD_momentum' in safe_det:
        print(f"\nSafe Haven:")
        print(f"  GLD: {safe_det['GLD_momentum']:+.2%}")
        print(f"  TLT: {safe_det['TLT_momentum']:+.2%}")
    
    print("="*80)
    
    # Tallenna historia
    detector.save_regime_history(regime_data, "seasonality_reports")
    
    print("\n✅ TEST COMPLETE!")


if __name__ == "__main__":
    test_regime_detector()