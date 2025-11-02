#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_regime_history.py
Luo historiallinen regime-data ajamalla RegimeDetector 90 päivää taaksepäin
"""

from datetime import datetime, timedelta
from regime_detector import RegimeDetector
import pandas as pd

def simulate_history(days_back: int = 90):
    """Simuloi regime-historia"""
    print(f"\n[SimulateHistory] Simulating {days_back} days of regime history...")
    
    # Luo detector
    detector = RegimeDetector(
        macro_price_cache_dir="seasonality_reports/price_cache",
        equity_price_cache_dir="seasonality_reports/price_cache"
    )
    
    # Kerää historia
    history = []
    end_date = datetime.now()
    
    for i in range(days_back, -1, -1):
        date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
        
        print(f"[{days_back - i + 1}/{days_back + 1}] Detecting regime for {date}...", end='\r')
        
        try:
            regime_data = detector.detect_regime(date=date)
            history.append(regime_data)
        except Exception as e:
            print(f"\n[WARN] Failed for {date}: {e}")
            continue
    
    print(f"\n[SimulateHistory] Collected {len(history)} regime observations")
    
    # Tallenna
    if history:
        # Lataa olemassa oleva (jos on)
        history_path = "seasonality_reports/regime_history.csv"
        
        # Yhdistä uusi ja vanha
        new_df = pd.DataFrame([
            {
                'date': h['date'],
                'regime': h['regime'],
                'composite_score': h['composite_score'],
                'confidence': h['confidence'],
                'equity_signal': h['components']['equity']['signal'],
                'volatility_signal': h['components']['volatility']['signal'],
                'credit_signal': h['components']['credit']['signal'],
                'safe_haven_signal': h['components']['safe_haven']['signal'],
                'breadth_signal': h['components']['breadth']['signal']
            }
            for h in history
        ])
        
        # Tallenna (korvaa vanha)
        new_df.to_csv(history_path, index=False)
        print(f"[SimulateHistory] Saved to {history_path}")
        
        # Näytä tilastot
        print("\n" + "="*80)
        print("REGIME DISTRIBUTION:")
        print("="*80)
        print(new_df['regime'].value_counts().to_string())
        print("="*80)
    
    return history

if __name__ == "__main__":
    simulate_history(days_back=90)