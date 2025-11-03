#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_portfolio.py
===================
Helper script: Päivitä portfolio_state.json manuaalisesti

Features:
- Lisää/päivitä positioita
- Poista positioita (DELETE-komento)
- Listaa nykyiset positiot
- Päivitä cash
- UTF-8 tallennos (ei BOM)

Usage:
    python update_portfolio.py
"""

import json
import sys
import os
from datetime import date

def load_portfolio(path="seasonality_reports/portfolio_state.json"):
    """Lataa portfolio state"""
    if not os.path.exists(path):
        print(f"[WARN] Portfolio state ei löydy, luodaan uusi")
        return {
            "positions": {},
            "cash": 100000.0,
            "counters": {
                "day_entries": 0,
                "week_entries": 0,
                "week_start": str(date.today()),
                "last_day": str(date.today())
            },
            "settings": {
                "max_positions": 5,
                "max_entries_day": 1,
                "max_entries_week": 3,
                "max_weight_pct": 20.0
            },
            "last_updated": str(date.today())
        }
    
    try:
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Virhe luettaessa portfolio_state.json: {e}")
        sys.exit(1)

def save_portfolio(state, path="seasonality_reports/portfolio_state.json"):
    """Tallenna portfolio state (UTF-8 ilman BOM)"""
    try:
        # Varmista että kansio on olemassa
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Tallenna UTF-8 ilman BOM
        with open(path, "w", encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"[ERROR] Virhe tallennettaessa: {e}")
        return False

def display_positions(positions):
    """Näytä nykyiset positiot taulukkona"""
    if not positions:
        print("Ei positioita.\n")
        return
    
    print("\n" + "="*70)
    print(f"{'Ticker':<8} | {'Qty':>5} | {'Entry Price':>12} | {'Entry Date':<12} | {'Regime':<16}")
    print("="*70)
    
    for ticker, pos in sorted(positions.items()):
        qty = pos.get('quantity', 0)
        price = pos.get('entry_price', 0.0)
        date_str = pos.get('entry_date', 'Unknown')
        regime = pos.get('regime_at_entry', 'Unknown')
        
        print(f"{ticker:<8} | {qty:>5} | ${price:>11.2f} | {date_str:<12} | {regime:<16}")
    
    print("="*70 + "\n")

def add_or_update_position(positions):
    """Lisää tai päivitä positio"""
    ticker = input("Ticker (esim. AAPL): ").strip().upper()
    
    if not ticker:
        return False
    
    print(f"\nLisätään/päivitetään: {ticker}")
    
    # Entry price
    while True:
        try:
            entry_price = input(f"  Entry price (esim. 345.75): $").strip()
            entry_price = float(entry_price)
            break
        except ValueError:
            print("  [ERROR] Anna numero (esim. 345.75)")
    
    # Quantity
    while True:
        try:
            quantity = input(f"  Quantity (esim. 10): ").strip()
            quantity = int(quantity)
            break
        except ValueError:
            print("  [ERROR] Anna kokonaisluku (esim. 10)")
    
    # Entry date
    entry_date = input(f"  Entry date (YYYY-MM-DD) [{date.today()}]: ").strip()
    if not entry_date:
        entry_date = str(date.today())
    
    # Regime
    regime = input(f"  Regime [NEUTRAL_BULLISH]: ").strip().upper()
    if not regime:
        regime = "NEUTRAL_BULLISH"
    
    # Tallenna
    positions[ticker] = {
        "entry_price": entry_price,
        "quantity": quantity,
        "entry_date": entry_date,
        "regime_at_entry": regime
    }
    
    print(f"\n✅ {ticker} lisätty/päivitetty!\n")
    return True

def delete_position(positions):
    """Poista positio"""
    if not positions:
        print("\n[WARN] Ei positioita poistettavaksi.\n")
        return False
    
    print("\nNykyiset positiot:")
    for ticker in sorted(positions.keys()):
        print(f"  - {ticker}")
    
    ticker = input("\nPoista ticker (esim. AAPL): ").strip().upper()
    
    if not ticker:
        return False
    
    if ticker not in positions:
        print(f"\n[WARN] {ticker} ei löydy portfoliosta.\n")
        return False
    
    # Vahvista poisto
    confirm = input(f"\n⚠️  Poistetaanko {ticker}? (y/n): ").strip().lower()
    
    if confirm == 'y' or confirm == 'yes':
        del positions[ticker]
        print(f"\n✅ {ticker} poistettu!\n")
        return True
    else:
        print("\n❌ Poisto peruttu.\n")
        return False

def update_cash(state):
    """Päivitä cash"""
    current_cash = state.get('cash', 0.0)
    print(f"\nNykyinen cash: ${current_cash:,.2f}")
    
    new_cash = input(f"Uusi cash (tyhjä = ei muutosta): $").strip()
    
    if new_cash:
        try:
            state['cash'] = float(new_cash.replace(',', ''))
            print(f"✅ Cash päivitetty: ${state['cash']:,.2f}\n")
            return True
        except ValueError:
            print("[ERROR] Virheellinen määrä\n")
            return False
    
    return False

def main():
    """Pääohjelma"""
    
    print("\n" + "="*70)
    print("📊 PORTFOLIO UPDATE HELPER")
    print("="*70)
    
    # Lataa portfolio
    state = load_portfolio()
    positions = state.get("positions", {})
    
    while True:
        # Näytä menu
        print("\n" + "-"*70)
        print(f"Positioita: {len(positions)}/5")
        print(f"Cash: ${state.get('cash', 0.0):,.2f}")
        print("-"*70)
        
        display_positions(positions)
        
        print("TOIMINNOT:")
        print("  [A] Lisää/päivitä positio")
        print("  [D] Poista positio")
        print("  [C] Päivitä cash")
        print("  [L] Listaa positiot")
        print("  [S] Tallenna ja lopeta")
        print("  [Q] Lopeta tallentamatta")
        print()
        
        choice = input("Valinta (A/D/C/L/S/Q): ").strip().upper()
        
        if choice == 'A':
            add_or_update_position(positions)
        
        elif choice == 'D':
            delete_position(positions)
        
        elif choice == 'C':
            update_cash(state)
        
        elif choice == 'L':
            display_positions(positions)
        
        elif choice == 'S':
            # Tallenna
            state["positions"] = positions
            state["last_updated"] = str(date.today())
            
            if save_portfolio(state):
                print("\n" + "="*70)
                print("✅ PORTFOLIO TALLENNETTU!")
                print("="*70)
                print(f"\nPositioita: {len(positions)}/5")
                print(f"Cash: ${state['cash']:,.2f}")
                print(f"Päivitetty: {state['last_updated']}")
                print(f"\nTiedosto: seasonality_reports/portfolio_state.json\n")
                break
            else:
                print("\n[ERROR] Tallennus epäonnistui!")
        
        elif choice == 'Q':
            confirm = input("\n⚠️  Lopetetaanko tallentamatta? (y/n): ").strip().lower()
            if confirm == 'y' or confirm == 'yes':
                print("\n❌ Muutoksia ei tallennettu.\n")
                break
        
        else:
            print("\n[ERROR] Tuntematon valinta\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Keskeytetty. Muutoksia ei tallennettu.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Odottamaton virhe: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)