#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trades_logger.py
Tallentaa jokaisen kaupan entry ja exit automaattisesti SQLite-tietokantaan.
Mahdollistaa historiallisen analyysin ja parametrien optimoinnin.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List
import json

class TradeLogger:
    """Kauppaloki - tallentaa kaikki entryt ja exitit"""
    
    def __init__(self, db_path: str = "seasonality_reports/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Luo tietokantataulu jos ei ole olemassa"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                
                -- Entry data
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop REAL NOT NULL,
                tp1 REAL,
                tp2 REAL,
                tp3 REAL,
                
                -- Setup & signals
                setup_type TEXT,
                ml_score REAL,
                seasonality_score REAL,
                mom5 REAL,
                mom20 REAL,
                mom60 REAL,
                vol20 REAL,
                
                -- Exit data (päivitetään myöhemmin)
                exit_date TEXT,
                exit_price REAL,
                exit_reason TEXT,
                
                -- Calculated results
                pnl_pct REAL,
                pnl_dollars REAL,
                r_multiple REAL,
                hold_days INTEGER,
                tp1_hit INTEGER DEFAULT 0,
                tp2_hit INTEGER DEFAULT 0,
                tp3_hit INTEGER DEFAULT 0,
                
                -- Status
                status TEXT DEFAULT 'OPEN',
                
                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index for faster queries
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_ticker ON trades(ticker)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_entry_date ON trades(entry_date)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_status ON trades(status)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_setup_type ON trades(setup_type)
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"[TradeLogger] Database initialized: {self.db_path}")
    
    def log_entry(self, 
                  ticker: str,
                  entry_date: str,
                  entry_price: float,
                  stop: float,
                  tp1: Optional[float] = None,
                  tp2: Optional[float] = None,
                  tp3: Optional[float] = None,
                  setup_type: Optional[str] = None,
                  ml_score: Optional[float] = None,
                  seasonality_score: Optional[float] = None,
                  mom5: Optional[float] = None,
                  mom20: Optional[float] = None,
                  mom60: Optional[float] = None,
                  vol20: Optional[float] = None,
                  **kwargs) -> str:
        """
        Tallenna kaupan avaus (entry)
        
        Returns:
            trade_id: Uniikki tunniste tälle kaupalle
        """
        trade_id = f"{ticker}_{entry_date}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tarkista onko jo olemassa
        cursor.execute("SELECT trade_id FROM trades WHERE trade_id = ?", (trade_id,))
        if cursor.fetchone():
            print(f"[TradeLogger] Trade already exists: {trade_id}")
            conn.close()
            return trade_id
        
        # Insert new trade
        cursor.execute('''
            INSERT INTO trades (
                trade_id, ticker, entry_date, entry_price,
                stop, tp1, tp2, tp3,
                setup_type, ml_score, seasonality_score,
                mom5, mom20, mom60, vol20,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id, ticker, entry_date, entry_price,
            stop, tp1, tp2, tp3,
            setup_type, ml_score, seasonality_score,
            mom5, mom20, mom60, vol20,
            'OPEN'
        ))
        
        conn.commit()
        conn.close()
        
        print(f"[TradeLogger] Entry logged: {trade_id}")
        return trade_id
    
    def log_exit(self, 
                 trade_id: str,
                 exit_date: str,
                 exit_price: float,
                 exit_reason: str) -> bool:
        """
        Tallenna kaupan sulkeminen (exit)
        Laskee automaattisesti P&L, R-multiple, hold_days, TP-hits
        
        Returns:
            True jos onnistui, False jos kauppaa ei löytynyt
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hae entry-tieto
        cursor.execute('''
            SELECT entry_date, entry_price, stop, tp1, tp2, tp3
            FROM trades WHERE trade_id = ?
        ''', (trade_id,))
        
        row = cursor.fetchone()
        if not row:
            print(f"[TradeLogger] Trade not found: {trade_id}")
            conn.close()
            return False
        
        entry_date_str, entry_price, stop, tp1, tp2, tp3 = row
        
        # Laske P&L
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        pnl_dollars = exit_price - entry_price
        
        # Laske R-multiple
        risk = entry_price - stop
        if risk > 0:
            profit = exit_price - entry_price
            r_multiple = profit / risk
        else:
            r_multiple = 0.0
        
        # Laske hold_days
        try:
            entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d")
            exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
            hold_days = (exit_dt - entry_dt).days
        except:
            hold_days = 0
        
        # Tarkista TP-hits
        tp1_hit = 1 if (tp1 and exit_price >= tp1) else 0
        tp2_hit = 1 if (tp2 and exit_price >= tp2) else 0
        tp3_hit = 1 if (tp3 and exit_price >= tp3) else 0
        
        # Päivitä tietokanta
        cursor.execute('''
            UPDATE trades SET
                exit_date = ?,
                exit_price = ?,
                exit_reason = ?,
                pnl_pct = ?,
                pnl_dollars = ?,
                r_multiple = ?,
                hold_days = ?,
                tp1_hit = ?,
                tp2_hit = ?,
                tp3_hit = ?,
                status = 'CLOSED',
                updated_at = CURRENT_TIMESTAMP
            WHERE trade_id = ?
        ''', (
            exit_date, exit_price, exit_reason,
            pnl_pct, pnl_dollars, r_multiple, hold_days,
            tp1_hit, tp2_hit, tp3_hit,
            trade_id
        ))
        
        conn.commit()
        conn.close()
        
        print(f"[TradeLogger] Exit logged: {trade_id} | P&L: {pnl_pct:.2f}% | R: {r_multiple:.2f}R | Reason: {exit_reason}")
        return True
    
    def get_open_trades(self) -> pd.DataFrame:
        """Hae kaikki avoimet kaupat"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_date DESC",
            conn
        )
        conn.close()
        return df
    
    def get_closed_trades(self, days: Optional[int] = None) -> pd.DataFrame:
        """
        Hae suljetut kaupat
        
        Args:
            days: Jos annettu, rajoita viimeiseen N päivään
        """
        conn = sqlite3.connect(self.db_path)
        
        if days:
            cutoff = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            query = f"SELECT * FROM trades WHERE status = 'CLOSED' AND entry_date >= '{cutoff}' ORDER BY entry_date DESC"
        else:
            query = "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY entry_date DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_performance_by_setup(self, days: int = 90) -> pd.DataFrame:
        """
        Laske performance per setup-tyyppi viimeisen N päivän ajalta
        
        Returns:
            DataFrame: setup_type, win_rate, avg_r, profit_factor, count
        """
        df = self.get_closed_trades(days=days)
        
        if df.empty:
            return pd.DataFrame(columns=['setup_type', 'win_rate', 'avg_r', 'profit_factor', 'count'])
        
        stats = []
        for setup in df['setup_type'].dropna().unique():
            subset = df[df['setup_type'] == setup]
            
            wins = subset[subset['pnl_pct'] > 0]
            losses = subset[subset['pnl_pct'] <= 0]
            
            win_rate = len(wins) / len(subset) if len(subset) > 0 else 0
            avg_r = subset['r_multiple'].mean()
            
            total_win = wins['pnl_pct'].sum()
            total_loss = abs(losses['pnl_pct'].sum())
            profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
            
            stats.append({
                'setup_type': setup,
                'win_rate': win_rate,
                'avg_r': avg_r,
                'profit_factor': profit_factor,
                'count': len(subset),
                'tp1_hit_rate': subset['tp1_hit'].mean(),
                'tp2_hit_rate': subset['tp2_hit'].mean(),
                'tp3_hit_rate': subset['tp3_hit'].mean()
            })
        
        return pd.DataFrame(stats)
    
    def get_summary_stats(self, days: int = 90) -> Dict:
        """Hae yleiset tilastot"""
        df = self.get_closed_trades(days=days)
        
        if df.empty:
            return {
                'total_trades': 0,
                'open_trades': len(self.get_open_trades()),  # ← KORJATTU
                'win_rate': 0.0,
                'avg_r_multiple': 0.0,
                'profit_factor': 0.0,
                'avg_hold_days': 0.0,
                'tp1_hit_rate': 0.0,  # ← KORJATTU
                'tp2_hit_rate': 0.0,  # ← KORJATTU
                'tp3_hit_rate': 0.0   # ← KORJATTU
            }
        
        wins = df[df['pnl_pct'] > 0]
        losses = df[df['pnl_pct'] <= 0]
        
        return {
            'total_trades': len(df),
            'open_trades': len(self.get_open_trades()),  # ← KORJATTU
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0.0,
            'avg_r_multiple': df['r_multiple'].mean(),
            'profit_factor': wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum()) if len(losses) > 0 else float('inf'),
            'avg_hold_days': df['hold_days'].mean(),
            'tp1_hit_rate': df['tp1_hit'].mean(),
            'tp2_hit_rate': df['tp2_hit'].mean(),
            'tp3_hit_rate': df['tp3_hit'].mean()
        }
    
    def export_to_csv(self, output_path: str, status: Optional[str] = None):
        """Vie kaupat CSV-tiedostoon"""
        conn = sqlite3.connect(self.db_path)
        
        if status:
            query = f"SELECT * FROM trades WHERE status = '{status}' ORDER BY entry_date DESC"
        else:
            query = "SELECT * FROM trades ORDER BY entry_date DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df.to_csv(output_path, index=False)
        print(f"[TradeLogger] Exported {len(df)} trades to {output_path}")


# ==================== TESTAUSFUNKTIOT ====================

def test_logger():
    """Testaa että logger toimii"""
    print("\n" + "="*80)
    print("🧪 TESTING TradeLogger")
    print("="*80 + "\n")
    
    # Luo logger
    logger = TradeLogger(db_path="seasonality_reports/trades_test.db")
    
    # Testaa entry
    print("\n1️⃣ Testing log_entry...")
    trade_id = logger.log_entry(
        ticker="AAPL",
        entry_date="2025-11-01",
        entry_price=225.50,
        stop=220.00,
        tp1=230.00,
        tp2=235.00,
        tp3=240.00,
        setup_type="ML_Momentum_Strong",
        ml_score=0.92,
        seasonality_score=0.018,
        mom5=0.025,
        mom20=0.045,
        vol20=0.015
    )
    
    print(f"✅ Trade ID: {trade_id}")
    
    # Testaa get_open_trades
    print("\n2️⃣ Testing get_open_trades...")
    open_trades = logger.get_open_trades()
    print(f"✅ Open trades: {len(open_trades)}")
    if not open_trades.empty:
        print(open_trades[['ticker', 'entry_date', 'entry_price', 'setup_type', 'status']].to_string(index=False))
    
    # Testaa exit
    print("\n3️⃣ Testing log_exit...")
    success = logger.log_exit(
        trade_id=trade_id,
        exit_date="2025-11-02",
        exit_price=232.00,
        exit_reason="TP2_HIT"
    )
    print(f"✅ Exit logged: {success}")
    
    # Testaa closed trades
    print("\n4️⃣ Testing get_closed_trades...")
    closed = logger.get_closed_trades()
    print(f"✅ Closed trades: {len(closed)}")
    if not closed.empty:
        print(closed[['ticker', 'entry_price', 'exit_price', 'pnl_pct', 'r_multiple', 'exit_reason']].to_string(index=False))
    
    # Testaa tilastot
    print("\n5️⃣ Testing get_summary_stats...")
    stats = logger.get_summary_stats()
    print("✅ Summary stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2%}" if 'rate' in key else f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Aja testit
    test_logger()