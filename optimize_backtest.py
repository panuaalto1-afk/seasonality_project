"""
OPTIMIZED BACKTEST ANALYZER - Option 1
======================================
Post-process simulation results with:
- $2,000 position size
- Max 3 trades per day (best signals)
- Max 15 simultaneous positions
- 0.10% commission (min €5) per side
- Optimized stop/take profit levels

Usage: python optimize_backtest.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ============================================================================
# CONFIGURATION - Option 1
# ============================================================================
CONFIG = {
    'position_size': 2000,          # $2,000 per trade
    'max_trades_per_day': 3,        # Max 3 best trades per day
    'max_positions': 15,             # Max 15 simultaneous positions
    'commission_pct': 0.001,         # 0.10%
    'commission_min_eur': 5.0,       # Min €5
    'stop_mult': 1.0,                # Tighter stop (from 1.5)
    'tp_mult': 3.0,                  # Same take profit
}

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("  🎯 OPTIMIZED BACKTEST - Option 1 ($2,000/position)")
print("=" * 70)

input_file = 'seasonality_reports/backtest_results/simulated_trades.csv'
try:
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} trades from: {input_file}")
except FileNotFoundError:
    print(f"❌ Error: File not found: {input_file}")
    print("   Run this first: python advanced_backtest_analyzer.py --mode simulation")
    exit(1)

df_original = df.copy()

# ============================================================================
# STEP 1: Filter to max 3 trades per day (best signal_strength)
# ============================================================================
print("\n[1/6] 📅 Filtering to max 3 trades per day...")

df['entry_date'] = pd.to_datetime(df['entry_date'])
df['exit_date'] = pd.to_datetime(df['exit_date'])

# Sort by date and signal strength
df = df.sort_values(['entry_date', 'signal_strength'], ascending=[True, False])

# Keep only top 3 per day
df = df.groupby(df['entry_date'].dt.date).head(CONFIG['max_trades_per_day'])

print(f"   Original: {len(df_original)} trades")
print(f"   Filtered: {len(df)} trades (max {CONFIG['max_trades_per_day']}/day)")

# ============================================================================
# STEP 2: Simulate max 15 positions simultaneously
# ============================================================================
print("\n[2/6] 🎯 Simulating max 15 simultaneous positions...")

df = df.sort_values('entry_date').reset_index(drop=True)
active_positions = []
selected_indices = []

for idx, trade in df.iterrows():
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    
    # Remove positions that closed before this entry
    active_positions = [p for p in active_positions if p['exit_date'] > entry_date]
    
    # Check if we can add this trade
    if len(active_positions) < CONFIG['max_positions']:
        active_positions.append({
            'ticker': trade['ticker'],
            'entry_date': entry_date,
            'exit_date': exit_date
        })
        selected_indices.append(idx)

df = df.loc[selected_indices].copy()
print(f"   Selected: {len(df)} trades (max {CONFIG['max_positions']} positions)")

# ============================================================================
# STEP 3: Recalculate with new position size ($2,000)
# ============================================================================
print("\n[3/6] 💰 Recalculating with $2,000 position size...")

df['position_size'] = CONFIG['position_size']
df['shares'] = df['position_size'] / df['entry_price']

# Recalculate stop loss and take profit with new multipliers
# Note: ATR is implicit in the original stop/take profit levels
# We'll scale them based on the new multipliers
if CONFIG['stop_mult'] != 1.5 or CONFIG['tp_mult'] != 3.0:
    # Original multipliers were 1.5 and 3.0
    df['stop_loss'] = df['entry_price'] - (df['entry_price'] - df['stop_loss']) * (CONFIG['stop_mult'] / 1.5)
    df['take_profit'] = df['entry_price'] + (df['take_profit'] - df['entry_price']) * (CONFIG['tp_mult'] / 3.0)
    print(f"   ⚙️  Adjusted stop/TP: SL={CONFIG['stop_mult']}×ATR, TP={CONFIG['tp_mult']}×ATR")

# ============================================================================
# STEP 4: Add commissions (0.10%, min €5 per side)
# ============================================================================
print("\n[4/6] 💸 Adding commissions (0.10%, min €5)...")

# Entry commission
entry_value = df['entry_price'] * df['shares']
df['entry_commission'] = np.maximum(entry_value * CONFIG['commission_pct'], CONFIG['commission_min_eur'])

# Exit commission
exit_value = df['exit_price'] * df['shares']
df['exit_commission'] = np.maximum(exit_value * CONFIG['commission_pct'], CONFIG['commission_min_eur'])

# Total commission
df['total_commission'] = df['entry_commission'] + df['exit_commission']

print(f"   Total commissions: ${df['total_commission'].sum():,.2f}")

# ============================================================================
# STEP 5: Calculate P&L (gross and net)
# ============================================================================
print("\n[5/6] 📊 Calculating P&L...")

df['gross_pnl'] = (df['exit_price'] - df['entry_price']) * df['shares']
df['net_pnl'] = df['gross_pnl'] - df['total_commission']
df['net_pnl_pct'] = (df['net_pnl'] / df['position_size']) * 100

# ============================================================================
# STEP 6: Calculate Metrics
# ============================================================================
print("\n[6/6] 📈 Calculating metrics...")

total_trades = len(df)
total_gross_pnl = df['gross_pnl'].sum()
total_commission = df['total_commission'].sum()
total_net_pnl = df['net_pnl'].sum()

wins = df[df['net_pnl'] > 0]
losses = df[df['net_pnl'] <= 0]

win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0

profit_factor = abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf')

# Sharpe ratio (annualized)
returns = df['net_pnl_pct'].values
avg_return = returns.mean()
std_return = returns.std()
avg_holding_days = df['holding_days'].mean()
sharpe = (avg_return / std_return) * np.sqrt(252 / avg_holding_days) if std_return > 0 else 0

# Sortino ratio (downside deviation)
downside_returns = returns[returns < 0]
downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.01
sortino = (avg_return / downside_std) * np.sqrt(252 / avg_holding_days) if downside_std > 0 else 0

# Max drawdown
df = df.sort_values('exit_date')
df['cumulative_pnl'] = df['net_pnl'].cumsum()
running_max = df['cumulative_pnl'].expanding().max()
drawdown = df['cumulative_pnl'] - running_max
max_drawdown = drawdown.min()

# Calculate max drawdown percentage (relative to peak equity)
initial_capital = CONFIG['position_size'] * CONFIG['max_positions']  # $30,000
peak_equity = initial_capital + running_max.max()
max_drawdown_pct = (max_drawdown / peak_equity) * 100 if peak_equity > 0 else 0

# ============================================================================
# PRINT RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("  📊 RESULTS SUMMARY - Option 1")
print("=" * 70)

print("\n💼 PORTFOLIO CONFIGURATION:")
print(f"   Position Size:       ${CONFIG['position_size']:,}")
print(f"   Max Positions:       {CONFIG['max_positions']}")
print(f"   Max Trades/Day:      {CONFIG['max_trades_per_day']}")
print(f"   Stop Loss:           {CONFIG['stop_mult']}× ATR")
print(f"   Take Profit:         {CONFIG['tp_mult']}× ATR")
print(f"   Commission:          {CONFIG['commission_pct']*100}% (min €{CONFIG['commission_min_eur']})")

print("\n💰 P&L:")
print(f"   Total Gross P&L:     ${total_gross_pnl:>10,.2f}")
print(f"   Total Commissions:   ${total_commission:>10,.2f}")
print(f"   Total Net P&L:       ${total_net_pnl:>10,.2f}")

print("\n📈 PERFORMANCE METRICS:")
print(f"   Total Trades:        {total_trades:>10}")
print(f"   Win Rate:            {win_rate:>10.1f}%")
print(f"   Profit Factor:       {profit_factor:>10.2f}")
print(f"   Sharpe Ratio:        {sharpe:>10.2f}")
print(f"   Sortino Ratio:       {sortino:>10.2f}")
print(f"   Max Drawdown:        ${max_drawdown:>10,.2f} ({max_drawdown_pct:.1f}%)")
print(f"   Avg Holding:         {avg_holding_days:>10.1f} days")

print("\n🏆 TRADE BREAKDOWN:")
print(f"   Winning Trades:      {len(wins):>10} (avg: ${avg_win:>10,.2f})")
print(f"   Losing Trades:       {len(losses):>10} (avg: ${avg_loss:>10,.2f})")

print("\n🚪 EXIT REASONS:")
exit_reasons = df['exit_reason'].value_counts()
for reason, count in exit_reasons.items():
    pct = (count / total_trades) * 100
    print(f"   {reason:15} {count:>4} ({pct:>5.1f}%)")

# ============================================================================
# COMPARISON WITH ORIGINAL
# ============================================================================
print("\n" + "=" * 70)
print("  📊 COMPARISON: Original vs. Option 1")
print("=" * 70)

original_pnl = df_original['pnl'].sum()
original_trades = len(df_original)
original_win_rate = (df_original['pnl'] > 0).mean() * 100

print(f"\n{'Metric':<25} {'Original':<15} {'Option 1':<15} {'Change':<15}")
print("-" * 70)
print(f"{'Total Trades':<25} {original_trades:<15} {total_trades:<15} {total_trades - original_trades:+d}")
print(f"{'P&L':<25} ${original_pnl:<14,.2f} ${total_net_pnl:<14,.2f} ${total_net_pnl - original_pnl:+,.2f}")
print(f"{'Win Rate':<25} {original_win_rate:<14.1f}% {win_rate:<14.1f}% {win_rate - original_win_rate:+.1f}%")
print(f"{'P&L per Trade':<25} ${original_pnl/original_trades:<14,.2f} ${total_net_pnl/total_trades:<14,.2f} ${(total_net_pnl/total_trades) - (original_pnl/original_trades):+,.2f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_csv = 'seasonality_reports/backtest_results/optimized_trades_option1.csv'
output_json = 'seasonality_reports/backtest_results/optimized_results_option1.json'

df.to_csv(output_csv, index=False)

results = {
    'config': CONFIG,
    'total_trades': total_trades,
    'total_gross_pnl': float(total_gross_pnl),
    'total_commission': float(total_commission),
    'total_net_pnl': float(total_net_pnl),
    'win_rate': float(win_rate),
    'avg_win': float(avg_win),
    'avg_loss': float(avg_loss),
    'profit_factor': float(profit_factor),
    'sharpe_ratio': float(sharpe),
    'sortino_ratio': float(sortino),
    'max_drawdown': float(max_drawdown),
    'max_drawdown_pct': float(max_drawdown_pct),
    'avg_holding_days': float(avg_holding_days),
    'exit_reasons': exit_reasons.to_dict()
}

with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("  💾 FILES SAVED")
print("=" * 70)
print(f"   Trades:  {output_csv}")
print(f"   Metrics: {output_json}")

print("\n" + "=" * 70)
print("  ✅ OPTIMIZATION COMPLETE!")
print("=" * 70)