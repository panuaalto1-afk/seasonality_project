"""
Backtest Visualizer - Creates plots and charts
Generates comprehensive visualization suite
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """
    Creates visualizations for backtest results.
    
    Plots:
    1. Equity curve
    2. Drawdown chart
    3. Monthly returns heatmap
    4. Yearly returns bar chart
    5. Regime performance
    6. Sector performance
    7. Trade distribution
    8. Win/Loss analysis
    """
    
    def __init__(self, results: Dict, output_dir: Path):
        """Initialize visualizer with results."""
        self.results = results
        self.output_dir = output_dir
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.equity_curve = results['equity_curve']
        self.trades_history = results['trades_history']
        self.performance = results.get('performance', {})
    
    def plot_equity_curve(self):
        """Plot equity curve over time."""
        logger.info("Creating equity curve plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df = self.equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Plot equity
        ax.plot(df['date'], df['total_value'], linewidth=2, label='Portfolio Value', color='#2E86AB')
        
        # Add initial value line
        initial = df['total_value'].iloc[0]
        ax.axhline(y=initial, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Initial Value')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax.set_title('Portfolio Equity Curve - Aggressive Strategy', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add performance text
        final = df['total_value'].iloc[-1]
        total_return = (final / initial - 1) * 100
        
        textstr = f'Initial: ${initial:,.0f}\nFinal: ${final:,.0f}\nReturn: {total_return:+.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Equity curve plot saved")
    
    def plot_drawdown(self):
        """Plot drawdown chart."""
        logger.info("Creating drawdown plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df = self.equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate drawdown
        cumulative = df['total_value']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        
        # Plot
        ax.fill_between(df['date'], drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax.plot(df['date'], drawdown, linewidth=2, color='darkred')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add max drawdown line
        max_dd = drawdown.min()
        ax.axhline(y=max_dd, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(df['date'].iloc[-1], max_dd, f'  Max DD: {max_dd:.2f}%', 
                verticalalignment='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'drawdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Drawdown plot saved")
    
    def plot_monthly_returns_heatmap(self):
        """Plot monthly returns as heatmap."""
        logger.info("Creating monthly returns heatmap...")
        
        monthly_df = self.performance.get('monthly_returns', pd.DataFrame())
        
        if monthly_df.empty:
            logger.warning("No monthly returns data available")
            return
        
        # Pivot for heatmap
        monthly_df['year'] = monthly_df['year_month'].dt.year
        monthly_df['month'] = monthly_df['year_month'].dt.month
        
        pivot = monthly_df.pivot(index='month', columns='year', values='return_pct')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Return (%)'}, linewidths=0.5,
                    vmin=-10, vmax=10, ax=ax)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Month', fontsize=12, fontweight='bold')
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticklabels(month_names, rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'monthly_returns_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Monthly returns heatmap saved")
    
    def plot_yearly_returns(self):
        """Plot yearly returns as bar chart."""
        logger.info("Creating yearly returns plot...")
        
        yearly_df = self.performance.get('yearly_breakdown', pd.DataFrame())
        
        if yearly_df.empty:
            logger.warning("No yearly breakdown data available")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Bar chart
        colors = ['green' if x > 0 else 'red' for x in yearly_df['return_pct']]
        bars = ax.bar(yearly_df['year'], yearly_df['return_pct'], color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Yearly Returns', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'yearly_returns.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Yearly returns plot saved")
    
    def plot_regime_performance(self):
        """Plot performance by regime."""
        logger.info("Creating regime performance plot...")
        
        regime_df = self.performance.get('regime_breakdown', pd.DataFrame())
        
        if regime_df.empty:
            logger.warning("No regime breakdown data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Win rate by regime
        regime_df_sorted = regime_df.sort_values('win_rate', ascending=False)
        colors = plt.cm.RdYlGn(regime_df_sorted['win_rate'] / 100)
        
        ax1.barh(regime_df_sorted['regime'], regime_df_sorted['win_rate'], color=colors, edgecolor='black')
        ax1.set_xlabel('Win Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Regime', fontsize=12, fontweight='bold')
        ax1.set_title('Win Rate by Regime', fontsize=13, fontweight='bold')
        ax1.axvline(x=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (regime, val) in enumerate(zip(regime_df_sorted['regime'], regime_df_sorted['win_rate'])):
            ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)
        
        # Plot 2: Total P&L by regime
        regime_df_sorted2 = regime_df.sort_values('total_pnl', ascending=False)
        colors2 = ['green' if x > 0 else 'red' for x in regime_df_sorted2['total_pnl']]
        
        ax2.barh(regime_df_sorted2['regime'], regime_df_sorted2['total_pnl'], color=colors2, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Total P&L ($)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Regime', fontsize=12, fontweight='bold')
        ax2.set_title('Total P&L by Regime', fontsize=13, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (regime, val) in enumerate(zip(regime_df_sorted2['regime'], regime_df_sorted2['total_pnl'])):
            ax2.text(val + 500 if val > 0 else val - 500, i, f'${val:,.0f}', 
                    va='center', ha='left' if val > 0 else 'right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'regime_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Regime performance plot saved")
    
    def plot_sector_performance(self):
        """Plot performance by sector."""
        logger.info("Creating sector performance plot...")
        
        sector_df = self.performance.get('sector_breakdown', pd.DataFrame())
        
        if sector_df.empty:
            logger.warning("No sector breakdown data available")
            return
        
        # Take top 15 sectors by total return
        sector_df_top = sector_df.nlargest(15, 'total_return')
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors = ['green' if x > 0 else 'red' for x in sector_df_top['total_return']]
        bars = ax.barh(sector_df_top['sector'], sector_df_top['total_return'], 
                      color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Total Return ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sector', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Sectors by Total Return', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (sector, val, trades, wr) in enumerate(zip(
            sector_df_top['sector'], 
            sector_df_top['total_return'],
            sector_df_top['num_trades'],
            sector_df_top['win_rate']
        )):
            label = f'${val:,.0f} ({trades} trades, {wr:.0f}% WR)'
            ax.text(val + 200 if val > 0 else val - 200, i, label,
                   va='center', ha='left' if val > 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sector_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Sector performance plot saved")
    
    def plot_trade_distribution(self):
        """Plot distribution of trade returns."""
        logger.info("Creating trade distribution plot...")
        
        if self.trades_history.empty:
            logger.warning("No trades history available")
            return
        
        sells = self.trades_history[self.trades_history['action'] == 'SELL']
        
        if sells.empty or 'pnl_pct' not in sells.columns:
            logger.warning("No sell trades with P&L data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of returns
        ax1.hist(sells['pnl_pct'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=sells['pnl_pct'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
        ax1.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Trade Returns', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_returns = np.sort(sells['pnl_pct'])
        cumulative = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns) * 100
        
        ax2.plot(sorted_returns, cumulative, linewidth=2, color='steelblue')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax2.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Probability (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Distribution of Returns', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'trade_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Trade distribution plot saved")
    
    def plot_win_loss_analysis(self):
        """Plot win/loss analysis."""
        logger.info("Creating win/loss analysis plot...")
        
        if self.trades_history.empty:
            logger.warning("No trades history available")
            return
        
        sells = self.trades_history[self.trades_history['action'] == 'SELL']
        
        if sells.empty:
            return
        
        wins = sells[sells['pnl'] > 0]
        losses = sells[sells['pnl'] <= 0]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Win/Loss count
        counts = [len(wins), len(losses)]
        labels = [f'Wins ({len(wins)})', f'Losses ({len(losses)})']
        colors = ['green', 'red']
        
        ax1.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Win vs Loss Distribution', fontsize=13, fontweight='bold')
        
        # 2. Average win vs loss (absolute)
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        ax2.bar(['Avg Win', 'Avg Loss'], [avg_win, avg_loss], color=['green', 'red'], alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Average P&L ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Win vs Loss (Dollars)', fontsize=13, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (label, val) in enumerate(zip(['Avg Win', 'Avg Loss'], [avg_win, avg_loss])):
            ax2.text(i, val + 50 if val > 0 else val - 50, f'${val:,.2f}',
                    ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')
        
        # 3. Average win vs loss (percentage)
        avg_win_pct = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss_pct = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        
        ax3.bar(['Avg Win', 'Avg Loss'], [avg_win_pct, avg_loss_pct], color=['green', 'red'], alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Average Return (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Average Win vs Loss (Percentage)', fontsize=13, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, (label, val) in enumerate(zip(['Avg Win', 'Avg Loss'], [avg_win_pct, avg_loss_pct])):
            ax3.text(i, val + 0.5 if val > 0 else val - 0.5, f'{val:.2f}%',
                    ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')
        
        # 4. Hold time comparison
        if 'days_held' in sells.columns:
            avg_hold_win = wins['days_held'].mean() if len(wins) > 0 else 0
            avg_hold_loss = losses['days_held'].mean() if len(losses) > 0 else 0
            
            ax4.bar(['Wins', 'Losses'], [avg_hold_win, avg_hold_loss], color=['green', 'red'], alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Average Hold Time (Days)', fontsize=12, fontweight='bold')
            ax4.set_title('Hold Time: Wins vs Losses', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for i, (label, val) in enumerate(zip(['Wins', 'Losses'], [avg_hold_win, avg_hold_loss])):
                ax4.text(i, val + 0.5, f'{val:.1f} days',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'win_loss_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Win/loss analysis plot saved")
    
    def create_all_plots(self):
        """Create all visualization plots."""
        logger.info("\n" + "=" * 60)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 60)
        
        plots_created = []
        
        try:
            self.plot_equity_curve()
            plots_created.append("Equity Curve")
        except Exception as e:
            logger.error(f"Error creating equity curve: {str(e)}")
        
        try:
            self.plot_drawdown()
            plots_created.append("Drawdown Chart")
        except Exception as e:
            logger.error(f"Error creating drawdown plot: {str(e)}")
        
        try:
            self.plot_monthly_returns_heatmap()
            plots_created.append("Monthly Returns Heatmap")
        except Exception as e:
            logger.error(f"Error creating monthly heatmap: {str(e)}")
        
        try:
            self.plot_yearly_returns()
            plots_created.append("Yearly Returns")
        except Exception as e:
            logger.error(f"Error creating yearly returns: {str(e)}")
        
        try:
            self.plot_regime_performance()
            plots_created.append("Regime Performance")
        except Exception as e:
            logger.error(f"Error creating regime plot: {str(e)}")
        
        try:
            self.plot_sector_performance()
            plots_created.append("Sector Performance")
        except Exception as e:
            logger.error(f"Error creating sector plot: {str(e)}")
        
        try:
            self.plot_trade_distribution()
            plots_created.append("Trade Distribution")
        except Exception as e:
            logger.error(f"Error creating trade distribution: {str(e)}")
        
        try:
            self.plot_win_loss_analysis()
            plots_created.append("Win/Loss Analysis")
        except Exception as e:
            logger.error(f"Error creating win/loss plot: {str(e)}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"CREATED {len(plots_created)} PLOTS:")
        for plot in plots_created:
            logger.info(f"  ✓ {plot}")
        logger.info(f"\nPlots saved to: {self.plots_dir}")
        logger.info("=" * 60)