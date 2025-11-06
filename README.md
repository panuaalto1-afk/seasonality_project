# 🌟 Seasonality Trading System - Complete Technical Documentation

**Project Owner:** panuaalto1-afk  
**Repository:** https://github.com/panuaalto1-afk/seasonality_project  
**Last Updated:** 2025-11-06 18:59 UTC  
**Python Version:** 3.11+  
**Trading Universe:** S&P 500 constituents (~500 stocks)  
**Trading Strategy:** Long/Short seasonality + ML-driven momentum + Inverse ETF hedging

---

## 📋 Table of Contents

1. [Directory Structure](#-complete-directory-structure)
2. [Daily Automated Schedule](#-daily-automated-schedule-weekdays-only)
3. [Data Flow Diagram](#-complete-data-flow-diagram)
4. [Inverse ETF System](#-inverse-etf-trading-system) ⭐ NEW
5. [Regime Detection](#-regime-based-position-limits)
6. [Stop Loss & Take Profit](#-stop-loss--take-profit-calculation-atr-based)
7. [Portfolio State](#-portfolio-state-schema)
8. [Command Line Usage](#-command-line-usage)
9. [Troubleshooting](#-troubleshooting)

---

## 📁 Complete Directory Structure

```
C:\Users\panua\seasonality_project\
│
├── 📊 PRICE DATA (Two Separate Caches)
│   │
│   ├── seasonality_reports\runs\2025-10-04_0903\price_cache\
│   │   ├── AMD.csv                    # 20-year history, OVERWRITE daily (10:00 UTC)
│   │   ├── AAPL.csv                   # All stock prices (516 files)
│   │   ├── NVDA.csv                   # Updated by: build_prices_from_constituents.py
│   │   ├── SH.csv                     # ⭐ Inverse ETFs (NEW)
│   │   ├── PSQ.csv                    # ⭐ Added automatically in bearish regimes
│   │   ├── DOG.csv
│   │   ├── RWM.csv
│   │   └── ...                        # Used by: ml_unified_pipeline.py, auto_decider.py
│   │
│   └── seasonality_reports\price_cache\
│       ├── ^SPX.csv                   # Index prices, updated 12:00 UTC
│       ├── ^VIX.csv                   # Updated by: build_prices_from_indexes.py
│       ├── TLT.csv                    # Used by: regime_detector.py
│       ├── GLD.csv
│       └── ...
│
├── 🤖 ML PIPELINE & FEATURES
│   ├── ml_unified_pipeline.py         # Main orchestrator (11:00 UTC)
│   │                                  # ⭐ Auto-adds inverse ETFs in bearish regimes
│   ├── ml_features.py                 # Technical indicators (RSI, MACD, ATR, BB, etc.)
│   ├── ml_sector_features.py          # Sector rotation signals
│   ├── ml_sector_rotation.py          # Sector strength ranking
│   ├── ml_models.py                   # XGBoost/LightGBM models
│   └── seasonality_calc.py            # Historical seasonality patterns
│
├── 📈 REGIME DETECTION
│   ├── regime_detector.py             # Market regime classifier
│   ├── regime_strategies.py           # Regime-specific configs
│   │                                  # ⭐ Includes inverse ETF allocations
│   └── seasonality_reports\aggregates\regime_research\
│       └── 2025-10-17\                # Regime analysis results
│
├── 🎯 TRADE DECISION ENGINE
│   ├── auto_decider.py                # Main automation (15:55 UTC) ⚡
│   │                                  # ⭐ CRISIS mode: Exit longs, buy inverse ETFs
│   │                                  # ⭐ Bearish modes: Include inverse ETFs in candidates
│   ├── make_exit_watchlist.py         # Stop-loss monitoring (16:05 UTC)
│   └── send_trades_email.py           # Email notifications (automatic)
│
├── 📊 OPTIONS STRATEGIES (Separate Pipeline)
│   ├── optio_seasonality_signal.py    # Generate signals (15:00 UTC)
│   ├── optio_seasonality_price_enricher.py  # Enrich prices (15:30 UTC)
│   ├── optio_unified_daily.py         # Unified pipeline (15:30 UTC)
│   └── seasonality_reports\aggregates\
│       ├── optio_signals\2025-11-06\  # Daily options signals
│       │   ├── top_breakout_long.csv
│       │   ├── top_breakout_short.csv
│       │   └── exit_alerts.csv
│       │
│       └── optio_signals_enriched\2025-11-06\
│           └── optio_price_enriched_*.csv  # Priced options
│
├── 📋 SEASONALITY DATA
│   ├── aggregate_seasonality_picker.py  # Daily aggregation (12:00 UTC)
│   ├── us_seasonality_full.py         # Monthly full rebuild (02:00 UTC, 20th)
│   │                                  # ⭐ Auto-adds inverse ETFs to universe
│   └── seasonality_reports\
│       ├── us_seasonality_*.csv       # Seasonality patterns (root level)
│       └── aggregates\
│           └── segments\2025-11-06\   # Ticker pools
│               └── tickers_pool.csv
│
├── 🗂️ DAILY RUNS & OUTPUTS
│   └── seasonality_reports\runs\
│       ├── 2025-11-06_0000\           # Today's run
│       │   ├── reports\
│       │   │   ├── features_2025-11-06.csv
│       │   │   ├── labels_2025-11-06.csv
│       │   │   ├── top_long_candidates_RAW_2025-11-06.csv
│       │   │   ├── top_long_candidates_GATED_2025-11-06.csv  ← AUTO_DECIDER INPUT ⚡
│       │   │   ├── top_short_candidates_RAW_2025-11-06.csv
│       │   │   └── top_short_candidates_GATED_2025-11-06.csv
│       │   │
│       │   └── actions\20251106\
│       │       ├── trade_candidates.csv      # BUY orders (EntryPrice, Stop/TP)
│       │       │                             # ⭐ May include inverse ETFs in CRISIS
│       │       ├── sell_candidates.csv       # SELL orders (CurrentPrice, Stop/TP)
│       │       ├── action_plan.txt           # Human-readable summary
│       │       ├── portfolio_after_sim.csv   # Expected portfolio
│       │       └── exit_watchlist.csv        # Stop-loss monitoring
│       │
│       └── 2025-10-04_0903\
│           └── price_cache\          # ← CANONICAL STOCK PRICE CACHE
│               └── *.csv             #    (516 stocks + 4 inverse ETFs)
│
├── 🧪 TESTING & UTILITIES
│   ├── test_inverse_etfs.py          # ⭐ Test inverse ETF imports
│   ├── test_crisis_scenario.py       # ⭐ Simulate CRISIS mode
│   ├── inverse_etf_downloader.py     # ⭐ Download inverse ETF prices
│   ├── advanced_backtest_analyzer.py
│   ├── backtest_utils.py
│   └── backtest_visualizer.py
│
└── 📋 CONFIGURATION
    ├── portfolio_state.json          # Current positions (CRITICAL!) ⚠️
    ├── .env                          # Email credentials (gitignored)
    └── .gitignore
```

---

## ⏰ Daily Automated Schedule (Weekdays Only)

| Time (UTC) | Time (ET) | Script | Purpose | Output |
|------------|-----------|--------|---------|--------|
| **02:00** (20th) | 21:00 (19th) | `us_seasonality_full.py` | Monthly full seasonality rebuild<br>⭐ **Adds inverse ETFs to universe** | `us_seasonality_*.csv` |
| **10:00** ⚡ | 05:00 | `build_prices_from_constituents.py` | Download stock prices (516 tickers)<br>⭐ **Includes SH, PSQ, DOG, RWM** | `2025-10-04_0903/price_cache/*.csv` |
| **11:00** ⚡ | 06:00 | `ml_unified_pipeline.py` | **Generate ML signals**<br>⭐ **Auto-adds inverse ETFs in bearish regimes** | `top_long_candidates_GATED_*.csv` |
| **12:00** | 07:00 | `build_prices_from_indexes.py` | Download index prices (SPX, VIX, TLT...) | `price_cache/^*.csv` |
| **12:00** | 07:00 | `aggregate_seasonality_picker.py` | Daily seasonality aggregation | `seasonality_agg_*.csv` |
| **15:00** | 10:00 | `optio_seasonality_signal.py` | Generate options signals | `top_breakout_*.csv` |
| **15:30** | 10:30 | `optio_seasonality_price_enricher.py` | Enrich options with prices | `optio_price_enriched_*.csv` |
| **15:30** | 10:30 | `optio_unified_daily.py` | Unified options pipeline | Final options candidates |
| **15:55** ⚡⚡⚡ | 10:55 | `auto_decider.py` | **STOCK TRADE DECISIONS**<br>⭐ **CRISIS: Exit longs, buy inverse ETFs** | `trade_candidates.csv`, `sell_candidates.csv` |
| **16:05** | 11:05 | `make_exit_watchlist.py` | Generate stop-loss alerts | `exit_watchlist.csv` |

**⏱️ Market Opens:** 09:30 ET (14:30 UTC) - Auto_decider completes 5 minutes **before** open

---

## 🔄 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ OVERNIGHT: Price Data Collection                                         │
└─────────────────────────────────────────────────────────────────────────┘

[02:00 UTC - Monthly 20th]
us_seasonality_full.py
         ↓
  ⭐ Add inverse ETFs to universe
         ├── INVERSE_ETFS = ['SH', 'PSQ', 'DOG', 'RWM']
         └── universe = list(set(universe + INVERSE_ETFS))
         ↓
  Full seasonality database rebuild

[10:00 UTC Daily] ⚡
build_prices_from_constituents.py
         ↓
  C:\...\runs\2025-10-04_0903\price_cache\
         ├── AMD.csv (OVERWRITE 20yr history)
         ├── AAPL.csv
         ├── SH.csv   ⭐ (Inverse S&P 500)
         ├── PSQ.csv  ⭐ (Inverse Nasdaq)
         ├── DOG.csv  ⭐ (Inverse Dow 30)
         ├── RWM.csv  ⭐ (Inverse Russell 2000)
         └── ... (520 total tickers)

[12:00 UTC Daily]
build_prices_from_indexes.py
         ↓
  C:\...\seasonality_reports\price_cache\
         ├── ^SPX.csv (OVERWRITE)
         ├── ^VIX.csv
         ├── TLT.csv
         └── ...

[12:00 UTC Daily]
aggregate_seasonality_picker.py
         ↓
  seasonality_agg_2025-11-06.csv


┌─────────────────────────────────────────────────────────────────────────┐
│ MORNING: ML Signal Generation (11:00 UTC)                                │
└─────────────────────────────────────────────────────────────────────────┘

[11:00 UTC] ⚡ CRITICAL PATH
ml_unified_pipeline.py
         ├── Reads: 2025-10-04_0903/price_cache/*.csv
         ├── Detects regime: regime_detector.py
         │    └── If bearish → Auto-add inverse ETFs
         │
         ├── ⭐ REGIME-AWARE UNIVERSE:
         │    ├── BULL: 516 stocks only
         │    ├── NEUTRAL_BEARISH: 516 stocks + SH, PSQ
         │    ├── BEAR_WEAK: 516 stocks + SH, PSQ, DOG
         │    └── CRISIS: 516 stocks + SH, PSQ
         │
         ├── Calculates: mom5, mom20, mom60, vol20, ATR
         ├── Generates: Composite scores (0-1 ranking)
         └── Outputs:
             ├── features_2025-11-06.csv
             ├── top_long_candidates_RAW_2025-11-06.csv  (200 stocks)
             └── top_long_candidates_GATED_2025-11-06.csv  (filtered) ← AUTO_DECIDER INPUT


┌─────────────────────────────────────────────────────────────────────────┐
│ PRE-OPEN: Trade Decision Engine (15:55 UTC) ⚡⚡⚡                        │
└─────────────────────────────────────────────────────────────────────────┘

[15:55 UTC] ⚡ MOST CRITICAL SCRIPT ⚡

auto_decider.py
         │
         ├─[INPUT 1]─→ top_long_candidates_GATED_2025-11-06.csv
         │             (from ml_unified_pipeline.py)
         │
         ├─[INPUT 2]─→ portfolio_state.json
         │             (current 8 positions)
         │
         ├─[INPUT 3]─→ regime_detector.py
         │             ├── Reads: price_cache/^SPX.csv, ^VIX.csv, TLT.csv
         │             └── Returns: NEUTRAL_BULLISH / CRISIS / etc.
         │
         ├─[LOGIC]────→ Decide BUY/SELL/HOLD
         │             │
         │             ├── ⭐ CRISIS MODE LOGIC:
         │             │    ├── Separate: inverse_etfs vs longs
         │             │    │    └── inverse = {SH, PSQ, DOG, RWM}
         │             │    ├── SELL: All long positions
         │             │    ├── BUY: Inverse ETFs (80% allocation)
         │             │    └── HOLD: Existing inverse ETFs
         │             │
         │             ├── ⭐ BEARISH MODE LOGIC:
         │             │    ├── Include inverse ETFs in candidate pool
         │             │    ├── Allocate: 20-60% to inverse ETFs
         │             │    └── Reduce: Long positions to 2-6 max
         │             │
         │             └── NORMAL MODE:
         │                  ├── Compare: Portfolio vs. Top-8 candidates
         │                  ├── Regime filter: NEUTRAL_BULLISH → max 8 pos, 90% size
         │                  └── Calculate: Stop Loss & Take Profit (ATR-based)
         │
         └─[OUTPUTS]──→ actions/20251106/
                        ├── trade_candidates.csv       ← BUY orders + Stop/TP
                        │                              ⭐ May include SH, PSQ in CRISIS
                        ├── sell_candidates.csv        ← SELL orders + Stop/TP
                        ├── action_plan.txt            ← Human summary
                        └── portfolio_after_sim.csv    ← Expected state

         ↓

send_trades_email.py (AUTOMATIC)
         ↓
  📧 Email to: panu.aalto1@gmail.com
     Attachments: trade_candidates.csv, sell_candidates.csv, action_plan.txt
```

---

## 🛡️ Inverse ETF Trading System

### Overview
Automatic inverse ETF allocation during bearish market regimes for portfolio protection. **Fully integrated** with regime detection and auto_decider logic.

### Supported Inverse ETFs

| Ticker | Name | Tracks | Leverage | Expense Ratio |
|--------|------|--------|----------|---------------|
| **SH** | ProShares Short S&P 500 | Inverse SPY | 1x | 0.89% |
| **PSQ** | ProShares Short QQQ | Inverse Nasdaq | 1x | 0.95% |
| **DOG** | ProShares Short Dow 30 | Inverse DIA | 1x | 0.95% |
| **RWM** | ProShares Short Russell 2000 | Inverse IWM | 1x | 0.95% |

**⚠️ Note:** Using 1x leverage for stability. 3x leveraged ETFs (SQQQ, SPXS) can be added for aggressive strategies.

### Allocation by Regime

| Regime | Short % | Max Positions | Inverse ETFs | Strategy |
|--------|---------|---------------|--------------|----------|
| **CRISIS** | **80%** | 2 | SH, PSQ | Exit all longs, buy inverse ETFs |
| **BEAR_STRONG** | 60% | 2 | SH, PSQ, DOG | Defensive + inverse hedging |
| **BEAR_WEAK** | 40% | 4 | SH, PSQ, DOG | Mean reversion + hedging |
| **NEUTRAL_BEARISH** | 20% | 6 | SH, PSQ | Cautious with small hedge |
| **NEUTRAL_BULLISH** | 0% | 8 | None | No shorts |
| **BULL_WEAK** | 0% | 10 | None | Selective momentum |
| **BULL_STRONG** | 0% | 12 | None | Full momentum |

### CRISIS Mode Example

**Scenario:** Market crashes, VIX > 40, SPX < 200-day MA

**Initial Portfolio:**
```
Cash: $50,000
AAPL: $15,000 (100 shares @ $150)
MSFT: $15,000 (50 shares @ $300)
Total: $80,000
```

**auto_decider.py Actions (15:55 UTC):**
```
[CRISIS MODE] Exiting all long positions

SELL:
- AAPL: 100 shares @ $150.00 → $15,000 cash
- MSFT: 50 shares @ $300.00 → $15,000 cash
Reason: CRISIS_EXIT_LONGS

BUY (80% allocation = $64,000):
- SH: 868 shares @ $36.90 → $32,000 (40% allocation)
- PSQ: 1,049 shares @ $30.50 → $32,000 (40% allocation)
Reason: INVERSE_ETF_CRISIS_80%

HOLD:
- Cash: $16,000 (20%)
```

**Result:**
- ✅ Protected against market decline
- ✅ Profit if S&P 500 drops (e.g., -10% market = +10% SH gain)
- ✅ Maintain liquidity for opportunities

### Bearish Mode Example

**Scenario:** NEUTRAL_BEARISH regime (score: -0.05)

**Portfolio Before:**
```
8 positions (AMD, AAPL, MSFT, NVDA, GOOGL, META, TSLA, CRM)
```

**auto_decider.py Actions:**
```
[NEUTRAL_BEARISH MODE] Max 6 positions, 20% inverse allocation

SELL (weakest 2 longs):
- META: Sell (ml_score dropped)
- CRM: Sell (momentum fading)

BUY (inverse ETFs):
- SH: $8,000 (10% allocation)
- PSQ: $8,000 (10% allocation)

HOLD (strongest 4 longs):
- NVDA, AAPL, MSFT, AMD (top ml_scores)
```

**Result:**
- ✅ Reduced long exposure (6 → 4 positions)
- ✅ Added 20% inverse hedge
- ✅ Maintain quality long positions

### Implementation Details

#### 1. **us_seasonality_full.py** (Universe Generation)
```python
# Lines 347-350
INVERSE_ETFS = ['SH', 'PSQ', 'DOG', 'RWM']
universe = list(set(universe + INVERSE_ETFS))
print(f"[INFO] Added {len(INVERSE_ETFS)} inverse ETFs to universe: {', '.join(INVERSE_ETFS)}")
```

#### 2. **ml_unified_pipeline.py** (Auto-Include in Bearish Regimes)
```python
# Lines 326-332
BEAR_MARKET_INVERSE_ETFS = ['SH', 'PSQ', 'DOG', 'RWM']

if regime in ['NEUTRAL_BEARISH', 'WEAK_BEARISH', 'BEAR_WEAK', 'BEAR_STRONG', 'CRISIS']:
    original_count = len(universe)
    universe = list(set(universe + BEAR_MARKET_INVERSE_ETFS))
    added_count = len(universe) - original_count
    if added_count > 0:
        print(f"[INFO] Added {added_count} inverse ETFs for {regime} regime")
```

#### 3. **auto_decider.py** (CRISIS & Bearish Logic)
```python
# Lines 425-458: CRISIS Mode
if regime == 'CRISIS':
    print(f"\n[CRISIS MODE] Exiting all long positions")
    
    # Separate inverse ETFs from regular stocks
    all_inverse_etfs = set(['SH', 'PSQ', 'DOG', 'RWM', 'SQQQ'])
    inverse_etfs_in_portfolio = current_tickers & all_inverse_etfs
    longs_in_portfolio = current_tickers - all_inverse_etfs
    
    # Sell all longs
    decisions['sell'] = list(longs_in_portfolio)
    for ticker in longs_in_portfolio:
        decisions['reason'][ticker] = 'CRISIS_EXIT_LONGS'
    
    # Buy inverse ETFs if strategy allows
    if strategy and not no_new_positions:
        inverse_to_buy, inverse_reasons = allocate_inverse_etfs(
            candidates_df, portfolio_state, regime_data, strategy
        )
        decisions['buy'] = inverse_to_buy
        decisions['reason'].update(inverse_reasons)
    
    # Hold existing inverse ETFs
    decisions['hold'] = list(inverse_etfs_in_portfolio)
    
    return decisions

# Lines 487-496: Bearish Mode
if strategy and regime in ['NEUTRAL_BEARISH', 'BEAR_WEAK', 'BEAR_STRONG']:
    inverse_to_add, inverse_reasons = allocate_inverse_etfs(
        candidates_df, portfolio_state, regime_data, strategy
    )
    # Add inverse ETFs to candidate pool
    if inverse_to_add:
        candidate_tickers = candidate_tickers.union(set(inverse_to_add))
        decisions['reason'].update(inverse_reasons)
```

#### 4. **regime_strategies.py** (Regime Configs)
```python
# Lines 15-102 (excerpt)
'CRISIS': {
    'allow_shorts': True,
    'short_allocation': 0.80,  # 80% to inverse ETFs
    'inverse_etfs': ['SH', 'PSQ'],
    'max_positions': 2,
    'position_size_factor': 0.90
},
'BEAR_WEAK': {
    'allow_shorts': True,
    'short_allocation': 0.40,  # 40% to inverse ETFs
    'inverse_etfs': ['SH', 'PSQ', 'DOG'],
    'max_positions': 4,
    'position_size_factor': 0.80
},
'NEUTRAL_BEARISH': {
    'allow_shorts': True,
    'short_allocation': 0.20,  # 20% to inverse ETFs
    'inverse_etfs': ['SH', 'PSQ'],
    'max_positions': 6,
    'position_size_factor': 0.70
}
```

### Testing Inverse ETF System

#### Test 1: Import & Configuration
```bash
python test_inverse_etfs.py
```

**Expected Output:**
```
✅ All imports successful
✅ Inverse ETFs available: ['SH', 'PSQ', 'DOG', 'RWM']
✅ NEUTRAL_BEARISH: shorts=True, allocation=20%, ETFs=['SH', 'PSQ']
✅ BEAR_WEAK: shorts=True, allocation=40%, ETFs=['SH', 'PSQ', 'DOG']
✅ CRISIS: shorts=True, allocation=80%, ETFs=['SH', 'PSQ']
✅ All systems ready for inverse ETF trading!
```

#### Test 2: CRISIS Scenario Simulation
```bash
python test_crisis_scenario.py
```

**Expected Output:**
```
🧪 Testing CRISIS scenario simulation

Portfolio:
  Cash: $50,000
  Positions: $30,000
  Total: $80,000

Regime: CRISIS
Short allocation target: 80%
Target inverse allocation: $64,000

[SHORT ALLOCATION]
  Regime: CRISIS
  Target allocation: 80.0% ($64,000)
  Selected inverse ETFs: SH, PSQ

Selected inverse ETFs: ['SH', 'PSQ']
Reasons: ['INVERSE_ETF_CRISIS_80%', 'INVERSE_ETF_CRISIS_80%']

✅ CRISIS mode would buy: SH, PSQ
```

#### Test 3: Download Inverse ETF Prices
```bash
python inverse_etf_downloader.py
```

**Expected Output:**
```
📥 Downloading inverse ETF price data...

Downloading SH... ✅ 502 days, latest: $36.90
Downloading PSQ... ✅ 502 days, latest: $30.50
Downloading DOG... ✅ 502 days, latest: $24.20
Downloading RWM... ✅ 502 days, latest: $16.90

✅ Inverse ETF data ready!
```

### Configuration Options

#### Add 3x Leveraged Inverse ETFs (Advanced)

Edit `regime_strategies.py`:
```python
'CRISIS': {
    'allow_shorts': True,
    'short_allocation': 0.60,  # Reduce % due to 3x leverage
    'inverse_etfs': ['SQQQ', 'SPXS'],  # 3x short ETFs
    'max_positions': 2
}
```

Edit `ml_unified_pipeline.py`:
```python
BEAR_MARKET_INVERSE_ETFS = ['SH', 'PSQ', 'DOG', 'RWM', 'SQQQ', 'SPXS']
```

**⚠️ Warning:** 3x leveraged ETFs have:
- Higher volatility decay
- Daily rebalancing losses
- NOT suitable for holding > 1-2 weeks

#### Customize Allocation Percentages

Edit `regime_strategies.py`:
```python
'NEUTRAL_BEARISH': {
    'short_allocation': 0.30,  # Change from 20% to 30%
}
```

### When Inverse ETFs Are Used

| Market Condition | Inverse ETF Action | Rationale |
|-----------------|-------------------|-----------|
| **VIX > 40** | CRISIS → 80% allocation | Extreme fear, market panic |
| **SPX < 200 MA** | BEAR_STRONG → 60% allocation | Confirmed downtrend |
| **Breadth < 30%** | BEAR_WEAK → 40% allocation | Weak market internals |
| **Negative momentum** | NEUTRAL_BEARISH → 20% allocation | Early warning signal |
| **Normal conditions** | 0% allocation | Long-only strategy |

### Limitations & Best Practices

#### ✅ Best Practices:
1. **Short-term hedging** (1-4 weeks maximum)
2. **CRISIS mode only** for aggressive allocations
3. **Monitor daily** for exit signals
4. **Combine with cash** (don't go 100% inverse)

#### ❌ Avoid:
1. **Long-term holding** (decay over time)
2. **100% inverse allocation** (leaves no liquidity)
3. **3x leverage** unless experienced
4. **Emotional decisions** (follow regime signals)

#### ⚠️ Risks:
- **Volatility decay** in sideways markets
- **Tracking errors** due to daily rebalancing
- **Whipsaw losses** if regime changes rapidly
- **Contango** (futures-based ETFs)

---

## 🧠 Regime-Based Position Limits

| Regime | Max Positions | Position Size | Inverse ETF % | Strategy Type | Trigger |
|--------|--------------|---------------|---------------|---------------|---------|
| **CRISIS** | 2 | 90% | **80%** | Exit longs, buy inverse ETFs | VIX > 40 OR Score < -0.20 |
| **BEAR_STRONG** | 2 | 80% | **60%** | Defensive + inverse hedging | Score: -0.20 to -0.10 |
| **BEAR_WEAK** | 4 | 80% | **40%** | Mean reversion + hedging | Score: -0.10 to 0 |
| **NEUTRAL_BEARISH** | 6 | 70% | **20%** | Cautious with small hedge | Score: 0 to -0.10 |
| **NEUTRAL_BULLISH** | 8 | 90% | 0% | Balanced long-only | Score: 0 to +0.10 |
| **BULL_WEAK** | 10 | 100% | 0% | Selective momentum | Score: +0.10 to +0.20 |
| **BULL_STRONG** | 12 | 120% | 0% | Aggressive momentum | Score > +0.20 |

**Current Regime (2025-11-06):** NEUTRAL_BULLISH (score: 0.158, confidence: 16.1%)

**Regime Detection Sources:**
- SPX trend (20/50/200 SMA crossovers)
- VIX level (fear gauge)
- Sector breadth (% sectors above 50-day MA)
- Market internals (advance/decline)
- **NEW:** Inverse ETF allocation rules

---

## 🎯 Stop Loss & Take Profit Calculation (ATR-Based)

**Method:** 14-period Average True Range (Wilder's method)

```python
# For BUY positions (new entries)
EntryPrice = Current Close Price
ATR = 14-day Average True Range
StopLoss = EntryPrice - (1.0 × ATR)      # 1 ATR risk
TakeProfit = EntryPrice + (3.0 × ATR)    # 3 ATR target

# Risk/Reward Ratio: 1:3

# For SELL positions (exits)
CurrentPrice = Latest Close
StopLoss = Original EntryPrice - (1.0 × ATR)
TakeProfit = Original EntryPrice + (3.0 × ATR)
```

**Example (AMD):**
```
EntryPrice: $250.05
ATR (14d): $6.12
StopLoss: $243.93  (-2.4% risk)
TakeProfit: $268.41  (+7.3% target)
```

**⭐ Inverse ETF Stop Loss:**
```
# SH (Inverse S&P 500)
EntryPrice: $36.90
ATR (14d): $0.85
StopLoss: $36.05  (tighter stop for inverse ETFs)
TakeProfit: $39.45  (profit if market drops)
```

**Implementation:** `auto_decider.py` → `enrich_with_stop_tp()` function

---

## 📊 Portfolio State Schema

```json
{
  "positions": {
    "AMD": {
      "entry_date": "2025-11-03",
      "entry_price": 250.05,
      "quantity": 4,
      "regime_at_entry": "NEUTRAL_BULLISH"
    },
    "SH": {
      "entry_date": "2025-11-06",
      "entry_price": 36.90,
      "quantity": 868,
      "regime_at_entry": "CRISIS",
      "is_inverse_etf": true
    }
  },
  "cash": 16000.0,
  "counters": {
    "day_entries": 2,
    "week_entries": 5,
    "week_start": "2025-11-04",
    "last_day": "2025-11-06"
  },
  "settings": {
    "max_positions": 8,
    "max_entries_day": 3,
    "max_entries_week": 10,
    "max_weight_pct": 20.0,
    "inverse_etfs_enabled": true
  },
  "last_updated": "2025-11-06"
}
```

**⚠️ CRITICAL:** This file is updated ONLY when `auto_decider.py` runs with `--commit 1`

---

## 💻 Command Line Usage

### **ML Pipeline (11:00 UTC)**
```bash
python ml_unified_pipeline.py \
  --universe_csv "seasonality_reports/aggregates/constituents_raw.csv" \
  --today "2025-11-06" \
  --gate_alpha 0.10 \
  --run_root "seasonality_reports/runs/2025-11-06_0000"
```

### **Auto Decider (15:55 UTC)**
```bash
python auto_decider.py \
  --project_root "." \
  --universe_csv "seasonality_reports/aggregates/constituents_raw.csv" \
  --run_root "seasonality_reports/runs/2025-11-06_0000" \
  --price_cache_dir "seasonality_reports/runs/2025-10-04_0903/price_cache" \
  --today "2025-11-06" \
  --max_positions 8 \
  --position_size 1000.0 \
  --commit 1
```

**Flags:**
- `--commit 0`: Dry-run (don't update portfolio_state.json)
- `--commit 1`: LIVE mode (update portfolio) ⚡
- `--no_new_positions`: Exit-only mode (sell all, buy nothing)

### **⭐ Test CRISIS Mode (Simulation)**
```bash
# Simulate CRISIS mode without committing
python auto_decider.py \
  --project_root "." \
  --universe_csv "seasonality_reports/aggregates/constituents_raw.csv" \
  --run_root "seasonality_reports/runs/2025-11-06_0000" \
  --price_cache_dir "seasonality_reports/runs/2025-10-04_0903/price_cache" \
  --today "2025-11-06" \
  --max_positions 2 \
  --position_size 1000.0 \
  --commit 0 \
  --force_regime CRISIS
```

### **Exit Watchlist (16:05 UTC)**
```bash
python make_exit_watchlist.py \
  --price_cache_dir "seasonality_reports/runs/2025-10-04_0903/price_cache" \
  --actions_dir "seasonality_reports/runs/2025-11-06_0000/actions/20251106" \
  --stop_mult 2.0
```

### **⭐ Download Inverse ETF Prices**
```bash
# One-time setup or daily refresh
python inverse_etf_downloader.py
```

---

## 📈 Current System Status (2025-11-06)

**Portfolio:**
- Positions: 8/8 (full)
- Tickers: AMD, LLY, GILD, BMY, NVDA, CRM, TMO, AAPL
- Cash: $85,000
- Total Equity: ~$93,000 (estimated)
- **Inverse ETFs:** None (NEUTRAL_BULLISH regime)

**Market Regime:** NEUTRAL_BULLISH (16.1% confidence)

**Inverse ETF Status:**
- System: ✅ Operational
- Price Data: ✅ Downloaded (SH, PSQ, DOG, RWM)
- Last CRISIS Mode: Never triggered (backtest only)

**Recent Actions:**
- 2025-11-06: No trades (portfolio = top-8)
- 2025-11-05: Bought BMY, TMO, AAPL
- 2025-11-04: Bought LLY, GILD
- 2025-11-03: Bought AMD, NVDA, CRM

---

## 🐻 Bear Market Strategy Summary

### Current Protection Layers

1. **⭐ Inverse ETFs (NEW)**
   - CRISIS: 80% allocation to SH, PSQ
   - BEAR_STRONG: 60% allocation to SH, PSQ, DOG
   - BEAR_WEAK: 40% allocation
   - NEUTRAL_BEARISH: 20% allocation

2. **Regime-Based Position Limits**
   - CRISIS: 0-2 positions (mostly inverse ETFs)
   - BEAR_STRONG: 2 positions (quality defensive)
   - BEAR_WEAK: 4 positions (mean reversion)

3. **Cash Preservation**
   - CRISIS: 20% cash minimum
   - BEAR_STRONG: 40% cash
   - BEAR_WEAK: 60% cash

4. **Stop Loss Monitoring**
   - 1.0 ATR automatic exit signal
   - Daily watchlist alerts

---

## 🐛 Troubleshooting

### **Issue 1: auto_decider.py fails at 15:55**
```bash
# Check ML pipeline completed:
ls seasonality_reports/runs/2025-11-06_0000/reports/top_long_candidates_GATED_2025-11-06.csv

# If missing, run manually:
python ml_unified_pipeline.py --today "2025-11-06" --run_root "seasonality_reports/runs/2025-11-06_0000"
```

### **Issue 2: No email received**
```bash
# Check .env file:
EMAIL_USER=panu.aalto1@gmail.com
EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# Test manually:
python send_trades_email.py
```

### **Issue 3: Portfolio state corrupted**
```bash
# Restore from Git:
git checkout HEAD -- seasonality_reports/portfolio_state.json
```

### **Issue 4: Price cache missing/outdated**
```bash
# Stock prices:
python build_prices_from_constituents.py \
  --const "seasonality_reports/aggregates/constituents_raw.csv" \
  --run_root "seasonality_reports/runs/2025-10-04_0903" \
  --overwrite

# Index prices:
python build_prices_from_indexes.py \
  --run_root "seasonality_reports" \
  --overwrite
```

### **⭐ Issue 5: Inverse ETFs not in candidates**
```bash
# Re-run universe generation:
python us_seasonality_full.py

# Re-download inverse ETF prices:
python inverse_etf_downloader.py

# Verify prices exist:
ls seasonality_reports\runs\2025-10-04_0903\price_cache\ | findstr "SH PSQ DOG RWM"
```

### **⭐ Issue 6: CRISIS mode not triggering**
```bash
# Check regime detection:
python regime_detector.py

# Force CRISIS mode (testing):
python auto_decider.py --commit 0 --force_regime CRISIS --today "2025-11-06" ...
```

---

## 📞 Contact & Support

**GitHub:** https://github.com/panuaalto1-afk/seasonality_project  
**Email:** panu.aalto1@gmail.com  
**Trading Hours:** 09:30-16:00 ET (14:30-21:00 UTC)  
**Critical Decision Time:** 10:55 ET (15:55 UTC) ⚡

---

## 📝 Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-06 | **v4.0** | **⭐ INVERSE ETF SYSTEM DEPLOYED**<br>- Added SH, PSQ, DOG, RWM support<br>- CRISIS mode: Exit longs, buy inverse ETFs<br>- Bearish modes: 20-60% inverse allocation<br>- Auto-add to universe<br>- Test suite created |
| 2025-11-06 | v3.0 | Complete documentation with folder structure, dataflow, options pipeline |
| 2025-11-06 | v2.1 | Added exit_watchlist monitoring, options integration notes |
| 2025-11-06 | v2.0 | Corrected workflow timing, price cache locations |
| 2025-11-06 | v1.3 | Added ATR-based Stop/TP calculation |
| 2025-11-05 | v1.2 | Regime-aware position sizing |
| 2025-11-04 | v1.1 | Email automation |
| 2025-11-03 | v1.0 | Initial auto_decider deployment |

---

## ✅ Daily Pre-Flight Checklist

**Before 15:55 UTC (10:55 ET):**
- [ ] ✅ 10:00 UTC: Stock prices updated
  - [ ] **⭐ Verify inverse ETFs:** Check SH, PSQ, DOG, RWM in price_cache
- [ ] ✅ 11:00 UTC: ML pipeline completed
  - [ ] **⭐ Check regime:** Verify current market regime
- [ ] ✅ 12:00 UTC: Index prices updated
- [ ] ✅ 15:55 UTC: **auto_decider.py runs**
  - [ ] **⭐ CRISIS check:** If regime = CRISIS, verify inverse ETFs in output

**After Market Open (14:30 UTC / 09:30 ET):**
- [ ] Review email: trade_candidates.csv (BUY orders)
  - [ ] **⭐ If inverse ETFs present:** Verify allocation % matches regime
- [ ] Review email: sell_candidates.csv (SELL orders)
- [ ] Check exit_watchlist.csv for stop-loss breaches
- [ ] Execute trades manually (or via broker API)
- [ ] **⭐ Monitor inverse ETF positions:** Check hedge performance

---

## 🎯 Key Reminders

1. **Inverse ETF Rules:**
   - **CRISIS:** Exit all longs, buy inverse ETFs (80%)
   - **BEAR_STRONG:** Reduce longs, add inverse ETFs (60%)
   - **BEAR_WEAK:** Balanced with inverse ETFs (40%)
   - **NEUTRAL_BEARISH:** Small hedge (20%)
   - **Bullish regimes:** No inverse ETFs

2. **Testing Before Live:**
   - Always test CRISIS mode with `--commit 0` first
   - Run `test_crisis_scenario.py` for validation
   - Verify inverse ETF prices are current

3. **Price Cache Locations:**
   - Stocks + Inverse ETFs: `runs/2025-10-04_0903/price_cache/`
   - Indexes: `seasonality_reports/price_cache/`

4. **Email Timing:**
   - Sent automatically after auto_decider completes
   - Expect by 16:00-16:05 UTC (11:00-11:05 ET)

---

## ⚖️ Disclaimer

This system is for educational and research purposes. Inverse ETFs carry significant risks:

**⚠️ RISKS:**
- **Volatility decay** in sideways markets
- **Daily rebalancing** causes tracking errors
- **Not suitable** for long-term holding
- **3x leverage** magnifies both gains AND losses
- **Market whipsaws** can cause rapid losses

**✅ BEST PRACTICES:**
- Test in simulation mode first
- Monitor daily - don't "set and forget"
- Use stop losses even on inverse ETFs
- Keep cash reserves
- Exit inverse positions when regime improves

**Always perform due diligence and risk management before trading.**

---

## 🏆 System Status

✅ Seasonality Analysis - Operational  
✅ ML Pipeline - Operational  
✅ Regime Detection - Operational  
✅ **⭐ Inverse ETF System - Operational (v4.0)**  
✅ Auto Decider - Operational  
✅ Testing Suite - Complete  
✅ Email Notifications - Operational  
✅ Stop Loss Monitoring - Operational  

**Last Updated:** 2025-11-06 18:59 UTC  
**Version:** 4.0 (Inverse ETF System Deployed)

---

**🎯 Happy Trading! Remember: The best trade is often no trade.** 🚀

**⭐ New Feature:** Inverse ETF system adds powerful downside protection. Test thoroughly before relying on it in live markets.