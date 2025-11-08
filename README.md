YHTEENVETO: Seasonality Trading System

Päivitetty: 2025-11-07 22:41 (Suomen aika)
Tila: ✅ Kaikki 10 ajastettua tehtävää toimii
Regime System: ✅ Palautettu GitHubista (regime_detector.py, regime_strategies.py)
📋 1. AJASTETUT TEHTÄVÄT (10 kpl)
Aika	Task Scheduler Nimi	Skripti	Tulos	Seuraava
02:00	Seasonality – US Seasonality Full	us_seasonality_full.py	❌ Ei ajettu (seuraava: 20.12.2025)	20.12.2025 02:56
10:00	Seasonality Prices klo 1000	build_prices_from_constituents.py	✅ 7.11.2025 10:00	08.11.2025 10:00
10:30	Aggregate seasonality picker klo 1030	run_aggregate_picker_daily.cmd	✅ 7.11.2025 10:30	10.11.2025 10:30
10:50	Trading_UpdateRegimePrices	update_regime_prices.bat	✅ 7.11.2025 21:59	10.11.2025 10:50
11:00	ML Unified pipeline klo 1100	ml_unified_pipeline.py	✅ 7.11.2025 11:00	10.11.2025 11:00
12:00	Seasonality – Build index prices	build_prices_from_indexes.py	✅ 7.11.2025 12:00	10.11.2025 12:00
15:00	Optio seasonality signal klo 1500	run_optio_signals_daily.cmd	✅ 7.11.2025 15:00	10.11.2025 15:00
15:30	Optio seasonality price enricher 1530	run_optio_enricher_daily.cmd	✅ 7.11.2025 15:30	10.11.2025 15:30
15:45	Optio unified daily 1545	run_optio_unified_daily.cmd	✅ 7.11.2025 15:45	10.11.2025 15:45
15:55	seasonality_auto_decider	run_auto_decider.cmd	✅ 7.11.2025 21:26	10.11.2025 15:55
📂 2. HAKEMISTOPUU
Code

C:\Users\panua\seasonality_project\
│
├── 🐍 PYTHON-SKRIPTIT (Juuressa)
│   ├── us_seasonality_full.py                    [02:00 Kuukausittain 20. päivä]
│   ├── build_prices_from_constituents.py         [10:00 Päivittäin]
│   ├── aggregate_seasonality_picker.py           [10:30 Päivittäin]
│   ├── build_prices_from_indexes.py              [10:50 & 12:00 Päivittäin]
│   ├── ml_unified_pipeline.py                    [11:00 Päivittäin]
│   ├── optio_seasonality_signal.py               [15:00 Päivittäin]
│   ├── optio_seasonality_price_enricher.py       [15:30 Päivittäin]
│   ├── optio_unified_daily.py                    [15:45 Päivittäin]
│   ├── auto_decider.py                           [15:55 Päivittäin]
│   ├── regime_detector.py                        ⭐ Palautettu GitHubista 7.11.2025
│   └── regime_strategies.py                      ⭐ Palautettu GitHubista 7.11.2025
│
├── 📋 BATCH-TIEDOSTOT (Task Scheduler wrapperit)
│   ├── update_regime_prices.bat                  ⭐ Luotu 7.11.2025
│   ├── run_auto_decider.cmd
│   ├── run_aggregate_picker_daily.cmd
│   ├── run_optio_signals_daily.cmd
│   ├── run_optio_enricher_daily.cmd
│   └── run_optio_unified_daily.cmd
│
├── ⚙️  KONFIGURAATIOT
│   ├── .env                                      [Email: EMAIL_USER, EMAIL_APP_PASSWORD]
│   └── seasonality_reports\
│       ├── portfolio_state.json                  ⚠️ KRIITTINEN - Nykyiset positiot
│       └── Constituents_raw.csv                  [Universe: ~500 osaketta]
│
├── 💾 PRICE CACHE (Kaksi erillistä)
│   │
│   ├── seasonality_reports\runs\2025-10-04_0903\price_cache\
│   │   └── 517 tiedostoa                         [Osakkeet: AMD, AAPL... + Indeksit: SPY, QQQ, IWM...]
│   │                                            [Käyttö: auto_decider.py - Hinnat, ATR, SL/TP]
│   │
│   └── seasonality_reports\price_cache\
│       └── 16 tiedostoa                          [Indeksit: ^SPX, ^VIX, SPY, QQQ, IWM, TLT, GLD...]
│                                                [Käyttö: regime_detector.py - Regime tunnistus]
│
├── 📊 PÄIVITTÄISET TULOKSET
│   └── seasonality_reports\runs\2025-11-07_0000\
│       │
│       ├── reports\
│       │   ├── features_2025-11-07.csv
│       │   ├── labels_2025-11-07.csv
│       │   ├── top_long_candidates_RAW_2025-11-07.csv
│       │   ├── top_long_candidates_GATED_2025-11-07.csv    ⚡ auto_decider INPUT
│       │   ├── top_short_candidates_RAW_2025-11-07.csv
│       │   └── top_short_candidates_GATED_2025-11-07.csv
│       │
│       └── actions\20251107\
│           ├── action_plan.txt                   [Yhteenveto kauppapäätöksistä]
│           ├── trade_candidates.csv              [BUY orders: Entry, SL, TP]
│           ├── sell_candidates.csv               [SELL orders: P/L%]
│           ├── portfolio_after_sim.csv           [Portfolio kauppojen jälkeen]
│           └── exit_watchlist.csv                [Stop-loss seuranta]
│
├── 📈 OPTIO-TULOKSET
│   └── seasonality_reports\aggregates\
│       │
│       ├── optio_signals\2025-11-07\
│       │   ├── top_breakout_long.csv             [Long optio-signaalit]
│       │   ├── top_breakout_short.csv            [Short optio-signaalit]
│       │   ├── exit_alerts.csv                   [Exit signaalit]
│       │   └── *.html                            [Raportti HTML]
│       │
│       └── optio_signals_enriched\2025-11-07\
│           ├── optio_price_enriched_all.csv      [Hinnoitetut optiot]
│           ├── optio_price_enriched_long.csv
│           ├── optio_price_enriched_short.csv
│           └── regime_sector_momentum.csv
│
└── 📜 LOKIT
    ├── logs\
    │   ├── update_regime_prices_last.log         [10:50 ajo]
    │   ├── auto_decider_last.log                 [15:55 ajo]
    │   ├── auto_decider_debug.log
    │   └── email_test.log
    │
    └── seasonality_reports\logs\
        ├── auto_decider.log
        └── optio_unified_daily.log

🔄 3. DATA FLOW (Päivittäinen Prosessi)
Code

[02:00 Kuukauden 20. päivä]
us_seasonality_full.py
    └─> Rakentaa seasonality-tietokannan (20v historia)

[10:00]
build_prices_from_constituents.py
    └─> Lataa 517 osakkeen hinnat (20v, OVERWRITE)
        └─> runs/2025-10-04_0903/price_cache/*.csv

[10:30]
aggregate_seasonality_picker.py
    └─> Aggregoi päivän seasonality-signaalit

[10:50] ⭐ UUSI
update_regime_prices.bat → build_prices_from_indexes.py
    └─> Lataa 16 indeksin hinnat (SPY, QQQ, IWM, ^SPX, ^VIX...)
        └─> seasonality_reports/price_cache/*.csv

[11:00]
ml_unified_pipeline.py
    ├─> LUKEE: runs/2025-10-04_0903/price_cache/*.csv
    ├─> KUTSUU: regime_detector.py (jos löytyy)
    │   └─> LUKEE: seasonality_reports/price_cache/*.csv
    ├─> Laskee: ML features (momentum, volatility, ATR...)
    └─> TUOTTAA: top_long_candidates_GATED_2025-11-07.csv

[12:00]
build_prices_from_indexes.py (toinen ajo)
    └─> Lataa indeksit runs/2025-10-04_0903/price_cache/ (sama kuin 10:50)

[15:00]
optio_seasonality_signal.py
    └─> TUOTTAA: top_breakout_long/short.csv

[15:30]
optio_seasonality_price_enricher.py
    └─> TUOTTAA: optio_price_enriched_*.csv (hinnoitetut optiot)

[15:45]
optio_unified_daily.py
    └─> Yhdistää optio-signaalit

[15:55] ⚡⚡⚡ KRIITTISIN
run_auto_decider.cmd → auto_decider.py
    ├─> LUKEE: top_long_candidates_GATED_2025-11-07.csv
    ├─> LUKEE: portfolio_state.json
    ├─> LUKEE: runs/2025-10-04_0903/price_cache/*.csv (osakkeiden hinnat)
    │
    ├─> KUTSUU: regime_detector.py
    │   └─> LUKEE: seasonality_reports/price_cache/*.csv (indeksit)
    │   └─> PALAUTTAA: regime (BULL/NEUTRAL/BEAR/CRISIS)
    │
    ├─> KUTSUU: regime_strategies.py
    │   └─> PALAUTTAA: max_positions, position_size_factor
    │
    ├─> PÄÄTTÄÄ: BUY / SELL / HOLD
    │
    ├─> TUOTTAA: actions/20251107/
    │   ├── action_plan.txt
    │   ├── trade_candidates.csv (BUY)
    │   ├── sell_candidates.csv (SELL)
    │   └── portfolio_after_sim.csv
    │
    ├─> PÄIVITTÄÄ: portfolio_state.json (jos --commit 1)
    │
    └─> LÄHETTÄÄ: Email 📧 panu.aalto1@gmail.com
        └─> Liitteet: action_plan.txt, trade_candidates.csv, sell_candidates.csv

🧠 4. REGIME DETECTION SYSTEM
regime_detector.py (Palautettu 7.11.2025)

Tarkoitus: Tunnistaa markkinaregime 5 komponentin perusteella

Komponentit:

    Equity Momentum (SPY, QQQ, IWM) - 35% paino
    Volatility (SPY realized vol) - 20% paino
    Credit Spreads (HYG vs LQD) - 20% paino
    Safe Haven Flows (GLD, TLT) - 15% paino
    Market Breadth (SPY vs IWM korrelaatio) - 10% paino

Input: seasonality_reports/price_cache/*.csv (16 indeksiä)

Output:
Python

{
    'date': '2025-11-07',
    'regime': 'NEUTRAL_BULLISH',
    'composite_score': 0.158,
    'confidence': 0.72,
    'components': {
        'equity': {'signal': 0.45, ...},
        'volatility': {'signal': 0.32, ...},
        ...
    }
}

Regimes:

    BULL_STRONG (score ≥ 0.50)
    BULL_WEAK (score ≥ 0.25)
    NEUTRAL_BULLISH (score ≥ 0.0)
    NEUTRAL_BEARISH (score ≥ -0.25)
    BEAR_WEAK (score ≥ -0.50)
    BEAR_STRONG (score ≥ -0.75)
    CRISIS (score < -0.75)

Tallennus: seasonality_reports/regime_history.csv
regime_strategies.py (Palautettu 7.11.2025)

Tarkoitus: Määrittää kaupankäyntiparametrit regimen mukaan

Strategiat regimeittäin:
Regime	Strategy Type	Max Positions	Position Size	Entry Style	Min ML Score
BULL_STRONG	Momentum	12	130%	Aggressive	0.70
BULL_WEAK	Momentum	10	100%	Selective	0.75
NEUTRAL_BULLISH	Balanced	8	90%	Selective	0.75
NEUTRAL_BEARISH	Defensive Quality	6	70%	Conservative	0.80
BEAR_WEAK	Mean Reversion	4	50%	Very Conservative	0.85
BEAR_STRONG	Defensive	2	30%	Extreme Conservative	0.90
CRISIS	Capital Preservation	0	0%	No Entries	1.0

Signal Weights (esim. BULL_STRONG):

    Momentum: 70%
    Quality: 20%
    Value: 10%

Stop/TP Multipliers:

    BULL_STRONG: SL 1.5x ATR, TP 2.0x ATR
    NEUTRAL_BULLISH: SL 1.0x ATR, TP 1.2x ATR
    BEAR_WEAK: SL 0.8x ATR, TP 0.8x ATR

🔧 5. TÄMÄN PÄIVÄN KORJAUKSET (7.11.2025)
✅ Korjaus 1: Email Ei Lähtenyt Task Schedulerista

Ongelma: python-dotenv puuttui .venv:stä
Ratkaisu: pip install python-dotenv
Tila: ✅ Toimii
✅ Korjaus 2: update_regime_prices.bat Puuttui

Ongelma: Task Trading_UpdateRegimePrices viittasi puuttuvaan tiedostoon
Ratkaisu: Luotiin update_regime_prices.bat
Tila: ✅ Toimii
✅ Korjaus 3: SPY, QQQ, IWM Eivät Päivittyneet

Ongelma: build_prices_from_indexes.py ei sisältänyt näitä tickereitä
Ratkaisu: Lisättiin CROSS_ASSET listaan: SPY, QQQ, IWM, ^SPX, ^VIX
Tila: ✅ Toimii
✅ Korjaus 4: regime_detector.py ja regime_strategies.py Puuttuivat

Ongelma: Tiedostot puuttuivat projektin juuresta
Ratkaisu: Ladattiin GitHubista
Tila: ✅ Palautettu (tarkista että toimivat)
📧 6. EMAIL-ILMOITUKSET

Lähettäjä: panu.aalto1@gmail.com
Vastaanottaja: panu.aalto1@gmail.com
Liitteet:

    action_plan.txt (yhteenveto)
    trade_candidates.csv (BUY orders)
    sell_candidates.csv (SELL orders)
    portfolio_after_sim.csv (portfolio kauppojen jälkeen)

Konfiguraatio: .env tiedostossa:
Code

EMAIL_USER=panu.aalto1@gmail.com
EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

ML Unified Pipeline 

Täysin uudistettu ML-pohjainen signaaligeneraattori, joka yhdistää momentum-analyysin, kausiluonteisuuden, markkinaregiimit ja ATR-pohjaiset trading-tasot.
🔧 Keskeiset Parannukset:
1. Regime Detection (7-tilainen markkinaympäristö)

    Itsenäinen RegimeCalculator (kopio regime_detector.py logiikasta)
    5 komponenttia: Equity, Volatility, Credit Spread, Safe Haven, Market Breadth
    7 regimeä:
        BULL_STRONG, BULL_WEAK
        NEUTRAL_BULLISH, NEUTRAL_BEARISH
        BEAR_WEAK, BEAR_STRONG
        CRISIS
    Data: Macro ETF hinnat (SPY, QQQ, IWM, GLD, TLT, HYG, LQD, VIX)

2. Multi-Window Seasonality Analysis

    Week-of-Year: Keskimääräinen viikkotason tuotto (10v historia)
    Day-of-Year: 20 päivän forward return (±3 päivän window)
    Month-of-Year: Kuukausitason kausiluonteisuus
    Quarter-of-Year: Kvartaalitason trendit
    Segmentit: Bullish/Bearish jaksojen tunnistus
        Käyttää {TICKER}_segments_up.csv ja _segments_down.csv
        Sisältää: segment strength, days into segment

3. Trading Levels Calculator

    Entry Price: T-1 close (edellisen päivän päätöskurssi)
    ATR-14: 14 päivän Average True Range
    Stop Loss: Entry - (ATR × regime_multiplier)
        Regime-kohtaiset kertoimet (0.8-2.0)
    Take Profit: Entry + (ATR × regime_multiplier)
        Regime-kohtaiset kertoimet (0.5-2.5)

4. ML Scoring (Placeholder for Future Enhancement)

    Nykyinen: Momentum + Seasonality blend
        50% momentum (mom5 + mom20)
        50% seasonality (week_avg + 20d_avg)
    Tulevaisuus: LightGBM/XGBoost regressio
        Target: 20 päivän forward return
        Features: Momentum + Seasonality + Regime (30+ features)

📊 Output Format:
Enhanced Features (24 saraketta):
Code

ticker, asof_date,
mom5, mom20, mom60, vol20,                          # Momentum
season_week_avg, season_week_hit_rate,              # Seasonality (viikko)
season_20d_avg, season_20d_hit_rate,                # Seasonality (20d)
season_month_avg, season_quarter_avg,               # Seasonality (kk/kvartaali)
in_bullish_segment, in_bearish_segment,             # Segmentit
days_into_segment, segment_strength,                # Segment info
entry_price, stop_loss, take_profit, atr_14,        # Trading levels
sl_distance_pct, tp_distance_pct,                   # SL/TP etäisyys %
regime, regime_score,                               # Regime
ml_expected_return, score_long, score_short         # ML/Ranking

Tiedostot:
Code

seasonality_reports/runs/{YYYY-MM-DD_HHMM}/reports/
├── features_{YYYY-MM-DD}.csv              # Kaikki featuret (516 riviä)
├── top_long_candidates_RAW_{date}.csv     # Top 200 (ei filtteröity)
├── top_long_candidates_GATED_{date}.csv   # Filtteröity (gate_alpha)
├── top_short_candidates_RAW_{date}.csv
├── top_short_candidates_GATED_{date}.csv
└── summary_{date}.txt                      # Yhteenveto + regime info

🔗 Integraatio auto_decider.py:hyn:
Python

# auto_decider.py lukee:
gated_csv = "top_long_candidates_GATED_{date}.csv"

# Käyttää sarakkeita:
- ticker              # Osakkeen tunniste
- score_long          # Ranking score (0-1)
- entry_price         # Entry hinta
- stop_loss           # Stop loss taso
- take_profit         # Take profit taso
- (+ muut optionaaliset)

# Soveltaa regime_strategies.py:
- Position sizing (regime-kohtainen)
- Max positions (8 default)
- Risk management

⚙️ CLI Parametrit:
bash

python ml_unified_pipeline.py \
    --today "YYYY-MM-DD" \
    --universe_csv "seasonality_reports/constituents_raw.csv" \
    --gate_alpha 0.10 \
    --train_years 10 \
    --run_root "seasonality_reports/runs/{YYYY-MM-DD_HHMM}"

📅 Päivittäinen Workflow:
Code

10:00 → update_price_cache_spy.py
        Päivittää osake- ja ETF-hinnat

11:00 → ml_unified_pipeline.py (ENHANCED)
        ├─ Regime detection
        ├─ Seasonality calculation
        ├─ Trading levels
        └─ Tuottaa: top_long_candidates_GATED.csv

15:55 → auto_decider.py
        ├─ Lukee: GATED.csv
        ├─ Soveltaa: regime_strategies.py
        └─ Tekee: Kaupat

🔍 Tekninen Toteutus:

Moduulit:

    RegimeCalculator (520 riviä)
        Itsenäinen regime detection
        5 komponenttia → composite score → 7 regimeä

    SeasonalityCalculator (200 riviä)
        Walk-forward safe (ei future leak)
        Multi-window approach (viikko/päivä/kk/kvartaali)
        Segment detection

    TradingLevelsCalculator (150 riviä)
        ATR calculation (fallback: close-to-close volatility)
        Regime-pohjaiset SL/TP multipliers

    ML Model (Placeholder) (100 riviä)
        Nykyinen: Momentum + Seasonality blend
        Tulevaisuus: LightGBM regression

Yhteensä: ~1200 riviä Python-koodia
