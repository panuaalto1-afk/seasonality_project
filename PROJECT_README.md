# Seasonality Project (Windows)

**Juuri:** `C:\Users\OMISTAJA\seasonality_project\`  
**Raporttijuuri:** `seasonality_reports\`  *(pienellä, alaviiva)*

## Kansiot
- `seasonality_reports\price_cache\` — Päivähinnat CSV:nä: `<TICKER>.csv`
- `seasonality_reports\vintage\` — Vintage-seasonality ulostulot (per-tikkeri + kooste)
- `seasonality_reports\Seasonality_up_down_week\<YYYY-MM-DD_HHMM>\` — Viikkoraporttien ajokohtainen alikansio
- `seasonality_reports\intraday\YYYY\Kuukausi\` — Päivän Breakout / Combo / RSI / Combo EXTADJ -watchlistit
- `seasonality_reports\aggregates\` — Koosteet & kalenterit
- `seasonality_reports\Constituents_raw.csv` — Fundamenttisuodatuksen snapshot (tikkerilista)

## Oletusparametrit
- **Vintage lookback:** 10 vuotta  
- **Horisontit (K):** 10, 21, 63 pörssipäivää  
- **Komissiot (backtest):** 0.2% sisään + 0.2% ulos  
- **Salkku:** 100 000 USD, positio 33% pääomasta

## Skriptit (juuressa)
- `build_prices_from_constituents.py` — Lataa hinnat `Constituents_raw.csv`-listalle → `price_cache\`
- `build_prices_from_indexes.py` — Lataa hinnat SPX/DOW/NASDAQ/R2000 → `price_cache\` (tukee `--from-locals`, `--skip-mcap`)
- `build_vintage_from_price_cache.py` — Laskee **vintage**t kaikille `price_cache\*.csv` → `vintage\`

## Nimeämissäännöt
- Pidä **ainoana** juurena `seasonality_reports\` (pienellä).  
- Viikkoraportit **aina** aikaleimattuun alikansioon: `Seasonality_up_down_week\YYYY-MM-DD_HHMM\`.

## Pika-ajot (CMD)
```bat
cd C:\Users\OMISTAJA\seasonality_project

:: riippuvuudet
py -m pip install --upgrade yfinance pandas numpy lxml

:: hinnat funda-listalle (20v)
py build_prices_from_constituents.py ^
  --const "seasonality_reports\Constituents_raw.csv" ^
  --outdir "seasonality_reports\price_cache" ^
  --years 20

:: vintage kaikille cache-CSV:ille
py build_vintage_from_price_cache.py ^
  --price-dir "seasonality_reports\price_cache" ^
  --out-dir   "seasonality_reports\vintage" ^
  --lookback 10 ^
  --ks 10 21 63
