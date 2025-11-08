@echo off
REM ============================================
REM Update Regime Prices - Daily (10:50)
REM Paivittaa indeksi/ETF hinnat regime detectionia varten
REM Kriittiset tickerit: SPY, QQQ, IWM, ^SPX, ^VIX
REM ============================================

cd /d "C:\Users\panua\seasonality_project"

set "VENV=.venv\Scripts\python.exe"
set "LOG=logs\update_regime_prices_last.log"

if not exist "logs" mkdir "logs"

echo ============================================ > "%LOG%"
echo UPDATE REGIME PRICES - %date% %time% >> "%LOG%"
echo ============================================ >> "%LOG%"
echo. >> "%LOG%"

REM Paivita indeksit ja cross-asset proxies
echo [INFO] Updating regime detection prices... >> "%LOG%"
echo [INFO] Critical tickers: SPY, QQQ, IWM, ^SPX, ^VIX >> "%LOG%"
echo. >> "%LOG%"

"%VENV%" build_prices_from_indexes.py ^
    --reports_root "seasonality_reports" ^
    --overwrite >> "%LOG%" 2>&1

set RC=%ERRORLEVEL%
echo. >> "%LOG%"
echo Exit code: %RC% >> "%LOG%"
echo End: %date% %time% >> "%LOG%"

if %RC% EQU 0 (
    echo.
    echo ============================================
    echo SUCCESS - Regime Prices Updated
    echo ============================================
    echo Critical tickers: SPY, QQQ, IWM updated
    echo Location: seasonality_reports\price_cache\
) else (
    echo.
    echo ============================================
    echo FAILED - Exit Code: %RC%
    echo ============================================
    echo Check logs: %LOG%
)

exit /b %RC%
