@'
@echo off
REM ============================================
REM Exit Watchlist - Daily Execution (FIXED v2)
REM ============================================

set "ROOT=C:\Users\panua\seasonality_project"
set "VENV=%ROOT%\.venv\Scripts\python.exe"
set "LOG=%ROOT%\logs\exit_watchlist_last.log"

REM Create logs directory
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

REM Get today's date using PowerShell (YYYY-MM-DD)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i

echo ============================================ > "%LOG%"
echo EXIT WATCHLIST - %TODAY% >> "%LOG%"
echo ============================================ >> "%LOG%"
echo Start: %date% %time% >> "%LOG%"
echo. >> "%LOG%"

REM Static price cache (20y history)
set "PRICE_CACHE=%ROOT%\seasonality_reports\runs\2025-10-04_0903\price_cache"

echo Using price cache: %PRICE_CACHE% >> "%LOG%"
echo Using portfolio: seasonality_reports\portfolio_state.json >> "%LOG%"
echo Output dir: seasonality_reports\exit_watchlist >> "%LOG%"
echo Date: %TODAY% >> "%LOG%"
echo. >> "%LOG%"

REM Check if make_exit_watchlist.py uses --output_dir or --actions_dir
REM First, try to run with --help to see parameters
"%VENV%" "%ROOT%\make_exit_watchlist.py" --help > "%ROOT%\logs\exit_help.txt" 2>&1

REM Run exit watchlist
"%VENV%" "%ROOT%\make_exit_watchlist.py" ^
    --portfolio_json "seasonality_reports\portfolio_state.json" ^
    --price_cache_dir "%PRICE_CACHE%" ^
    --output_dir "seasonality_reports\exit_watchlist" ^
    --date "%TODAY%" ^
    --stop_mult 2.0 ^
    --take_mult 3.0 >> "%LOG%" 2>&1

set RC=%ERRORLEVEL%

REM If failed with --output_dir, try --actions_dir
if %RC% NEQ 0 (
    echo First attempt failed, trying --actions_dir... >> "%LOG%"
    "%VENV%" "%ROOT%\make_exit_watchlist.py" ^
        --price_cache_dir "%PRICE_CACHE%" ^
        --actions_dir "seasonality_reports\exit_watchlist" ^
        --portfolio_csv "seasonality_reports\portfolio_state.json" ^
        --stop_mult 2.0 >> "%LOG%" 2>&1
    set RC=%ERRORLEVEL%
)

echo. >> "%LOG%"
echo Exit code: %RC% >> "%LOG%"
echo End: %date% %time% >> "%LOG%"

if %RC% EQU 0 (
    echo ✅ EXIT WATCHLIST SUCCESS
) else (
    echo ❌ EXIT WATCHLIST FAILED - Check %LOG%
)

type "%LOG%"
exit /b %RC%
'@ | Out-File -FilePath "run_make_exit_watchlist_daily.cmd" -Encoding ASCII

Write-Host "✅ run_make_exit_watchlist_daily.cmd luotu (PowerShell date)" -ForegroundColor Green