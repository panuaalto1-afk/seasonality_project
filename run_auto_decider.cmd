Write-Host "`n🔧 LISÄTÄÄN DEBUG LOGGING run_auto_decider.cmd:ään..." -ForegroundColor Cyan

@'
@echo off
REM ============================================
REM Auto Decider - Daily Execution (DEBUG v2)
REM ============================================

REM PAKOTA WORKING DIRECTORY
cd /d "C:\Users\panua\seasonality_project"

set "ROOT=C:\Users\panua\seasonality_project"
set "VENV=%ROOT%\.venv\Scripts\python.exe"
set "LOG=%ROOT%\logs\auto_decider_last.log"
set "DEBUG_LOG=%ROOT%\logs\auto_decider_debug.log"

if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

REM Get today's date
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i

REM ===== DEBUG LOGGING =====
echo ============================================ > "%DEBUG_LOG%"
echo DEBUG LOG - %TODAY% %TIME% >> "%DEBUG_LOG%"
echo ============================================ >> "%DEBUG_LOG%"
echo. >> "%DEBUG_LOG%"
echo ENVIRONMENT: >> "%DEBUG_LOG%"
echo   Current Directory: %CD% >> "%DEBUG_LOG%"
echo   Username: %USERNAME% >> "%DEBUG_LOG%"
echo   UserProfile: %USERPROFILE% >> "%DEBUG_LOG%"
echo   ComputerName: %COMPUTERNAME% >> "%DEBUG_LOG%"
echo. >> "%DEBUG_LOG%"
echo FILE CHECKS: >> "%DEBUG_LOG%"
echo   .env exists: >> "%DEBUG_LOG%"
if exist "%ROOT%\.env" (
    echo     YES >> "%DEBUG_LOG%"
    echo     Location: %ROOT%\.env >> "%DEBUG_LOG%"
) else (
    echo     NO - THIS IS THE PROBLEM! >> "%DEBUG_LOG%"
)
echo. >> "%DEBUG_LOG%"
echo   Python exists: >> "%DEBUG_LOG%"
if exist "%VENV%" (
    echo     YES >> "%DEBUG_LOG%"
) else (
    echo     NO - THIS IS THE PROBLEM! >> "%DEBUG_LOG%"
)
echo. >> "%DEBUG_LOG%"

REM Test .env loading with Python
echo TESTING .ENV LOADING: >> "%DEBUG_LOG%"
"%VENV%" -c "import os; from pathlib import Path; from dotenv import load_dotenv; load_dotenv('.env'); print('EMAIL_USER:', os.getenv('EMAIL_USER')); print('EMAIL_APP_PASSWORD:', 'SET' if os.getenv('EMAIL_APP_PASSWORD') else 'NOT SET')" >> "%DEBUG_LOG%" 2>&1
echo. >> "%DEBUG_LOG%"

REM ===== NORMAL LOGGING =====
echo ============================================ > "%LOG%"
echo AUTO DECIDER - %TODAY% >> "%LOG%"
echo ============================================ >> "%LOG%"
echo Start: %date% %time% >> "%LOG%"
echo Current Directory: %CD% >> "%LOG%"
echo. >> "%LOG%"

set "RUN_ROOT=%ROOT%\seasonality_reports\runs\%TODAY%_0000"
set "PRICE_CACHE=%ROOT%\seasonality_reports\runs\2025-10-04_0903\price_cache"

echo Using run root: %RUN_ROOT% >> "%LOG%"
echo Using price cache: %PRICE_CACHE% >> "%LOG%"
echo Date: %TODAY% >> "%LOG%"
echo. >> "%LOG%"

REM Check signals
set "SIGNALS_FILE=%RUN_ROOT%\reports\top_long_candidates_GATED_%TODAY%.csv"
if not exist "%SIGNALS_FILE%" (
    echo [ERROR] Signals not found: %SIGNALS_FILE% >> "%LOG%"
    echo [ERROR] Signals not found: %SIGNALS_FILE% >> "%DEBUG_LOG%"
    exit /b 1
)

echo [OK] Signals found >> "%LOG%"
echo [OK] Signals found >> "%DEBUG_LOG%"
echo. >> "%LOG%"

REM Run auto_decider
cd /d "%ROOT%"
"%VENV%" "%ROOT%\auto_decider.py" ^
    --project_root "%ROOT%" ^
    --universe_csv "seasonality_reports\Constituents_raw.csv" ^
    --run_root "%RUN_ROOT%" ^
    --price_cache_dir "%PRICE_CACHE%" ^
    --today "%TODAY%" ^
    --commit 1 ^
    --max_positions 8 ^
    --position_size 1000.0 >> "%LOG%" 2>&1

set RC=%ERRORLEVEL%
echo. >> "%LOG%"
echo Exit code: %RC% >> "%LOG%"
echo End: %date% %time% >> "%LOG%"
echo. >> "%DEBUG_LOG%"
echo EXIT CODE: %RC% >> "%DEBUG_LOG%"

exit /b %RC%
'@ | Out-File -FilePath "run_auto_decider.cmd" -Encoding ASCII

Write-Host "✅ DEBUG logging lisätty!" -ForegroundColor Green