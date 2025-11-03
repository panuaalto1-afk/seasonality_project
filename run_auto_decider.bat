@echo off
cd /d C:\Users\panua\seasonality_project

for /f %%i in ('powershell -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i

echo ========================================
echo AUTO DECIDER - %TODAY%
echo ========================================

set "RUNS_DIR=seasonality_reports\runs"
set "LATEST_RUN="

for /f "delims=" %%D in ('dir /b /ad /o-n "%RUNS_DIR%"') do (
    set "LATEST_RUN=%%D"
    goto :found_run
)

:found_run
if "%LATEST_RUN%"=="" (
    echo [ERROR] Ei loytynyt run-kansiota!
    exit /b 1
)

set "RUN_ROOT=%RUNS_DIR%\%LATEST_RUN%"
set "PRICE_CACHE=%RUN_ROOT%\price_cache"

echo [INFO] Run root:     %RUN_ROOT%
echo [INFO] Price cache:  %PRICE_CACHE%
echo [INFO] Today:        %TODAY%
echo ========================================

python auto_decider.py --project_root "." --universe_csv "seasonality_reports/constituents_raw.csv" --run_root "%RUN_ROOT%" --price_cache_dir "%PRICE_CACHE%" --today "%TODAY%" --commit 1

echo ========================================
echo [INFO] Auto decider valmis: %ERRORLEVEL%
echo ========================================

exit /b %ERRORLEVEL%