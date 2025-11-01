@echo off
REM Exit Watchlist - Päivittää exit-tasot
REM Ajetaan päivittäin klo 16:05

cd /d C:\Users\panua\seasonality_project
call .venv\Scripts\activate.bat

echo ============================================================
echo EXIT WATCHLIST - %date% %time%
echo ============================================================

python make_exit_watchlist.py

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Exit watchlist completed successfully
) else (
    echo [ERROR] Exit watchlist failed with error code %ERRORLEVEL%
)

echo ============================================================
echo Completed at %time%
echo ============================================================

REM Tallenna loki
echo [%date% %time%] Exit watchlist completed >> logs\exit_watchlist.log

pause