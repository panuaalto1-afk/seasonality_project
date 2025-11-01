@echo off
REM Auto Decider - Kauppapäätösten automaatio
REM Ajetaan päivittäin klo 15:55

cd /d C:\Users\panua\seasonality_project
call .venv\Scripts\activate.bat

echo ============================================================
echo AUTO DECIDER - %date% %time%
echo ============================================================

python auto_decider.py

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Auto decider completed successfully
) else (
    echo [ERROR] Auto decider failed with error code %ERRORLEVEL%
)

echo ============================================================
echo Completed at %time%
echo ============================================================

REM Tallenna loki
echo [%date% %time%] Auto decider completed >> logs\auto_decider.log

pause