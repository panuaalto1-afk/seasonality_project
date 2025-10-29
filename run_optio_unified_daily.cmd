@echo off
setlocal
REM --- POLUT ---
set "PROJ=C:\Users\panua\seasonality_project"
set "PY=C:\Users\panua\AppData\Local\Python\bin\python.exe"
set "PRICE=%PROJ%\seasonality_reports\runs\2025-10-04_0903\price_cache"
set "UNIV=%PROJ%\seasonality_reports\Constituents_raw.csv"
set "LOG=%PROJ%\seasonality_reports\logs\optio_unified_daily.log"

REM --- VALMISTELU ---
mkdir "%PROJ%\seasonality_reports\logs" 2>nul
cd /d "%PROJ%"
chcp 65001 >nul
set PYTHONIOENCODING=UTF-8

echo === %date% %time% START optio_unified_daily >> "%LOG%"
echo whoami: >> "%LOG%"
whoami >> "%LOG%"
echo CD: %CD% >> "%LOG%"

"%PY%" "%PROJ%\optio_unified_daily.py" ^
  --project_root "%PROJ%" ^
  --price_cache_dir "%PRICE%" ^
  --universe_csv "%UNIV%" >> "%LOG%" 2>&1

echo === %date% %time% END >> "%LOG%"
endlocal
