@echo off
setlocal EnableExtensions
set "ROOT=C:\Users\panua\seasonality_project"
set "PY=C:\Users\panua\AppData\Local\Python\bin\python.exe"
set "SCRIPT=%ROOT%\optio_seasonality_price_enricher.py"
set "PRICE_DIR=%ROOT%\seasonality_reports\runs\2025-10-04_0903\price_cache"
set "LOG=%ROOT%\logs\optio_enricher_last.log"

"%PY%" "%SCRIPT%" --root "%ROOT%" --optio-date latest --price-dir "%PRICE_DIR%" --refresh-online false --save-cache false --force-sides true --top-n 60 --round 3 > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

rem Jos ainoa “ongelma” on tyhjä long/short-maski, palautetaan onnistuminen.
findstr /C:"maski tyhjä" "%LOG%" >nul && set "RC=0"

rem Oikea virhe vain jos logissa on [ERROR].
findstr /C:"[ERROR]" "%LOG%" >nul && set "RC=2"

exit /b %RC%

