@echo off
setlocal ENABLEDELAYEDEXPANSION

REM --- POLUT ---
set PROJECT=C:\Users\OMISTAJA\seasonality_project
set PY=C:\Users\OMISTAJA\AppData\Local\Programs\Python\Python313\python.exe
REM *KÄYTÄ TÄTÄ VAKIOTA PRICE-CACHEA*
set SOURCE_CACHE=%PROJECT%\seasonality_reports\runs\2025-10-04_0903\price_cache

REM --- LOKI ---
set LOGDIR=%PROJECT%\seasonality_reports\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Ajastimessa %DATE% ja %TIME% vaihtelevat formaatteja – normalisoidaan:
set TS=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%
set TS=%TS: =0%
set TS=%TS::=%

cd /d "%PROJECT%"

echo [%%date%% %%time%%] Starting daily_one_click with source_cache="%SOURCE_CACHE%" >> "%LOGDIR%\daily_one_click_%TS%.log" 2>&1

"%PY%" daily_one_click.py ^
  --source_cache "%SOURCE_CACHE%" ^
  --no_download 1 ^
  --topoff 0 >> "%LOGDIR%\daily_one_click_%TS%.log" 2>&1

set EXITCODE=%ERRORLEVEL%
echo [%%date%% %%time%%] Finished with exitcode %EXITCODE% >> "%LOGDIR%\daily_one_click_%TS%.log" 2>&1
exit /b %EXITCODE%
