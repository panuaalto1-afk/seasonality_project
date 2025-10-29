@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=C:\Users\panua\seasonality_project"
set "PY=C:\Users\panua\AppData\Local\Python\bin\python.exe"
set "RUN_ROOT=%ROOT%\seasonality_reports\runs\2025-10-04_0903"
set "PRICE_DIR=%RUN_ROOT%\price_cache"
set "LOG=%ROOT%\logs\exit_watchlist_last.log"

mkdir "%ROOT%\logs" 2>nul
echo [BOOT] ROOT=%ROOT% > "%LOG%"
echo [BOOT] PY=%PY% >> "%LOG%"
echo [BOOT] RUN_ROOT=%RUN_ROOT% >> "%LOG%"
echo [BOOT] PRICE_DIR=%PRICE_DIR% >> "%LOG%"

rem Uusin actions-kansio
set "ACTIONS_DIR="
for /f "delims=" %%D in ('dir /b /ad "%RUN_ROOT%\actions" ^| findstr /R "^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]$" ^| sort') do set "ACTIONS_DIR=%RUN_ROOT%\actions\%%D"
if not defined ACTIONS_DIR (
  for /f %%A in ('"%PY%" -c "import datetime as d;print(d.datetime.now().strftime('%Y%m%d'))"') do set "ACTIONS_DIR=%RUN_ROOT%\actions\%%A"
  if not exist "%ACTIONS_DIR%" mkdir "%ACTIONS_DIR%"
  echo [WARN] fallback ACTIONS_DIR=%ACTIONS_DIR% >> "%LOG%"
)
echo [BOOT] ACTIONS_DIR=%ACTIONS_DIR% >> "%LOG%"

set "PF=%ACTIONS_DIR%\portfolio_after_sim.csv"
set "TC=%ACTIONS_DIR%\trade_candidates.csv"
set "SC=%ACTIONS_DIR%\sell_candidates.csv"
set "OUT=%ACTIONS_DIR%\exit_watchlist.csv"

rem Jos kaikki syötteet puuttuvat tai ovat ~tyhjiä -> luo tyhjä output ja palaa rc=0
set "EMPTY=1"
for %%F in ("%PF%" "%TC%" "%SC%") do (
  if exist "%%~F" for %%S in (%%~zF) do if %%S GTR 50 set "EMPTY="
)
if defined EMPTY (
  echo [INFO] Syotteet tyhjia/puuttuvat -> kirjoitetaan tyhja exit_watchlist.csv >> "%LOG%"
  >"%OUT%" echo ticker,side,stop_loss,trail_stop,take_profit,atr20_pct,note
  echo [DONE] rc=0 >> "%LOG%"
  exit /b 0
)

rem Muuten aja Python
"%PY%" "%ROOT%\make_exit_watchlist.py" ^
  --price_cache_dir "%PRICE_DIR%" ^
  --actions_dir "%ACTIONS_DIR%" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo [DONE] rc=%RC% >> "%LOG%"
exit /b %RC%


