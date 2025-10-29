@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "C:\Users\panua\seasonality_project" || exit /b 2

REM --- paths ---
set "ROOT=C:\Users\panua\seasonality_project"
set "PY=C:\Users\panua\AppData\Local\Python\bin\python.exe"
set "REPORTS=%ROOT%\seasonality_reports"
set "PRICE_CACHE_DIR=%REPORTS%\runs\2025-10-04_0903\price_cache"
set "LOG=%ROOT%\logs\universe_refresh_last.log"

if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

>>"%LOG%" echo [BOOT] %date% %time%
>>"%LOG%" echo [BOOT] ROOT=%ROOT%
>>"%LOG%" echo [BOOT] PY=%PY%
>>"%LOG%" echo [BOOT] PRICE_CACHE_DIR=%PRICE_CACHE_DIR%
>>"%LOG%" echo.

REM --- 1/2: rakenna Constituents_raw.csv price-cachesta ---
"%PY%" "%ROOT%\refresh_constituents_from_price_cache.py" ^
  --project_root "%ROOT%" ^
  --price_cache_dir "%PRICE_CACHE_DIR%" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  >>"%LOG%" echo [ERROR] refresh_constituents_from_price_cache rc=%RC%
  exit /b %RC%
) else (
  >>"%LOG%" echo [OK] refresh_constituents_from_price_cache DONE
)

REM --- 2/2: enrich (index yms.) jos skripti on olemassa ---
if exist "%ROOT%\rebuild_constituents_from_metadata.py" (
  "%PY%" "%ROOT%\rebuild_constituents_from_metadata.py" >> "%LOG%" 2>&1
  set "RC=%ERRORLEVEL%"
  if not "%RC%"=="0" (
    >>"%LOG%" echo [ERROR] rebuild_constituents_from_metadata rc=%RC%
    exit /b %RC%
  ) else (
    >>"%LOG%" echo [OK] rebuild_constituents_from_metadata DONE
  )
) else (
  >>"%LOG%" echo [WARN] rebuild_constituents_from_metadata.py not found - skipping enrichment
)

>>"%LOG%" echo [DONE] %date% %time%
exit /b 0

