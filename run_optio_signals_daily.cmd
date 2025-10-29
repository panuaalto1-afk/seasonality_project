@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

REM === Juuri (projektikansio) tämän .cmd-tiedoston sijainnista ===
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

REM === Lokit ===
set "LOGDIR=%ROOT%\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOG=%LOGDIR%\optio_signals_last.log"

> "%LOG%" (
  echo [BOOT] ROOT=%ROOT%
)

REM === Pythonin polku: 1) venv 2) käyttäjän Python 3) PATH ===
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%LOCALAPPDATA%\Python\bin\python.exe"
for %%P in ("%PY%") do if not exist "%%~fP" set "PY=python"

>> "%LOG%" echo [BOOT] PY=%PY%

REM === RUNS-kansio ja uusin price_cache ===
set "RUNS_DIR=%ROOT%\seasonality_reports\runs"
set "LATEST_RUN="
for /f "delims=" %%D in ('dir "%RUNS_DIR%" /ad /b /o:-n 2^>nul') do (
  set "LATEST_RUN=%%D"
  goto :gotrun
)
:gotrun

if defined LATEST_RUN (
  set "PRICE_DIR=%RUNS_DIR%\%LATEST_RUN%\price_cache"
) else (
  set "PRICE_DIR="
)

if exist "%PRICE_DIR%" (
  >> "%LOG%" echo [BOOT] PRICE_DIR=%PRICE_DIR%
) else (
  >> "%LOG%" echo [BOOT] PRICE_DIR=NOT FOUND \(ajetaan ilman price_cachea\)
  set "PRICE_DIR="
)

REM === Universe CSV: kokeile kahta polkua ja valitse löytyvä ===
set "UNIVERSE=%ROOT%\seasonality_reports\Constituents_raw.csv"
if not exist "%UNIVERSE%" set "UNIVERSE=%ROOT%\seasonality_reports\aggregates\Constituents_raw.csv"

if not exist "%UNIVERSE%" (
  >> "%LOG%" echo [ERROR] Universe CSV puuttuu:
  >> "%LOG%" echo        1^] %ROOT%\seasonality_reports\Constituents_raw.csv
  >> "%LOG%" echo        2^] %ROOT%\seasonality_reports\aggregates\Constituents_raw.csv
  exit /b 3
)

>> "%LOG%" echo [BOOT] UNIVERSE=%UNIVERSE%

REM === Aja optio_seasonality_signal.py ===
pushd "%ROOT%" >nul

set "SCRIPT=%ROOT%\optio_seasonality_signal.py"
if not exist "%SCRIPT%" (
  >> "%LOG%" echo [ERROR] Ei loydy: %SCRIPT%
  popd >nul
  exit /b 4
)

REM Rakenna valinnainen price-argumentti
set "PRICE_ARG="
if defined PRICE_DIR if exist "%PRICE_DIR%" set "PRICE_ARG= --price-dir "%PRICE_DIR%""

REM Ajoparametrit (muokkaa tarvittaessa TOP_N tms.)
set "ARGS=--root "%ROOT%" --universe_csv "%UNIVERSE%"%PRICE_ARG% --top-n 60"

>> "%LOG%" echo [RUN ] "%PY%" "%SCRIPT%" %ARGS%

"%PY%" "%SCRIPT%" %ARGS%
set "EC=%ERRORLEVEL%"

if %EC% NEQ 0 (
  >> "%LOG%" echo [FAIL] Exit code %EC%
) else (
  >> "%LOG%" echo [OK  ] Exit code 0
)

popd >nul
exit /b %EC%

