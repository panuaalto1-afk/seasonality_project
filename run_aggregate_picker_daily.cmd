@echo off
setlocal EnableDelayedExpansion

REM --- Siirry projektin juureen ---
cd /d "C:\Users\panua\seasonality_project"

REM --- Python 3.14 polku (sama jota käytit komentoriviltä) ---
set PY=C:\Users\panua\AppData\Local\Python\bin\python.exe

REM --- Päiväys lokitiedoston nimeen (fi-FI: DD.MM.YYYY) ---
for /f "tokens=1-3 delims=. " %%a in ("%date%") do (
  set D=%%a
  set M=%%b
  set Y=%%c
)

set LOGDIR=%CD%\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOG=%LOGDIR%\aggregate_picker_!Y!-!M!-!D!.log

echo [!date! !time!] START >> "!LOG!"
"%PY%" -X utf8 -u "aggregate_seasonality_picker.py" >> "!LOG!" 2>&1
set RC=%ERRORLEVEL%
echo [!date! !time!] EXIT !RC! >> "!LOG!"
exit /b !RC!
