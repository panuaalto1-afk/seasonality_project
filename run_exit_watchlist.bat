@echo off
cd /d C:\Users\panua\seasonality_project

for /f %%i in ('powershell -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i

echo ========================================
echo EXIT WATCHLIST - %TODAY%
echo ========================================

python make_exit_watchlist.py

echo ========================================
echo [INFO] Exit watchlist valmis: %ERRORLEVEL%
echo ========================================

exit /b %ERRORLEVEL%