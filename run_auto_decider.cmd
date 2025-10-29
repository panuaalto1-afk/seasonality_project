@echo off
set "PROJ=C:\Users\panua\seasonality_project"
set "PY=C:\Users\panua\AppData\Local\Python\bin\python.exe"
set "RUN=%PROJ%\seasonality_reports\runs\2025-10-04_0903"
set "PRICE=%RUN%\price_cache"
set "UNIV=%PROJ%\seasonality_reports\Constituents_raw.csv"
set "LOG=%PROJ%\seasonality_reports\logs\auto_decider.log"
mkdir "%PROJ%\seasonality_reports\logs" 2>nul
cd /d "%PROJ%"
"%PY%" auto_decider.py --project_root "%PROJ%" --universe_csv "%UNIV%" --run_root "%RUN%" --price_cache_dir "%PRICE%" --today "2025-10-27" >> "%LOG%" 2>&1
