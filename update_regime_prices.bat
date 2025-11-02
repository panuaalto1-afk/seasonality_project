@echo off
chcp 65001 > nul
cd /d C:\Users\panua\seasonality_project
call .venv\Scripts\activate.bat
python update_regime_prices.py >> logs\update_regime_prices.log 2>&1