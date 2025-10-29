@echo off
cd /d C:\Users\OMISTAJA\seasonality_project
python daily_one_click.py >> logs\oneclick_%date:~-4%%date:~3,2%%date:~0,2%.log 2>&1
