# Seasonality + Price (+Optional Options) Pipeline

This folder contains:
- `ml_unified_pipeline.py` — single-file end-to-end pipeline
- `run_ml_daily.cmd` — Windows helper to run the pipeline

## Prereqs (Windows)
Open **Command Prompt** and install dependencies for Python 3.13:
```
C:\Users\OMISTAJA\AppData\Local\Programs\Python\Python313\python.exe -m pip install --upgrade pandas numpy scikit-learn
```

## Usage
Copy `ml_unified_pipeline.py` into your Seasonality project root (same level as `seasonality_reports`).
Then run:
```
run_ml_daily.cmd
```
It will read universe from:
```
seasonality_reports\runs\<RUN>\price_cache\*.csv
```
and write outputs to:
```
seasonality_reports\runs\<RUN>\features\
seasonality_reports\runs\<RUN>\labels\
seasonality_reports\runs\<RUN>\reports\
```

## Notes
- Options features are placeholders (NA) by default. You can add your own columns later and merge them by Date+ticker.
- Thresholds adapt to volatility regime via HV20 of SPY (or a proxy). Edit them in `DEFAULTS` if you like.
- The script chooses "today" automatically as the last common date across your universe minus 5 days (to allow labels). You can override with `--today YYYY-MM-DD`.

Good luck!
