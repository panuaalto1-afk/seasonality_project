import os
from pathlib import Path

def check_data():
    project_root = Path.cwd()
    print('\n' + '='*70)
    print('  BACKTEST DATA CHECKER')
    print('='*70)
    
    runs_dir = project_root / 'seasonality_reports' / 'runs'
    print(f'\nChecking: {runs_dir}')
    
    if not runs_dir.exists():
        print('❌ runs directory not found!')
        return
    
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    print(f'✅ Found {len(runs)} runs')
    
    if runs:
        latest = runs[0]
        print(f'Latest: {latest.name}')
        
        pc = latest / 'price_cache'
        if pc.exists():
            csvs = list(pc.glob('*.csv'))
            print(f'  ✅ price_cache: {len(csvs)} files')
        else:
            print(f'  ❌ No price_cache')

check_data()
