
### 10. `scripts/run_daily.bat`

```batch
@echo off
REM Daily prediction script for Windows

echo ========================================
echo Soccer Prediction AI - Daily Run
echo ========================================
echo.

REM Set Python path (adjust if needed)
set PYTHON_PATH=python

REM Set project directory
cd /d %~dp0..

REM Create logs directory if not exists
if not exist "logs" mkdir logs

REM Get current date
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do (
    set DATE=%%c-%%b-%%a
)

echo [%TIME%] Starting daily prediction pipeline...

REM Run data collection and prediction
%PYTHON_PATH% src/main.py --mode=full --config=config/config.yaml ^
    >> logs/predictions_%DATE%.log 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [%TIME%] Pipeline completed successfully
) else (
    echo [%TIME%] Pipeline failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

REM Generate summary report
%PYTHON_PATH% -c "
import json
from datetime import datetime
import glob

latest_pred = sorted(glob.glob('output/predictions/*.json'))[-1]
with open(latest_pred, 'r') as f:
    data = json.load(f)

print(f'Predictions generated: {len(data)}')
high_conf = [p for p in data if p['prediction']['confidence'] >= 0.65]
print(f'High confidence predictions: {len(high_conf)}')

if high_conf:
    print('\nTop 5 predictions:')
    for i, pred in enumerate(sorted(high_conf, 
                                   key=lambda x: x['prediction']['confidence'], 
                                   reverse=True)[:5]):
        print(f\"{i+1}. {pred['home_team']} vs {pred['away_team']}: \"
              f\"{pred['prediction']['winner']} \"
              f\"(Confidence: {pred['prediction']['confidence']:.1%})\")
" >> logs/summary_%DATE%.log

echo [%TIME%] Summary report generated
echo.
echo ========================================
echo Daily run completed
echo ========================================

REM Keep window open
pause
