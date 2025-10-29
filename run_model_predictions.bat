@echo off
echo ================================================================
echo                    Stock Prediction Model Runner
echo ================================================================
echo.
echo Starting production prediction pipeline...
echo.

cd /d "%~dp0"

python scripts\model_runner.py

echo.
echo Pipeline execution completed.
echo Press any key to exit...
pause > nul