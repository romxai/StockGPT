@echo off
REM Model prediction script for the Stock Prediction System

echo ===============================================
echo Stock Prediction System - Prediction Mode
echo ===============================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Get model path from user
set /p model_path="Enter path to trained model (e.g., models/hybrid_model_20241028_143022.pth): "

if not exist "%model_path%" (
    echo ERROR: Model file not found: %model_path%
    echo Please check the path and try again.
    pause
    exit /b 1
)

REM Run model evaluation/prediction
echo Running model predictions...
python main.py --mode evaluate --model-path "%model_path%"

echo.
echo Predictions completed! Check the results directory for outputs.
pause