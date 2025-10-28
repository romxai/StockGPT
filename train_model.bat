@echo off
REM Quick training script for the Stock Prediction System

echo ===============================================
echo Stock Prediction System - Training Mode
echo ===============================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run training with default configuration
echo Starting model training...
python main.py --mode train

echo.
echo Training completed! Check the models directory for saved models.
pause