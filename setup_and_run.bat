@echo off
REM Integrated Stock Prediction System - Setup and Execution Script
REM This script initializes the virtual environment and runs the system

echo ===============================================
echo Integrated Stock Prediction System
echo ===============================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements with better error handling
echo Installing Python packages...
echo Note: This may take several minutes...
pip install -r requirements.txt --no-cache-dir
if errorlevel 1 (
    echo.
    echo WARNING: Some packages failed to install
    echo Attempting to install core packages individually...
    echo.
    
    REM Install core packages first
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers pandas numpy scikit-learn
    pip install yfinance requests beautifulsoup4 feedparser
    pip install matplotlib seaborn plotly
    pip install optuna pyyaml tqdm
    
    echo.
    echo Core packages installed. Some optional packages may be missing.
    echo You can continue with the system setup.
    echo.
)

REM Note: spaCy removed to avoid numpy conflicts - using regex-based NER instead
echo NLP setup complete (using FinBERT + regex-based entity extraction)

REM Create necessary directories
echo Creating directory structure...
if not exist "logs" mkdir logs
if not exist "data\cache" mkdir data\cache
if not exist "models" mkdir models
if not exist "results" mkdir results

echo.
echo ===============================================
echo Setup completed successfully!
echo ===============================================
echo.

REM Ask user what to run
:menu
echo What would you like to do?
echo.
echo 1. Run full pipeline (data collection + training + evaluation)
echo 2. Run data collection only
echo 3. Run training only
echo 4. Run evaluation only
echo 5. Run with hyperparameter optimization
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo Running full pipeline...
    python main.py --mode full
    goto end
) else if "%choice%"=="2" (
    echo Running data collection...
    python main.py --mode collect
    goto end
) else if "%choice%"=="3" (
    echo Running training...
    python main.py --mode train
    goto end
) else if "%choice%"=="4" (
    echo Running evaluation...
    set /p model_path="Enter path to trained model: "
    python main.py --mode evaluate --model-path "%model_path%"
    goto end
) else if "%choice%"=="5" (
    echo Running full pipeline with hyperparameter optimization...
    echo WARNING: This may take several hours to complete.
    set /p confirm="Continue? (y/n): "
    if /i "%confirm%"=="y" (
        python main.py --mode full --optimize
    )
    goto end
) else if "%choice%"=="6" (
    goto end
) else (
    echo Invalid choice. Please enter 1-6.
    goto menu
)

:end
echo.
echo ===============================================
echo Task completed!
echo Check the results directory for outputs.
echo ===============================================
pause