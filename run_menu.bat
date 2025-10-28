@echo off
REM Quick launcher for the Stock Prediction System
REM Assumes environment and dependencies are already installed.

echo ===============================================
echo Integrated Stock Prediction System
echo ===============================================

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please run setup_and_run.bat first to create the environment.
    pause
    exit /b 1
)

echo Environment activated.
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
echo ===============================================
pause