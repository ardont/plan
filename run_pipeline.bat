@echo off

if not exist .venv (
    echo [ERROR] Virtual environment .venv not found.
    echo Please run setup.bat first to install dependencies.
    pause
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate

:: Set PYTHONPATH to project root to avoid import issues
set PYTHONPATH=.

:: Run pipeline passing all arguments
python main.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] An error occurred during pipeline execution.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Pipeline completed successfully. Results are in output/.
echo.
pause
