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

echo [INFO] Starting Streamlit web interface...
streamlit run src/ui/app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start web interface.
    pause
    exit /b 1
)
