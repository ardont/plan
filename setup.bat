@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Indoor Blueprint Recognition Environment Setup
echo ============================================================
echo.

:: Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo [INFO] Creating virtual environment .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
) else (
    echo [INFO] Virtual environment .venv already exists.
)

:: Activate environment and install dependencies
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing dependencies from requirements.txt...
echo (This may take a few minutes due to OpenCV and PyMuPDF download)
pip install -r requirements.txt --default-timeout=1000

if errorlevel 1 (
    echo.
    echo [WARNING] An error occurred during installation.
    echo If you have a slow internet connection, try installing via a mirror.
    echo To do this, open setup.bat and edit the pip install command.
    pause
    exit /b 1
)

:: Create necessary directories
echo.
echo [INFO] Creating folder structure for data...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\templates mkdir data\templates
if not exist data\synthetic mkdir data\synthetic
if not exist output mkdir output

echo.
echo ============================================================
echo   Installation completed successfully!
echo   To run the command line pipeline, use run_pipeline.bat
echo   To run the web interface, use run_ui.bat
echo ============================================================
echo.
pause
