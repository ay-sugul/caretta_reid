@echo off
setlocal enabledelayedexpansion

REM Navigate to project directory
cd /d "%~dp0"
echo Current directory: %cd%

echo.
echo ===== Python 3.11 Setup and Test Suite =====
echo.

REM Test if Python 3.11 is available via py launcher
echo Step 1: Testing Python 3.11 via py launcher...
py -3.11 --version
if errorlevel 1 (
    echo ERROR: Python 3.11 not found!
    pause
    exit /b 1
)
echo [OK] Python 3.11 found

echo.
echo Step 2: Creating virtual environment...
if not exist "venv\" (
    py -3.11 -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

echo.
echo Step 3: Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

echo.
echo Step 4: Installing/updating pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo ERROR: Failed to install pip!
    pause
    exit /b 1
)
echo [OK] pip is up to date

echo.
echo Step 5: Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    echo Please check requirements.txt and your internet connection
    pause
    exit /b 1
)
echo [OK] All dependencies installed

echo.
echo Step 6: Running pytest test suite...
pytest tests/ -v
if errorlevel 1 (
    echo.
    echo WARNING: Some tests failed, but dependencies are installed
    echo You can still run the project manually
    echo.
    pause
    exit /b 0
)

echo.
echo ===== Setup Complete! =====
echo.
echo Next steps:
echo 1. Create .env file: copy .env.example .env
echo 2. Edit .env with your paths if needed
echo 3. Run: python -m caretta_reid.database.embedding_store
echo 4. Run: python -m caretta_reid.demo.app
echo.
pause
