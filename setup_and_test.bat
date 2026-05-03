@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
echo Working directory: %cd%

REM Test if python is available
echo.
echo Testing Python...
py -3.11 --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo.
    echo Creating virtual environment...
    py -3.11 -m venv venv
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

REM Run pytest
echo.
echo Running pytest...
pytest tests/ -v --tb=short
if errorlevel 1 (
    echo ERROR: Some tests failed!
    pause
    exit /b 1
)

echo.
echo All setup complete and tests passed!
pause
