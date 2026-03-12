@echo off
echo ============================================
echo  LexiScan Auto - One-Click Setup (Windows)
echo ============================================
echo.

:: Step 1 - Create conda environment with Python 3.11
echo [1/4] Creating conda environment (Python 3.11)...
call conda create -n lexiscan python=3.11 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    echo Make sure Anaconda is installed and accessible.
    pause
    exit /b 1
)

:: Step 2 - Activate environment
echo.
echo [2/4] Activating lexiscan environment...
call conda activate lexiscan

:: Step 3 - Install Python dependencies
echo.
echo [3/4] Installing Python dependencies...
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check requirements.txt and your internet connection.
    pause
    exit /b 1
)

:: Step 4 - Download SpaCy model
echo.
echo [4/4] Downloading SpaCy language model...
python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo ERROR: SpaCy model download failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo To start the server, run:
echo   run.bat
echo.
pause
