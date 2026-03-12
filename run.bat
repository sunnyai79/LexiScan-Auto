@echo off
echo ============================================
echo  LexiScan Auto - Starting Server
echo ============================================
echo.

:: Activate conda environment
call conda activate lexiscan

:: Go to backend folder
cd backend

:: Start FastAPI server
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
uvicorn app.main:app --reload --port 8000

pause
