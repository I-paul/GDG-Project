@echo off
echo Starting Secure AI Application...

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed! Please install Python 3.8 or higher.
    pause
    exit /b
)

:: Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo Node.js is not installed! Please install Node.js 14 or higher.
    pause
    exit /b
)

:: Install Python dependencies if needed
echo Checking Python dependencies...
pip install -r backend/requirements.txt

:: Install Node.js dependencies if needed
echo Checking Node.js dependencies...
cd frontend
npm install
cd ..

:: Start Python backend
start cmd /k "cd backend && python app.py"

:: Wait for backend to initialize
timeout /t 5

:: Start React frontend
start cmd /k "cd frontend && npm start"

echo Application started! Access at http://localhost:3000