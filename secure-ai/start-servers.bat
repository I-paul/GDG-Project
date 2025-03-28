@echo off
echo Starting servers...

:: Start Python backend
start cmd /k "cd backend && python app.py"

:: Start React frontend (assuming it's running on port 3000)
start cmd /k "cd frontend && npm start"

echo Servers started! Access the application at http://localhost:3000