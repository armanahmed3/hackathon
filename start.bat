@echo off
REM Startup script for the Teachable Machine web app on Windows

echo Starting Teachable Machine web app...
echo Backend will be available at http://localhost:8080
echo Frontend can be accessed by opening ui/index.html in a browser

REM Start the backend server
uvicorn app:app --host 0.0.0.0 --port 8080 --reload