@echo off
echo Stopping existing servers...

:: Kill existing Python processes (API server)
taskkill /f /im python.exe 2>nul

:: Wait a moment
timeout /t 2 /nobreak >nul

echo Starting Flask API server...
cd /d "D:\charlesDevelopment\machine_learning_system_design\algorithms\q_learning"
start "Q-Learning API" cmd /k "python api.py"

:: Wait for API to start
timeout /t 3 /nobreak >nul

echo Starting Streamlit...
start "Q-Learning UI" cmd /k "streamlit run streamlit_app.py --server.headless true"

echo.
echo ========================================
echo Q-Learning Demo servers restarted!
echo ========================================
echo Flask API: http://localhost:5000
echo Streamlit UI: http://localhost:8501
echo.
pause