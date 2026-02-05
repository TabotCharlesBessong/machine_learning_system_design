@echo off
echo Starting Q-Learning Demo Servers...

echo.
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Starting Flask API Server...
start "Q-Learning API" cmd /k "python api.py"

echo Waiting for API server to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Streamlit UI...
start "Q-Learning UI" cmd /k "streamlit run streamlit_app.py"

echo.
echo ========================================
echo Q-Learning Demo is starting up!
echo ========================================
echo.
echo Flask API: http://localhost:5000
echo Streamlit UI: http://localhost:8501
echo.
echo Press any key to exit...
pause > nul