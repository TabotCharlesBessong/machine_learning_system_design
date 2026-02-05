#!/bin/bash

echo "Starting Q-Learning Demo Servers..."

echo
echo "Installing required packages..."
pip install -r requirements.txt

echo
echo "Starting Flask API Server..."
python api.py &
API_PID=$!

echo "Waiting for API server to start..."
sleep 5

echo
echo "Starting Streamlit UI..."
streamlit run streamlit_app.py &
UI_PID=$!

echo
echo "========================================"
echo "Q-Learning Demo is running!"
echo "========================================"
echo
echo "Flask API: http://localhost:5000"
echo "Streamlit UI: http://localhost:8501"
echo
echo "Press Ctrl+C to stop all servers"

# Function to handle cleanup
cleanup() {
    echo "Stopping servers..."
    kill $API_PID $UI_PID 2>/dev/null
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for background processes
wait