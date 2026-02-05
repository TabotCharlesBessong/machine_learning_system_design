# Q-Learning API & UI Demo

This directory contains a complete demonstration of the Q-Learning algorithm with both a Flask API backend and a Streamlit web interface.

## 🚀 Quick Start

### Windows
```bash
start_demo.bat
```

### Linux/Mac
```bash
chmod +x start_demo.sh
./start_demo.sh
```

### Manual Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server:**
   ```bash
   python api.py
   ```

3. **Start the UI (in a new terminal):**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🌐 Access Points

- **Flask API:** http://localhost:5000
- **Streamlit UI:** http://localhost:8501

## 📁 File Structure

```
q_learning/
├── algorithm.py          # Core Q-Learning implementation
├── environment.py        # Grid World environment
├── api.py               # Flask API server
├── streamlit_app.py     # Streamlit web interface
├── requirements.txt     # Python dependencies
├── start_demo.bat       # Windows startup script
├── start_demo.sh        # Linux/Mac startup script
└── README.md           # This file
```

## 🔧 API Endpoints

### Core Operations
- `GET /` - API information
- `POST /create_environment` - Create new environment
- `POST /create_agent` - Create new agent
- `POST /train` - Train agent
- `POST /predict` - Get action prediction
- `POST /evaluate` - Evaluate agent performance

### Visualization
- `GET /visualize/grid` - Get grid state
- `GET /visualize/qtable` - Get Q-table heatmap
- `GET /visualize/training` - Get training progress

### Simulation
- `POST /simulate` - Run complete episode simulation

### Model Management
- `POST /save_model` - Save trained model
- `POST /load_model` - Load trained model
- `GET /list_models` - List available models

## 🎮 Using the Demo

### 1. Environment Setup
- Adjust grid size, start/goal positions
- Configure obstacles (default: [(1,1), (2,3), (3,1)])
- Click "Create New Environment"

### 2. Agent Training
- Set learning parameters (α, γ, ε, decay)
- Choose number of training episodes
- Click "Start Training"

### 3. Evaluation & Visualization
- Evaluate agent performance
- View training progress charts
- Examine Q-table heatmap
- Simulate agent episodes

### 4. Interactive Features
- Real-time training progress
- State-by-state action prediction
- Episode path visualization
- Q-value inspection

## 🔬 API Usage Examples

### Create Environment
```python
import requests

response = requests.post('http://localhost:5000/create_environment', json={
    "env_id": "my_env",
    "size": 5,
    "start": [0, 0],
    "goal": [4, 4],
    "obstacles": [[1, 1], [2, 3], [3, 1]]
})
```

### Train Agent
```python
response = requests.post('http://localhost:5000/train', json={
    "agent_id": "my_agent",
    "n_episodes": 500,
    "verbose": False
})
```

### Get Prediction
```python
response = requests.post('http://localhost:5000/predict', json={
    "agent_id": "my_agent",
    "state": 0,
    "training": False
})
```

## 🎯 Features

### Q-Learning Algorithm
- ε-greedy exploration strategy
- Bellman equation Q-value updates
- Configurable hyperparameters
- Policy extraction from Q-table

### Grid World Environment
- Customizable grid size
- Configurable obstacles, start, and goal
- Reward system: +10 goal, -1 obstacle, -0.1 step, -0.2 wall
- State representation as integers

### Web Interface
- Interactive parameter tuning
- Real-time training monitoring
- Visual Q-table inspection
- Episode simulation and replay
- Performance metrics dashboard

### API Features
- RESTful design
- JSON request/response format
- Error handling and validation
- Model persistence
- Concurrent agent management

## 🔍 Algorithm Details

**Q-Learning Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Key Parameters:**
- `α` (learning_rate): How fast to update Q-values
- `γ` (discount_factor): Importance of future rewards
- `ε` (epsilon): Exploration vs exploitation balance
- `decay`: Rate at which epsilon decreases

## 🐛 Troubleshooting

1. **API Connection Error:**
   - Ensure Flask server is running on port 5000
   - Check firewall settings

2. **Package Import Errors:**
   - Install requirements: `pip install -r requirements.txt`
   - Use virtual environment if needed

3. **Streamlit Not Loading:**
   - Check port 8501 is available
   - Try running: `streamlit run streamlit_app.py --server.port 8502`

4. **Training Not Working:**
   - Verify environment and agent are created first
   - Check API logs for error messages

## 📊 Expected Results

With default parameters (5x5 grid, 300 episodes):
- **Success Rate:** ~95-100%
- **Average Reward:** ~8-10
- **Training Time:** 30-60 seconds
- **Convergence:** Around episode 200-250

## 🔮 Extensions

Possible improvements and extensions:
- Add more complex environments (mazes, dynamic obstacles)
- Implement other RL algorithms (SARSA, DQN)
- Add visualization of policy arrows
- Support for continuous action spaces
- Multi-agent scenarios
- Custom reward functions

## 📝 License

This implementation is for educational and demonstration purposes.