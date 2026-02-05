"""
Minimal Flask API server for testing
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Q-Learning API - Minimal Version",
        "version": "1.0.0",
        "status": "running"
    })

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Test endpoint working",
        "status": "success"
    })

@app.route('/visualize/grid', methods=['GET'])
def visualize_grid():
    env_id = request.args.get('env_id', 'demo')
    return jsonify({
        "status": "success",
        "message": f"Grid visualization for {env_id}",
        "visualization": {
            "grid_array": [[0, 0, 0], [0, 2, 0], [0, 0, 3]],
            "grid_string": "A . .\n. . .\n. . G",
            "agent_position": [1, 1],
            "goal_position": [2, 2],
            "obstacles": [],
            "size": 3
        }
    })

@app.route('/train', methods=['POST'])
def train():
    return jsonify({
        "status": "success",
        "message": "Training completed (test)",
        "training_stats": {
            "total_episodes": 100,
            "avg_recent_reward": 5.0,
            "current_epsilon": 0.1,
            "final_reward": 8.0
        }
    })

if __name__ == '__main__':
    print("🧪 Starting minimal Q-Learning API for testing...")
    print("Available endpoints:")
    print("- GET /")
    print("- GET /test") 
    print("- GET /visualize/grid")
    print("- POST /train")
    app.run(debug=True, host='0.0.0.0', port=5001)