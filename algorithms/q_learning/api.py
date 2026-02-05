"""
Flask API for Q-Learning Algorithm
Provides endpoints for training, inference, and visualization
"""
import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add current directory to path for imports
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.append(current_dir)

from algorithm import QLearningAgent
from environment import GridWorld

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store models and environments
models = {}
environments = {}


@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        "message": "Q-Learning API",
        "version": "1.0.0",
        "endpoints": {
            "/create_environment": "POST - Create new environment",
            "/create_agent": "POST - Create new agent",
            "/train": "POST - Train agent",
            "/predict": "POST - Get action prediction",
            "/evaluate": "POST - Evaluate agent",
            "/visualize/grid": "GET - Get grid visualization",
            "/visualize/qtable": "GET - Get Q-table visualization",
            "/visualize/training": "GET - Get training progress",
            "/save_model": "POST - Save trained model",
            "/load_model": "POST - Load trained model",
            "/list_models": "GET - List available models"
        }
    })


@app.route('/create_environment', methods=['POST'])
def create_environment():
    """Create a new grid world environment"""
    try:
        data = request.get_json()
        env_id = data.get('env_id', 'default')
        size = data.get('size', 5)
        obstacles = data.get('obstacles', [(1, 1), (2, 3), (3, 1)])
        start = tuple(data.get('start', [0, 0]))
        goal = tuple(data.get('goal', [4, 4]))
        
        # Convert obstacles to tuples if they're lists
        obstacles = [tuple(obs) for obs in obstacles]
        
        env = GridWorld(size=size, obstacles=obstacles, start=start, goal=goal)
        environments[env_id] = env
        
        return jsonify({
            "status": "success",
            "message": f"Environment '{env_id}' created successfully",
            "environment": {
                "env_id": env_id,
                "size": size,
                "n_states": env.n_states,
                "n_actions": env.n_actions,
                "start": start,
                "goal": goal,
                "obstacles": obstacles
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/create_agent', methods=['POST'])
def create_agent():
    """Create a new Q-learning agent"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'default')
        env_id = data.get('env_id', 'default')
        
        if env_id not in environments:
            return jsonify({"status": "error", "message": f"Environment '{env_id}' not found"}), 404
        
        env = environments[env_id]
        
        # Agent parameters
        learning_rate = data.get('learning_rate', 0.1)
        discount_factor = data.get('discount_factor', 0.95)
        epsilon = data.get('epsilon', 1.0)
        epsilon_decay = data.get('epsilon_decay', 0.995)
        epsilon_min = data.get('epsilon_min', 0.01)
        
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )
        
        models[agent_id] = {
            'agent': agent,
            'env_id': env_id
        }
        
        return jsonify({
            "status": "success",
            "message": f"Agent '{agent_id}' created successfully",
            "agent": {
                "agent_id": agent_id,
                "env_id": env_id,
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "epsilon": epsilon,
                "epsilon_decay": epsilon_decay,
                "epsilon_min": epsilon_min
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/train', methods=['POST'])
def train_agent():
    """Train a Q-learning agent"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'default')
        n_episodes = data.get('n_episodes', 100)
        verbose = data.get('verbose', False)
        
        if agent_id not in models:
            return jsonify({"status": "error", "message": f"Agent '{agent_id}' not found"}), 404
        
        agent = models[agent_id]['agent']
        env_id = models[agent_id]['env_id']
        env = environments[env_id]
        
        # Train the agent
        agent.train(env, n_episodes=n_episodes, verbose=verbose)
        
        # Get training statistics
        avg_reward = np.mean(agent.episode_rewards[-min(100, len(agent.episode_rewards)):])
        
        return jsonify({
            "status": "success",
            "message": f"Agent '{agent_id}' trained for {n_episodes} episodes",
            "training_stats": {
                "total_episodes": len(agent.episode_rewards),
                "avg_recent_reward": float(avg_reward),
                "current_epsilon": float(agent.epsilon),
                "final_reward": float(agent.episode_rewards[-1]) if agent.episode_rewards else 0
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/predict', methods=['POST'])
def predict_action():
    """Get action prediction from trained agent"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'default')
        state = data.get('state')
        training = data.get('training', False)
        
        if agent_id not in models:
            return jsonify({"status": "error", "message": f"Agent '{agent_id}' not found"}), 404
        
        if state is None:
            return jsonify({"status": "error", "message": "State parameter is required"}), 400
        
        agent = models[agent_id]['agent']
        action = agent.get_action(state, training=training)
        
        # Get Q-values for all actions in this state
        q_values = agent.q_table[state].tolist()
        
        # Get action name
        env_id = models[agent_id]['env_id']
        env = environments[env_id]
        action_name = env.action_names.get(action, 'Unknown')
        
        return jsonify({
            "status": "success",
            "prediction": {
                "action": int(action),
                "action_name": action_name,
                "q_values": q_values,
                "confidence": float(max(q_values)) if q_values else 0.0
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/evaluate', methods=['POST'])
def evaluate_agent():
    """Evaluate trained agent performance"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'default')
        n_episodes = data.get('n_episodes', 100)
        
        if agent_id not in models:
            return jsonify({"status": "error", "message": f"Agent '{agent_id}' not found"}), 404
        
        agent = models[agent_id]['agent']
        env_id = models[agent_id]['env_id']
        env = environments[env_id]
        
        results = agent.evaluate(env, n_episodes=n_episodes)
        
        return jsonify({
            "status": "success",
            "evaluation": {
                "mean_reward": float(results['mean_reward']),
                "success_rate": float(results['success_rate']),
                "n_episodes": n_episodes
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/visualize/grid', methods=['GET'])
def visualize_grid():
    """Get current grid state visualization"""
    try:
        env_id = request.args.get('env_id', 'default')
        
        if env_id not in environments:
            return jsonify({"status": "error", "message": f"Environment '{env_id}' not found"}), 404
        
        env = environments[env_id]
        
        return jsonify({
            "status": "success",
            "visualization": {
                "grid_array": env.get_grid_array().tolist(),
                "grid_string": env.render(),
                "agent_position": env.pos,
                "goal_position": env.goal,
                "obstacles": env.obstacles,
                "size": env.size
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/visualize/qtable', methods=['GET'])
def visualize_qtable():
    """Get Q-table visualization as base64 encoded image"""
    try:
        agent_id = request.args.get('agent_id', 'default')
        
        if agent_id not in models:
            return jsonify({"status": "error", "message": f"Agent '{agent_id}' not found"}), 404
        
        agent = models[agent_id]['agent']
        
        # Create Q-table heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(agent.q_table, cmap='viridis', aspect='auto')
        plt.colorbar(label='Q-Value')
        plt.title('Q-Table Heatmap')
        plt.xlabel('Actions')
        plt.ylabel('States')
        
        # Add action labels
        env_id = models[agent_id]['env_id']
        env = environments[env_id]
        action_labels = [env.action_names[i] for i in range(env.n_actions)]
        plt.xticks(range(env.n_actions), action_labels)
        
        # Convert plot to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            "status": "success",
            "visualization": {
                "image": f"data:image/png;base64,{img_str}",
                "q_table": agent.q_table.tolist()
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/visualize/training', methods=['GET'])
def visualize_training():
    """Get training progress visualization as base64 encoded image"""
    try:
        agent_id = request.args.get('agent_id', 'default')
        
        if agent_id not in models:
            return jsonify({"status": "error", "message": f"Agent '{agent_id}' not found"}), 404
        
        agent = models[agent_id]['agent']
        
        if not agent.episode_rewards:
            return jsonify({"status": "error", "message": "No training data available"}), 400
        
        # Create training progress plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Episode rewards
        ax1.plot(agent.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Moving average
        window = min(50, len(agent.episode_rewards) // 4)
        if len(agent.episode_rewards) > window:
            moving_avg = []
            for i in range(window, len(agent.episode_rewards)):
                moving_avg.append(np.mean(agent.episode_rewards[i-window:i]))
            ax2.plot(range(window, len(agent.episode_rewards)), moving_avg)
            ax2.set_title(f'Moving Average Reward (window={window})')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Reward')
            ax2.grid(True)
        
        plt.tight_layout()
        
        # Convert plot to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            "status": "success",
            "visualization": {
                "image": f"data:image/png;base64,{img_str}",
                "episode_rewards": agent.episode_rewards,
                "total_episodes": len(agent.episode_rewards)
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/save_model', methods=['POST'])
def save_model():
    """Save trained model to file"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'default')
        filename = data.get('filename', f'{agent_id}_model.pkl')
        
        if agent_id not in models:
            return jsonify({"status": "error", "message": f"Agent '{agent_id}' not found"}), 404
        
        agent = models[agent_id]['agent']
        agent.save_model(filename)
        
        return jsonify({
            "status": "success",
            "message": f"Model saved to {filename}"
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/load_model', methods=['POST'])
def load_model():
    """Load trained model from file"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'default')
        filename = data.get('filename')
        
        if not filename:
            return jsonify({"status": "error", "message": "Filename parameter is required"}), 400
        
        if agent_id not in models:
            return jsonify({"status": "error", "message": f"Agent '{agent_id}' not found"}), 404
        
        agent = models[agent_id]['agent']
        agent.load_model(filename)
        
        return jsonify({
            "status": "success",
            "message": f"Model loaded from {filename}"
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/list_models', methods=['GET'])
def list_models():
    """List all available models and environments"""
    try:
        model_list = []
        for agent_id, model_data in models.items():
            agent = model_data['agent']
            env_id = model_data['env_id']
            
            model_list.append({
                "agent_id": agent_id,
                "env_id": env_id,
                "n_states": agent.n_states,
                "n_actions": agent.n_actions,
                "episodes_trained": len(agent.episode_rewards),
                "current_epsilon": float(agent.epsilon)
            })
        
        env_list = []
        for env_id, env in environments.items():
            env_list.append({
                "env_id": env_id,
                "size": env.size,
                "n_states": env.n_states,
                "n_actions": env.n_actions,
                "start": env.start,
                "goal": env.goal,
                "obstacles": env.obstacles
            })
        
        return jsonify({
            "status": "success",
            "models": model_list,
            "environments": env_list
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/simulate', methods=['POST'])
def simulate_episode():
    """Simulate a complete episode and return the path taken"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', 'default')
        max_steps = data.get('max_steps', 50)
        training = data.get('training', False)
        
        if agent_id not in models:
            return jsonify({"status": "error", "message": f"Agent '{agent_id}' not found"}), 404
        
        agent = models[agent_id]['agent']
        env_id = models[agent_id]['env_id']
        env = environments[env_id]
        
        # Run simulation
        state = env.reset()
        path = [{"state": state, "position": env.pos, "action": None, "reward": 0}]
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state, training=training)
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            path.append({
                "state": next_state,
                "position": env.pos,
                "action": action,
                "action_name": env.action_names[action],
                "reward": reward
            })
            
            state = next_state
            if done:
                break
        
        return jsonify({
            "status": "success",
            "simulation": {
                "path": path,
                "total_reward": total_reward,
                "steps": len(path) - 1,
                "success": env.pos == env.goal,
                "final_position": env.pos
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    try:
        # Create default environment and agent
        print("🚀 Q-Learning API Server Starting...")
        
        print("Creating default environment...")
        default_env = GridWorld()
        environments['default'] = default_env
        
        print("Creating default agent...")
        default_agent = QLearningAgent(
            n_states=default_env.n_states,
            n_actions=default_env.n_actions
        )
        models['default'] = {
            'agent': default_agent,
            'env_id': 'default'
        }
        
        print("Default environment and agent created")
        print("API available at: http://localhost:5000")
        
        # List all registered routes for debugging
        print("\n📋 Registered API routes:")
        for rule in app.url_map.iter_rules():
            methods = ', '.join(rule.methods - {'HEAD', 'OPTIONS'})
            print(f"  {methods} {rule.rule}")
        
        print(f"\n🌟 Total routes registered: {len(list(app.url_map.iter_rules()))}")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        import traceback
        traceback.print_exc()