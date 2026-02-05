"""
Example usage of the Q-Learning API
Demonstrates how to programmatically use the API
"""
import requests
import json
import time
import matplotlib.pyplot as plt

# API Configuration
API_BASE_URL = "http://localhost:5000"

def create_and_train_agent():
    """Complete example of creating and training a Q-learning agent"""
    
    print("🚀 Q-Learning API Example")
    print("=" * 50)
    
    # Step 1: Create Environment
    print("\n1. Creating environment...")
    env_config = {
        "env_id": "example_env",
        "size": 6,
        "start": [0, 0],
        "goal": [5, 5],
        "obstacles": [[1, 1], [2, 3], [3, 1], [4, 4]]
    }
    
    response = requests.post(f"{API_BASE_URL}/create_environment", json=env_config)
    print(f"Environment created: {response.json()['status']}")
    
    # Step 2: Create Agent
    print("\n2. Creating agent...")
    agent_config = {
        "agent_id": "example_agent",
        "env_id": "example_env",
        "learning_rate": 0.15,
        "discount_factor": 0.9,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.05
    }
    
    response = requests.post(f"{API_BASE_URL}/create_agent", json=agent_config)
    print(f"Agent created: {response.json()['status']}")
    
    # Step 3: Train Agent
    print("\n3. Training agent...")
    training_episodes = 400
    
    # Train in batches to monitor progress
    batch_size = 100
    num_batches = training_episodes // batch_size
    
    training_progress = []
    
    for batch in range(num_batches):
        print(f"   Training batch {batch + 1}/{num_batches} ({batch_size} episodes)")
        
        train_config = {
            "agent_id": "example_agent",
            "n_episodes": batch_size,
            "verbose": False
        }
        
        response = requests.post(f"{API_BASE_URL}/train", json=train_config)
        if response.status_code == 200:
            stats = response.json()['training_stats']
            training_progress.append({
                'batch': batch + 1,
                'total_episodes': stats['total_episodes'],
                'avg_reward': stats['avg_recent_reward'],
                'epsilon': stats['current_epsilon']
            })
            print(f"   ✅ Batch {batch + 1} completed - Avg Reward: {stats['avg_recent_reward']:.2f}")
        else:
            print(f"   ❌ Batch {batch + 1} failed")
    
    # Step 4: Evaluate Performance
    print("\n4. Evaluating agent performance...")
    eval_config = {
        "agent_id": "example_agent",
        "n_episodes": 100
    }
    
    response = requests.post(f"{API_BASE_URL}/evaluate", json=eval_config)
    if response.status_code == 200:
        results = response.json()['evaluation']
        print(f"   📊 Evaluation Results:")
        print(f"   - Mean Reward: {results['mean_reward']:.2f}")
        print(f"   - Success Rate: {results['success_rate']:.1%}")
    
    # Step 5: Demonstrate Policy
    print("\n5. Demonstrating learned policy...")
    
    # Simulate a few episodes
    for episode in range(3):
        sim_config = {
            "agent_id": "example_agent",
            "max_steps": 30,
            "training": False
        }
        
        response = requests.post(f"{API_BASE_URL}/simulate", json=sim_config)
        if response.status_code == 200:
            simulation = response.json()['simulation']
            print(f"   Episode {episode + 1}:")
            print(f"   - Steps: {simulation['steps']}")
            print(f"   - Reward: {simulation['total_reward']:.1f}")
            print(f"   - Success: {'✅' if simulation['success'] else '❌'}")
    
    # Step 6: Show Grid State
    print("\n6. Current environment state:")
    response = requests.get(f"{API_BASE_URL}/visualize/grid", 
                          params={"env_id": "example_env"})
    if response.status_code == 200:
        grid_data = response.json()['visualization']
        print(grid_data['grid_string'])
        print("Legend: A=Agent, G=Goal, X=Obstacle, .=Empty")
    
    # Step 7: Demonstrate Action Prediction
    print("\n7. Action predictions for different states:")
    test_states = [0, 6, 12, 18, 24, 30]  # Sample states
    
    for state in test_states:
        pred_config = {
            "agent_id": "example_agent",
            "state": state,
            "training": False
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", json=pred_config)
        if response.status_code == 200:
            prediction = response.json()['prediction']
            print(f"   State {state:2d}: {prediction['action_name']:>5} (confidence: {prediction['confidence']:.3f})")
    
    print("\n" + "=" * 50)
    print("🎉 Example completed successfully!")
    print("💡 Tips:")
    print("   - Higher learning rate = faster learning but less stable")
    print("   - Higher discount factor = more importance on future rewards")
    print("   - Epsilon controls exploration vs exploitation balance")
    print("   - More episodes generally lead to better performance")

def analyze_hyperparameters():
    """Compare different hyperparameter settings"""
    
    print("\n🔬 Hyperparameter Analysis")
    print("=" * 50)
    
    configurations = [
        {"name": "Fast Learning", "lr": 0.3, "gamma": 0.9, "episodes": 200},
        {"name": "Balanced", "lr": 0.1, "gamma": 0.95, "episodes": 300},
        {"name": "Conservative", "lr": 0.05, "gamma": 0.99, "episodes": 400}
    ]
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\nTesting {config['name']} configuration...")
        
        # Create environment
        env_config = {
            "env_id": f"test_env_{i}",
            "size": 5,
            "start": [0, 0],
            "goal": [4, 4],
            "obstacles": [[1, 1], [2, 3], [3, 1]]
        }
        requests.post(f"{API_BASE_URL}/create_environment", json=env_config)
        
        # Create agent
        agent_config = {
            "agent_id": f"test_agent_{i}",
            "env_id": f"test_env_{i}",
            "learning_rate": config["lr"],
            "discount_factor": config["gamma"],
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01
        }
        requests.post(f"{API_BASE_URL}/create_agent", json=agent_config)
        
        # Train agent
        train_config = {
            "agent_id": f"test_agent_{i}",
            "n_episodes": config["episodes"],
            "verbose": False
        }
        requests.post(f"{API_BASE_URL}/train", json=train_config)
        
        # Evaluate
        eval_config = {
            "agent_id": f"test_agent_{i}",
            "n_episodes": 50
        }
        response = requests.post(f"{API_BASE_URL}/evaluate", json=eval_config)
        
        if response.status_code == 200:
            eval_results = response.json()['evaluation']
            results.append({
                'name': config['name'],
                'mean_reward': eval_results['mean_reward'],
                'success_rate': eval_results['success_rate'],
                'config': config
            })
            print(f"   ✅ {config['name']}: Reward={eval_results['mean_reward']:.2f}, Success={eval_results['success_rate']:.1%}")
    
    # Show comparison
    print("\n📈 Comparison Results:")
    print("-" * 60)
    print(f"{'Configuration':<15} {'Reward':<10} {'Success':<10} {'Episodes':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<15} {result['mean_reward']:<10.2f} {result['success_rate']:<10.1%} {result['config']['episodes']:<10}")
    
    # Find best
    best = max(results, key=lambda x: x['mean_reward'])
    print(f"\n🏆 Best configuration: {best['name']}")

if __name__ == "__main__":
    # Run basic example
    create_and_train_agent()
    
    # Optional: Run hyperparameter analysis
    choice = input("\nRun hyperparameter analysis? (y/n): ").lower().strip()
    if choice == 'y':
        analyze_hyperparameters()
    
    print("\n✨ Example session complete!")
    print("💡 Open Streamlit UI at http://localhost:8501 for interactive exploration!")