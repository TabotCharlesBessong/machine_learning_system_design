"""
Simple test script to verify the Q-Learning API functionality
"""
import requests
import json
import time

API_BASE_URL = "http://localhost:5000"

def test_api():
    """Test all major API endpoints"""
    
    print("🧪 Testing Q-Learning API...")
    
    # Test 1: Check API status
    print("\n1. Testing API connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("❌ API connection failed")
            return
    except:
        print("❌ Cannot connect to API. Make sure Flask server is running on localhost:5000")
        return
    
    # Test 2: Create environment
    print("\n2. Creating environment...")
    env_data = {
        "env_id": "test_env",
        "size": 5,
        "start": [0, 0],
        "goal": [4, 4],
        "obstacles": [[1, 1], [2, 3], [3, 1]]
    }
    
    response = requests.post(f"{API_BASE_URL}/create_environment", json=env_data)
    if response.status_code == 200:
        print("✅ Environment created successfully")
    else:
        print("❌ Failed to create environment")
        print(response.json())
        return
    
    # Test 3: Create agent
    print("\n3. Creating agent...")
    agent_data = {
        "agent_id": "test_agent",
        "env_id": "test_env",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 1.0
    }
    
    response = requests.post(f"{API_BASE_URL}/create_agent", json=agent_data)
    if response.status_code == 200:
        print("✅ Agent created successfully")
    else:
        print("❌ Failed to create agent")
        print(response.json())
        return
    
    # Test 4: Train agent
    print("\n4. Training agent...")
    train_data = {
        "agent_id": "test_agent",
        "n_episodes": 50,
        "verbose": False
    }
    
    response = requests.post(f"{API_BASE_URL}/train", json=train_data)
    if response.status_code == 200:
        result = response.json()
        stats = result.get('training_stats', {})
        print(f"✅ Training completed!")
        print(f"   - Episodes: {stats.get('total_episodes', 'N/A')}")
        print(f"   - Avg Reward: {stats.get('avg_recent_reward', 'N/A'):.2f}")
        print(f"   - Current Epsilon: {stats.get('current_epsilon', 'N/A'):.3f}")
    else:
        print("❌ Training failed")
        print(response.json())
        return
    
    # Test 5: Evaluate agent
    print("\n5. Evaluating agent...")
    eval_data = {
        "agent_id": "test_agent",
        "n_episodes": 20
    }
    
    response = requests.post(f"{API_BASE_URL}/evaluate", json=eval_data)
    if response.status_code == 200:
        result = response.json()
        eval_stats = result.get('evaluation', {})
        print(f"✅ Evaluation completed!")
        print(f"   - Mean Reward: {eval_stats.get('mean_reward', 'N/A'):.1f}")
        print(f"   - Success Rate: {eval_stats.get('success_rate', 'N/A'):.1%}")
    else:
        print("❌ Evaluation failed")
        print(response.json())
        return
    
    # Test 6: Get prediction
    print("\n6. Testing prediction...")
    pred_data = {
        "agent_id": "test_agent",
        "state": 0,
        "training": False
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=pred_data)
    if response.status_code == 200:
        result = response.json()
        pred = result.get('prediction', {})
        print(f"✅ Prediction successful!")
        print(f"   - Action: {pred.get('action')} ({pred.get('action_name')})")
        print(f"   - Confidence: {pred.get('confidence', 0):.3f}")
    else:
        print("❌ Prediction failed")
        print(response.json())
        return
    
    # Test 7: Simulate episode
    print("\n7. Simulating episode...")
    sim_data = {
        "agent_id": "test_agent",
        "max_steps": 25,
        "training": False
    }
    
    response = requests.post(f"{API_BASE_URL}/simulate", json=sim_data)
    if response.status_code == 200:
        result = response.json()
        sim = result.get('simulation', {})
        print(f"✅ Simulation completed!")
        print(f"   - Steps: {sim.get('steps')}")
        print(f"   - Total Reward: {sim.get('total_reward'):.1f}")
        print(f"   - Success: {sim.get('success')}")
        print(f"   - Final Position: {sim.get('final_position')}")
    else:
        print("❌ Simulation failed")
        print(response.json())
        return
    
    # Test 8: Visualize grid
    print("\n8. Testing grid visualization...")
    response = requests.get(f"{API_BASE_URL}/visualize/grid", params={"env_id": "test_env"})
    if response.status_code == 200:
        result = response.json()
        viz = result.get('visualization', {})
        print(f"✅ Grid visualization successful!")
        print("Grid:")
        print(viz.get('grid_string', ''))
    else:
        print("❌ Grid visualization failed")
        print(response.json())
    
    print("\n🎉 All tests completed successfully!")
    print("The Q-Learning API is working correctly!")

if __name__ == "__main__":
    test_api()