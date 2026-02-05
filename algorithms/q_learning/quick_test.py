"""
Quick test of the API endpoints to verify they're working
"""
import requests

API_BASE = "http://localhost:5000"

def test_endpoint(method, endpoint, data=None, params=None):
    try:
        if method == "GET":
            response = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=5)
        else:
            response = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=5)
        
        print(f"✅ {method} {endpoint}: {response.status_code}")
        if response.status_code != 200:
            print(f"   Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ {method} {endpoint}: {e}")
        return False

print("🧪 Testing API endpoints...")

# Test basic endpoints
test_endpoint("GET", "/")
test_endpoint("GET", "/visualize/grid", params={"env_id": "default"})
test_endpoint("GET", "/list_models")

# Test POST endpoints that should work with default data
test_endpoint("POST", "/train", {"agent_id": "default", "n_episodes": 1})
test_endpoint("POST", "/predict", {"agent_id": "default", "state": 0})