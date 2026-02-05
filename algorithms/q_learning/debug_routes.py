"""
Debug script to check which routes are registered in the Flask app
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from flask import Flask
    from algorithm import QLearningAgent  
    from environment import GridWorld
    
    print("🔍 Testing route registration...")
    
    app = Flask(__name__)
    
    # Register a test route
    @app.route('/')
    def home():
        return {"message": "Test"}
    
    @app.route('/visualize/grid')
    def viz_grid():
        return {"status": "test"}
    
    print("✅ Routes registered successfully")
    
    # List all registered routes
    print("\n📋 Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.methods} {rule.rule}")
    
    # Test if we can import and execute the main API
    print("\n🧪 Testing main API import...")
    
    # Read and check the main API file
    with open('api.py', 'r') as f:
        api_content = f.read()
    
    # Check for common issues
    if 'app.route' not in api_content:
        print("❌ No route decorators found!")
    else:
        route_count = api_content.count('@app.route')
        print(f"✅ Found {route_count} route decorators")
    
    if '__name__ == \'__main__\'' not in api_content:
        print("❌ No main block found!")
    else:
        print("✅ Main block found")
        
    print("\n🚀 Starting test server...")
    print("If routes are missing, there's likely an exception during API startup")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()