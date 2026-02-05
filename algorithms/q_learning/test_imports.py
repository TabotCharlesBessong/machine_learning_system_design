"""
Simple test to verify API server is working and identify issues
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    
    # Test core imports
    from flask import Flask
    print("✅ Flask imported successfully")
    
    from flask_cors import CORS
    print("✅ Flask-CORS imported successfully")
    
    import numpy as np
    print("✅ NumPy imported successfully")
    
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported successfully")
    
    # Test local imports
    from algorithm import QLearningAgent
    print("✅ QLearningAgent imported successfully")
    
    from environment import GridWorld
    print("✅ GridWorld imported successfully")
    
    # Test creating basic objects
    env = GridWorld()
    print("✅ GridWorld created successfully")
    
    agent = QLearningAgent(n_states=25, n_actions=4)
    print("✅ QLearningAgent created successfully")
    
    print("\n🎉 All imports and basic functionality working!")
    print("The API should work properly.")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("You may need to install missing packages:")
    print("pip install flask flask-cors numpy matplotlib")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("There may be an issue with the implementation files.")

print("\nTesting Flask app creation...")
try:
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def test():
        return {"message": "Test successful", "status": "ok"}
    
    print("✅ Flask app created with routes successfully")
    print("You can now run: python api.py")
    
except Exception as e:
    print(f"❌ Flask Error: {e}")