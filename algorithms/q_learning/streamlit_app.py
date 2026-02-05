"""
Streamlit UI for Q-Learning Algorithm Demonstration
Interactive web interface for training and visualizing Q-learning agents
"""
import streamlit as st
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from io import BytesIO
import base64

# Clear cache to ensure fresh start
st.cache_data.clear()
st.cache_resource.clear()

# Page configuration
st.set_page_config(
    page_title="Q-Learning Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:5000"

# Helper functions
def make_api_request(endpoint, method="GET", data=None, params=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            # Use params if provided, otherwise fall back to data for backwards compatibility
            query_params = params if params is not None else data
            response = requests.get(url, params=query_params)
        
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("🚨 Cannot connect to API server. Please make sure the Flask API is running on localhost:5000")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def visualize_grid(grid_data):
    """Create interactive grid visualization"""
    if not grid_data:
        return None
    
    grid_array = np.array(grid_data['grid_array'])
    
    # Create color mapping
    colors = {0: 'white', 1: 'black', 2: 'blue', 3: 'green'}
    labels = {0: 'Empty', 1: 'Obstacle', 2: 'Agent', 3: 'Goal'}
    
    # Create plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grid_array,
        colorscale=[[0, 'white'], [0.33, 'black'], [0.66, 'blue'], [1, 'green']],
        showscale=False,
        hoverinformation='text'
    ))
    
    # Add annotations
    annotations = []
    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            value = grid_array[i, j]
            symbol = {0: '·', 1: 'X', 2: 'A', 3: 'G'}[value]
            color = 'white' if value == 1 else 'black'
            
            annotations.append({
                'x': j, 'y': i,
                'text': symbol,
                'showarrow': False,
                'font': {'size': 20, 'color': color}
            })
    
    fig.update_layout(
        title="Grid World Environment",
        annotations=annotations,
        xaxis={'dtick': 1, 'side': 'top'},
        yaxis={'dtick': 1, 'autorange': 'reversed'},
        width=400, height=400
    )
    
    return fig

def visualize_qtable(qtable_data):
    """Create Q-table heatmap visualization"""
    if not qtable_data:
        return None
    
    qtable = np.array(qtable_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=qtable,
        colorscale='viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="Q-Table Values",
        xaxis_title="Actions (0:Up, 1:Down, 2:Left, 3:Right)",
        yaxis_title="States",
        width=600, height=400
    )
    
    return fig

def plot_training_progress(episode_rewards):
    """Plot training progress"""
    if not episode_rewards:
        return None
    
    df = pd.DataFrame({
        'Episode': range(len(episode_rewards)),
        'Reward': episode_rewards
    })
    
    fig = px.line(df, x='Episode', y='Reward', 
                  title='Training Progress - Episode Rewards')
    fig.update_layout(width=600, height=400)
    
    return fig

# Initialize session state
if 'agent_trained' not in st.session_state:
    st.session_state.agent_trained = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

# Main UI
st.title("🧠 Q-Learning Algorithm Demo")
st.markdown("Interactive demonstration of Q-Learning in a Grid World environment")

# Check API connection
with st.spinner("Connecting to API..."):
    api_status = make_api_request("/")

if not api_status or api_status.get('status') == 'error':
    st.warning("🚨 Cannot connect to Q-Learning API!")
    st.markdown("""
    **API Connection Failed - Running in Demo Mode**
    
    The Flask API server is not responding. This could be due to:
    1. API server not started: `python api.py`
    2. Port 5000 is blocked or in use
    3. Firewall blocking the connection
    
    **Demo Mode Features:**
    - Simulated Q-learning visualization
    - Example training charts
    - Mock environment display
    """)
    
    # Set demo mode flag
    st.session_state.demo_mode = True
    st.session_state.api_connected = False
else:
    st.session_state.api_connected = True
    st.session_state.demo_mode = False
    st.success("✅ Connected to Q-Learning API")

# Sidebar - Configuration
st.sidebar.header("🔧 Configuration")

# Environment Configuration
st.sidebar.subheader("Environment Settings")
env_size = st.sidebar.slider("Grid Size", 3, 8, 5)
start_pos = st.sidebar.selectbox("Start Position", [(0,0), (0,1), (1,0)], index=0)
goal_pos = st.sidebar.selectbox("Goal Position", [(4,4), (4,3), (3,4)], index=0)

# Agent Configuration
st.sidebar.subheader("Agent Parameters")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
discount_factor = st.sidebar.slider("Discount Factor (γ)", 0.1, 1.0, 0.95)
epsilon = st.sidebar.slider("Initial Epsilon (ε)", 0.1, 1.0, 1.0)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.9, 0.999, 0.995, step=0.001)

# Training Configuration
st.sidebar.subheader("Training Settings")
n_episodes = st.sidebar.slider("Training Episodes", 50, 1000, 300, step=50)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🌍 Environment")
    
    if st.session_state.get('demo_mode', False):
        st.info("Running in Demo Mode - Showing example environment")
        
        # Demo grid visualization
        demo_grid = np.array([
            [2, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 3]
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=demo_grid,
            colorscale=[[0, 'white'], [0.33, 'black'], [0.66, 'blue'], [1, 'green']],
            showscale=False
        ))
        
        # Add grid annotations
        annotations = []
        symbols = {0: '·', 1: 'X', 2: 'A', 3: 'G'}
        for i in range(5):
            for j in range(5):
                value = demo_grid[i, j]
                symbol = symbols[value]
                color = 'white' if value == 1 else 'black'
                annotations.append({
                    'x': j, 'y': i, 'text': symbol,
                    'showarrow': False,
                    'font': {'size': 20, 'color': color}
                })
        
        fig.update_layout(
            title="Demo Grid World",
            annotations=annotations,
            xaxis={'dtick': 1}, yaxis={'dtick': 1, 'autorange': 'reversed'},
            width=400, height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Demo Environment Details"):
            st.write("**Grid Size:** 5x5")
            st.write("**Agent Position:** (0, 0)")
            st.write("**Goal Position:** (4, 4)")
            st.write("**Obstacles:** [(1,1), (2,3), (3,1)]")
            st.code("A . . . .\n. X . . .\n. . . X .\n. X . . .\n. . . . G")
        
    else:
        # Real API mode (existing code)
        # Create environment button
        if st.button("🔄 Create New Environment", type="primary"):
            with st.spinner("Creating environment..."):
                response = make_api_request("/create_environment", "POST", {
                    "env_id": "demo",
                    "size": env_size,
                    "start": list(start_pos),
                    "goal": list(goal_pos),
                    "obstacles": [(1, 1), (2, 3), (3, 1)]  # Default obstacles
                })
                
                if response and response.get('status') == 'success':
                    st.success("Environment created successfully!")
                    
                    # Create agent
                    agent_response = make_api_request("/create_agent", "POST", {
                        "agent_id": "demo",
                        "env_id": "demo",
                        "learning_rate": learning_rate,
                        "discount_factor": discount_factor,
                        "epsilon": epsilon,
                        "epsilon_decay": epsilon_decay
                    })
                    
                    if agent_response and agent_response.get('status') == 'success':
                        st.success("Agent created successfully!")
                        st.session_state.agent_trained = False
        
        # Display current environment
        grid_response = make_api_request("/visualize/grid", params={"env_id": "demo"})
        if grid_response and grid_response.get('status') == 'success':
            grid_fig = visualize_grid(grid_response['visualization'])
            if grid_fig:
                st.plotly_chart(grid_fig, use_container_width=True)
            
            # Show environment details
            with st.expander("Environment Details"):
                viz_data = grid_response['visualization']
                st.write(f"**Grid Size:** {viz_data['size']}x{viz_data['size']}")
                st.write(f"**Agent Position:** {viz_data['agent_position']}")
                st.write(f"**Goal Position:** {viz_data['goal_position']}")
                st.write(f"**Obstacles:** {viz_data['obstacles']}")
                st.text("Grid Representation:")
                st.code(viz_data['grid_string'])

with col2:
    st.header("🤖 Agent Training")
    
    if st.session_state.get('demo_mode', False):
        st.info("Demo Mode - Simulated Training")
        
        col2a, col2b = st.columns([1, 1])
        
        with col2a:
            if st.button("🚀 Simulate Training", type="primary"):
                # Simulate training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                import time
                for i in range(1, 11):
                    status_text.text(f"Training episode batch {i}/10")
                    progress_bar.progress(i / 10)
                    time.sleep(0.3)  # Simulate training time
                
                st.success("✅ Simulated training completed! Demo metrics:")
                st.metric("Average Reward", "8.7")
                st.metric("Episodes", "500")
                st.metric("Success Rate", "94%")
        
        with col2b:
            st.metric("Demo Agent Status", "Ready")
            st.metric("Learning Progress", "Advanced")
            st.metric("Policy Quality", "Excellent")
        
        # Demo training chart
        st.subheader("📈 Demo Training Progress")
        episodes = range(1, 501)
        rewards = np.random.normal(8, 2, 500).cumsum() / np.arange(1, 501) * 8 + np.random.normal(0, 0.5, 500)
        
        df = pd.DataFrame({'Episode': episodes, 'Reward': rewards})
        fig = px.line(df, x='Episode', y='Reward', title='Demo Training Progress')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Real API mode (existing code)
        # Training controls
        col2a, col2b = st.columns([1, 1])
        
        with col2a:
            if st.button("🚀 Start Training", type="primary", disabled=st.session_state.training_in_progress):
                st.session_state.training_in_progress = True
                
                with st.spinner(f"Training agent for {n_episodes} episodes..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Train in batches to show progress
                    batch_size = 50
                    total_batches = max(1, n_episodes // batch_size)
                    
                    for batch in range(total_batches):
                        current_episodes = min(batch_size, n_episodes - batch * batch_size)
                        
                        status_text.text(f"Training batch {batch + 1}/{total_batches} ({current_episodes} episodes)")
                        
                        response = make_api_request("/train", "POST", {
                            "agent_id": "demo",
                            "n_episodes": current_episodes,
                            "verbose": False
                        })
                        
                        if response and response.get('status') == 'success':
                            progress = (batch + 1) / total_batches
                            progress_bar.progress(progress)
                        else:
                            st.error("Training failed!")
                            break
                    
                    status_text.text("Training completed!")
                    st.session_state.agent_trained = True
                    st.session_state.training_in_progress = False
                    
                    # Show training results
                    if response and response.get('status') == 'success':
                        stats = response['training_stats']
                        st.success(f"✅ Training completed! Average reward: {stats['avg_recent_reward']:.2f}")
        
        with col2b:
            if st.button("📊 Evaluate Agent", disabled=not st.session_state.agent_trained):
                with st.spinner("Evaluating agent..."):
                    eval_response = make_api_request("/evaluate", "POST", {
                        "agent_id": "demo",
                        "n_episodes": 100
                    })
                    
                    if eval_response and eval_response.get('status') == 'success':
                        eval_data = eval_response['evaluation']
                        st.metric("Average Reward", f"{eval_data['mean_reward']:.1f}")
                        st.metric("Success Rate", f"{eval_data['success_rate']:.1%}")

# Visualization section
if st.session_state.agent_trained:
    st.header("📈 Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Training Progress", "Q-Table", "Policy Simulation"])
    
    with tab1:
        st.subheader("Training Progress")
        training_response = make_api_request("/visualize/training", params={"agent_id": "demo"})
        if training_response and training_response.get('status') == 'success':
            # Display the matplotlib image
            image_data = training_response['visualization']['image']
            st.image(image_data)
            
            # Also create interactive plot
            episode_rewards = training_response['visualization']['episode_rewards']
            if episode_rewards:
                fig = plot_training_progress(episode_rewards)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Q-Table Heatmap")
        qtable_response = make_api_request("/visualize/qtable", params={"agent_id": "demo"})
        if qtable_response and qtable_response.get('status') == 'success':
            # Display the matplotlib image
            image_data = qtable_response['visualization']['image']
            st.image(image_data)
            
            # Also create interactive heatmap
            qtable_data = qtable_response['visualization']['q_table']
            fig = visualize_qtable(qtable_data)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Policy Simulation")
        
        col3a, col3b = st.columns([1, 1])
        
        with col3a:
            if st.button("🎮 Simulate Episode"):
                with st.spinner("Running simulation..."):
                    sim_response = make_api_request("/simulate", "POST", {
                        "agent_id": "demo",
                        "max_steps": 50,
                        "training": False
                    })
                    
                    if sim_response and sim_response.get('status') == 'success':
                        sim_data = sim_response['simulation']
                        
                        st.metric("Total Steps", sim_data['steps'])
                        st.metric("Total Reward", f"{sim_data['total_reward']:.1f}")
                        st.metric("Success", "✅ Yes" if sim_data['success'] else "❌ No")
                        
                        # Show path
                        st.subheader("Agent's Path")
                        path_df = pd.DataFrame(sim_data['path'][1:])  # Skip initial state
                        path_df['step'] = range(len(path_df))
                        
                        if not path_df.empty:
                            st.dataframe(
                                path_df[['step', 'action_name', 'reward', 'position']],
                                use_container_width=True
                            )
        
        with col3b:
            # State-by-state prediction
            st.subheader("Predict Action")
            test_state = st.number_input("Enter State (0-24)", 0, 24, 0)
            
            if st.button("🔮 Predict"):
                pred_response = make_api_request("/predict", "POST", {
                    "agent_id": "demo",
                    "state": test_state,
                    "training": False
                })
                
                if pred_response and pred_response.get('status') == 'success':
                    pred_data = pred_response['prediction']
                    
                    st.write(f"**Predicted Action:** {pred_data['action_name']}")
                    st.write(f"**Confidence:** {pred_data['confidence']:.3f}")
                    
                    # Show Q-values for all actions
                    q_values = pred_data['q_values']
                    actions = ['Up', 'Down', 'Left', 'Right']
                    
                    q_df = pd.DataFrame({
                        'Action': actions,
                        'Q-Value': q_values
                    })
                    
                    st.bar_chart(q_df.set_index('Action'))

# Footer
st.markdown("---")
st.markdown("""
**Q-Learning Demo** - Built with Streamlit and Flask
- 🔗 **Algorithm:** Q-Learning with ε-greedy exploration
- 🌍 **Environment:** Grid World with obstacles
- 🎯 **Goal:** Learn optimal path from start to goal
""")

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (5s)", False):
    time.sleep(5)
    st.rerun()