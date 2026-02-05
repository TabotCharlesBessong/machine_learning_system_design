"""
Q-Learning Algorithm Implementation

A simple and clean implementation of the Q-Learning algorithm
for reinforcement learning in discrete state-action spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from typing import Tuple, Dict, Optional


class QLearningAgent:
    """
    Simple Q-Learning Agent for Reinforcement Learning
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize Q-Learning Agent
        
        Args:
            n_states: Number of discrete states
            n_actions: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Simple training metrics
        self.episode_rewards = []
        
    def get_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool = False):
        """
        Update Q-value using Q-learning update rule
        """
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, environment, max_steps: int = 1000) -> Tuple[float, int]:
        """
        Train agent for one episode
        """
        state = environment.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = self.get_action(state, training=True)
            next_state, reward, done, info = environment.step(action)
            
            self.update_q_value(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        self.decay_epsilon()
        self.episode_rewards.append(total_reward)
        
        return total_reward, steps
    
    def train(self, environment, n_episodes: int, verbose: bool = True):
        """
        Train agent for multiple episodes
        """
        for episode in range(n_episodes):
            total_reward, steps = self.train_episode(environment)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1}: Avg Reward={avg_reward:.2f}, Epsilon={self.epsilon:.3f}")
        
        if verbose:
            print("Training completed!")
    
    def evaluate(self, environment, n_episodes: int = 100) -> Dict:
        """
        Evaluate trained agent
        """
        rewards = []
        
        for episode in range(n_episodes):
            state = environment.reset()
            total_reward = 0
            
            while True:
                action = self.get_action(state, training=False)  # No exploration
                state, reward, done, _ = environment.step(action)
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'success_rate': len([r for r in rewards if r > 0]) / len(rewards)
        }
    
    def plot_training_progress(self):
        """
        Plot simple training progress
        """
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Moving average
        window = 50
        if len(self.episode_rewards) > window:
            moving_avg = []
            for i in range(window, len(self.episode_rewards)):
                moving_avg.append(np.mean(self.episode_rewards[i-window:i]))
            plt.plot(range(window, len(self.episode_rewards)), moving_avg)
            plt.title('Moving Average Reward')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_q_table(self):
        """
        Plot Q-table as heatmap
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.q_table, cmap='viridis')
        plt.colorbar()
        plt.title('Q-Table Values')
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.show()
    
    def get_policy(self) -> np.ndarray:
        """
        Extract learned policy from Q-table
        """
        return np.argmax(self.q_table, axis=1)
    
    def save_model(self, filepath: str):
        """
        Save trained model
        """
        model_data = {
            'q_table': self.q_table,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'episode_rewards': self.episode_rewards
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.episode_rewards = model_data['episode_rewards']
        print(f"Model loaded from {filepath}")