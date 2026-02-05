"""
Simple Grid World Environment for Q-Learning
"""
import numpy as np


class GridWorld:
    """Simple 5x5 Grid World Environment"""
    
    def __init__(self, size=5, obstacles=None, start=(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or [(1, 1), (2, 3), (3, 1)]
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
        self.action_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        self.n_actions = 4
        self.n_states = size * size
        self.pos = self.start
    
    def reset(self):
        """Reset environment to starting position"""
        self.pos = self.start
        return self.pos[0] * self.size + self.pos[1]
    
    def step(self, action):
        """Take action and return (next_state, reward, done, info)"""
        move = self.actions[action]
        new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
        
        # Check boundaries
        if new_pos[0] < 0 or new_pos[0] >= self.size or new_pos[1] < 0 or new_pos[1] >= self.size:
            reward, done = -0.2, False  # Hit wall
        elif new_pos in self.obstacles:
            self.pos = new_pos
            reward, done = -1.0, False  # Hit obstacle
        elif new_pos == self.goal:
            self.pos = new_pos
            reward, done = 10.0, True  # Reached goal
        else:
            self.pos = new_pos
            reward, done = -0.1, False  # Normal move
        
        state = self.pos[0] * self.size + self.pos[1]
        return state, reward, done, {}
    
    def render(self):
        """Return grid as string representation"""
        grid = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == self.pos:
                    row.append('A')  # Agent
                elif (i, j) == self.goal:
                    row.append('G')  # Goal
                elif (i, j) in self.obstacles:
                    row.append('X')  # Obstacle
                else:
                    row.append('.')  # Empty
            grid.append(' '.join(row))
        return '\n'.join(grid)
    
    def get_grid_array(self):
        """Return grid as numpy array for visualization"""
        grid = np.zeros((self.size, self.size))
        
        # Set values: 0=empty, 1=obstacle, 2=agent, 3=goal
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.pos:
                    grid[i, j] = 2  # Agent
                elif (i, j) == self.goal:
                    grid[i, j] = 3  # Goal
                elif (i, j) in self.obstacles:
                    grid[i, j] = 1  # Obstacle
                else:
                    grid[i, j] = 0  # Empty
        
        return grid
    
    def state_to_position(self, state):
        """Convert state number to (row, col) position"""
        return (state // self.size, state % self.size)
    
    def position_to_state(self, position):
        """Convert (row, col) position to state number"""
        return position[0] * self.size + position[1]