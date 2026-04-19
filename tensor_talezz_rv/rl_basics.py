import numpy as np
import matplotlib.pyplot as plt
import time
import random

class SimpleGridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
        self.hole = (size//2, size//2)
        
    def reset(self):
        self.state = (0, 0)
        return self.state
        
    def step(self, action):
        x, y = self.state
        if action == 0: x = max(0, x-1) # up
        elif action == 1: x = min(self.size-1, x+1) # down
        elif action == 2: y = max(0, y-1) # left
        elif action == 3: y = min(self.size-1, y+1) # right
        
        self.state = (x, y)
        
        if self.state == self.goal:
            return self.state, 10.0, True
        elif self.state == self.hole:
            return self.state, -5.0, True
            
        return self.state, -0.1, False

class QLearningAgent:
    def __init__(self, size=4):
        self.size = size
        self.q_table = np.zeros((size, size, 4)) # Up, Down, Left, Right
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.rewards = []
        
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state[0], state[1]])
        
    def learn(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.lr * td_error

def rl_demo(env_name="gridworld", episodes=100, display=False):
    """
    A simple reinforcement learning demonstration. Learns to navigate a grid.
    """
    print(f"\n🧠 [RL Basics] Starting {env_name} Training Demo...")
    print("   The agent will try to navigate a grid, avoid holes, and reach the goal.")
    
    env = SimpleGridWorld(size=4)
    agent = QLearningAgent(size=4)
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
        agent.epsilon = max(0.1, agent.epsilon * 0.95) # Decay exploration
        agent.rewards.append(total_reward)
        
        if display and ep % 20 == 0:
            print(f"   Episode {ep}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

    print(f"✅ Training completed over {episodes} episodes!")
    return agent

def policy_visualizer(agent):
    """
    Visualizes the agent's learned policy using arrows.
    """
    print("\n🗺️  [RL Policy] Visualizing Learned Strategy...")
    size = agent.size
    grid = np.zeros((size, size))
    arrows = ['↑', '↓', '←', '→']
    
    print("-" * (size * 4 + 1))
    for i in range(size):
        row = "|"
        for j in range(size):
            if (i, j) == (size-1, size-1):
                row += " 🏁 |"
            elif (i, j) == (size//2, size//2):
                row += " 🕳️ |"
            else:
                best_action = np.argmax(agent.q_table[i, j])
                row += f" {arrows[best_action]} |"
        print(row)
        print("-" * (size * 4 + 1))
    print("Note: 🏁 = Goal (+10), 🕳️ = Hole (-5)")

def reward_curve(agent):
    """
    Plots cumulative reward over episodes to show how the agent improves.
    """
    print("\n📈 [RL Performance] Plotting reward curve... (Close window to continue)")
    plt.figure(figsize=(8, 4))
    plt.plot(agent.rewards, color='green', linewidth=2)
    plt.title("RL Agent Training Performance (Reward vs Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, alpha=0.3)
    plt.show()
