import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from experiments.rl_env import MetabolicLayoutEnv

import random
import numpy as np

def create_mock_graph():
    """Created a simple random graph for testing layout with weights/groups."""
    G = nx.erdos_renyi_graph(n=20, p=0.3)
    
    # Add attributes required by new Reward Function
    for u, v in G.edges():
        # Simulate flux weights: mostly low, some high (backbone)
        G.edges[u, v]['weight'] = random.choice([0.1, 0.1, 0.9])
        
    for n in G.nodes():
        # Simulate GNN groups
        G.nodes[n]['group'] = random.randint(0, 3)
        
    return G

def train_rl_layout():
    print("Setting up RL Environment...")
    graph = create_mock_graph()
    env = MetabolicLayoutEnv(graph)
    
    # Check if the environment follows Gym API
    check_env(env)
    print("Environment check passed.")

    # Initialize Agent
    # MlpPolicy is standard for vector inputs
    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train
    print("Starting Training (50000 steps)...")
    model.learn(total_timesteps=50000)
    
    print("Training complete.")
    model.save("ppo_metabolic_layout")
    print("Model saved to ppo_metabolic_layout.zip")
    
    # Test Run
    obs, info = env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # env.render()
        
    print("Inference test complete.")

if __name__ == "__main__":
    train_rl_layout()
