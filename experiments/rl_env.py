import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

class MetabolicLayoutEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The goal is to layout a metabolic network (Graph) on a 2D grid.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, graph, grid_size=50):
        super(MetabolicLayoutEnv, self).__init__()
        
        self.graph = graph
        self.num_nodes = len(graph.nodes)
        self.grid_size = grid_size
        self.node_mapping = {n: i for i, n in enumerate(graph.nodes)}
        self.idx_to_node = {i: n for n, i in self.node_mapping.items()}
        
        # State: [num_nodes, 2] array of (x, y) coordinates
        # Normalized to [0, 1] for stable training, mapped to grid_size for reward
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_nodes, 2), dtype=np.float32)
        
        # Action: Select a node (index) and Move it (dx, dy)
        # Simplified: Discrete actions for "Move Node i in Direction D"
        # Or Continuous: [node_idx, dx, dy] -> Hard for standard RL
        # Alternative: MultiDiscrete?
        # Let's try: Agent controls ONE node per step? Or moves ALL nodes?
        # Simpler approach: Action = [node_index, dx, dy] where dx, dy are in {-1, 0, 1}
        # But this has a variable action space size if node count changes.
        # FIX: For now, let's assume we are training a policy that takes the WHOLE state and outputs a shift for ALL nodes?
        # Too high dim.
        
        # "One-shot" approach used often in layout: Force-directed is iterative.
        # Let's make the agent control a single "cursor" or iterate through nodes?
        # For this prototype: Action = Shift for ALL nodes (continuous perturbation)
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(self.num_nodes, 2), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Random initialization
        self.state = np.random.rand(self.num_nodes, 2).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # Update positions
        # Clip to ensure within [0, 1]
        self.state = np.clip(self.state + action, 0, 1)
        
        reward = self._calculate_reward()
        done = False # Continuous task, or stop after N steps
        truncated = False
        info = {}
        
        return self.state, reward, done, truncated, info

    def _calculate_reward(self, prev_state=None, current_state=None):

        if current_state is None:
            current_state = self.state
            
        # 1. R_ortho_ratio: Percentage of bad edges
        # User requested: "make it proportional to how many percentage is not horizont and vertical"
        
        total_edges = self.graph.number_of_edges()
        if total_edges == 0:
            return 0.0
            
        bad_edge_count = 0
        epsilon = 0.05 # Strict tolerance
        
        # We also track global compactness here
        total_edge_length = 0.0
        
        for u, v in self.graph.edges:
            p1 = current_state[self.node_mapping[u]]
            p2 = current_state[self.node_mapping[v]]
            
            dx = abs(p1[0] - p2[0])
            dy = abs(p1[1] - p2[1])
            dist = np.sqrt(dx**2 + dy**2)
            total_edge_length += dist
            
            # Check if NOT orthogonal
            # It's bad if BOTH dx > epsilon AND dy > epsilon
            if dx > epsilon and dy > epsilon:
                bad_edge_count += 1
                
        ratio_bad = bad_edge_count / total_edges
        # Massive penalty proportional to the percentage
        R_ortho = -(ratio_bad * 1000.0) 
        
        # 2. R_compact: Global Compactness
        R_compact = -(total_edge_length * 5.0)
            
        # 3. R_overlap: Collision Avoidance (O(N^2))
        R_overlap = 0.0
        min_dist = 0.05
        
        # Vectorized overlap check
        diffs = current_state[:, np.newaxis, :] - current_state[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        
        mask = np.triu(np.ones(dists_sq.shape), k=1).astype(bool)
        pair_dists_sq = dists_sq[mask]
        
        collisions = np.sum(pair_dists_sq < min_dist**2)
        # Further increased punishment for overlapping (User requested)
        R_overlap = -collisions * 500.0 
        
        # 4. R_boundary: Stay INSIDE canvas
        R_boundary = 0.0
        margin = 0.05
        x_violations = np.sum(current_state[:, 0] < margin) + np.sum(current_state[:, 0] > (1 - margin))
        y_violations = np.sum(current_state[:, 1] < margin) + np.sum(current_state[:, 1] > (1 - margin))
        R_boundary = -(x_violations + y_violations) * 100.0

        # Composite
        total_reward = R_ortho + R_compact + R_overlap + R_boundary
        return float(total_reward)

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Current Reward: {self._calculate_reward(None, self.state)}")
