import itertools
from typing import Optional, Tuple, Any, Dict

import gymnasium as gym
import numpy as np


class DarkroomEnv(gym.Env):
    """
    A 2D grid world environment where an agent starts at (0, 0) and must
    navigate to a goal position in a dark room.

    The episode terminates if the agent reaches the goal or the time
    horizon is exceeded.

    ### Action Space
    The action space is a `gym.spaces.Discrete(5)` with the following actions:
    - 0: Move Right (+x)
    - 1: Move Left (-x)
    - 2: Move Up (+y)
    - 3: Move Down (-y)
    - 4: Stay

    ### Observation Space
    The observation space is the agent's current (x, y) coordinates.

    ### Rewards
    - A reward of 1.0 is given for reaching the goal state.
    - A reward of 0.0 is given for all other transitions.

    ### Termination
    An episode is terminated if the agent reaches the goal.

    ### Truncation
    An episode is truncated if the step count exceeds the horizon.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, dim: int, goal: Tuple[int, int], horizon: int):
        super().__init__()
        self.dim = dim
        self.goal = np.array(goal, dtype=np.float32)
        self.horizon = horizon

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=dim - 1, shape=(2,), dtype=np.float32
        )

        # Environment state
        self.state: Optional[np.ndarray] = None
        self.current_step: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.state = np.array([4, 4], dtype=np.float32)
        self.current_step = 0
        return self.state.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Cannot call step before reset.")

        # Apply action to state
        if action == 0:
            self.state[0] += 1
        elif action == 1:
            self.state[0] -= 1
        elif action == 2:
            self.state[1] += 1
        elif action == 3:
            self.state[1] -= 1
        # Action 4: Stay in place

        # Clip state to stay within grid boundaries
        self.state = np.clip(self.state, 0, self.dim - 1)
        self.current_step += 1

        # Check for goal-reaching (termination)
        terminated = np.array_equal(self.state, self.goal)
        reward = 1.0 if terminated else 0.0

        # Check for time limit (truncation)
        truncated = self.current_step >= self.horizon

        return self.state.copy(), reward, terminated, truncated, {}
    
    def sample_state(self) -> np.ndarray:
        """Sample a random state from the observation space."""
        return self.observation_space.sample()

    def opt_action(self, state: np.ndarray) -> int:
        """Returns the optimal integer action for a given state."""
        if state[0] < self.goal[0]:
            return 0  # Move Right
        elif state[0] > self.goal[0]:
            return 1  # Move Left
        elif state[1] < self.goal[1]:
            return 2  # Move Up
        elif state[1] > self.goal[1]:
            return 3  # Move Down
        else:
            return 4  # Stay


class DarkroomEnvPermuted(DarkroomEnv):
    """
    A Darkroom environment where the actions are permuted.
    The goal is fixed at the bottom-right corner.
    """
    def __init__(self, dim: int, perm_index: int, horizon: int):
        goal = (dim - 1, dim - 1)
        super().__init__(dim, goal, horizon)

        if not 0 <= perm_index < 120:
             raise ValueError("perm_index must be between 0 and 119 (5! - 1).")

        self.perm_index = perm_index
        
        # Generate the specific permutation of actions [0, 1, 2, 3, 4]
        actions = np.arange(self.action_space.n)
        self.perm_map = list(itertools.permutations(actions))[perm_index]
        self.inverse_perm_map = np.argsort(self.perm_map)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies the permuted action before calling the parent's step method."""
        permuted_action = self.perm_map[action]
        return super().step(permuted_action)
    
    def opt_action(self, state: np.ndarray) -> int:
        """Returns the optimal permuted action for a given state."""
        # Find the standard optimal action (e.g., 0 for "Move Right")
        standard_opt_action = super().opt_action(state)
        # Find which of our actions maps to that standard action
        return self.inverse_perm_map[standard_opt_action]