import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Call the legacy reset method for backward compatibility
        state = self._legacy_reset()
        return state, {}

    def _legacy_reset(self) -> np.ndarray:
        """Legacy reset method - to be implemented by subclasses"""
        raise NotImplementedError

    def transit(self, state, action):
        raise NotImplementedError

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Call legacy step method and adapt to new API
        state, reward, done, info = self._legacy_step(action)
        terminated = done
        truncated = False  # You can modify this based on your specific termination logic
        return state, reward, terminated, truncated, info

    def _legacy_step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Legacy step method - to be implemented by subclasses"""
        raise NotImplementedError

    def render(self, mode='human'):
        pass

    def deploy_eval(self, ctrl):
        return self.deploy(ctrl)

    def deploy(self, ctrl):
        ob, _ = self.reset()  # Unpack the new reset return format
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            ob, rew, terminated, truncated, _ = self.step(act)
            done = terminated or truncated

            rews.append(rew)
            next_obs.append(ob)

        obs = np.array(obs)
        acts = np.array(acts)
        next_obs = np.array(next_obs)
        rews = np.array(rews)

        return obs, acts, next_obs, rews

class DarkroomEnv(BaseEnv):
    def __init__(self, dim, goal, horizon):
        super().__init__()
        self.dim = dim
        self.goal = np.array(goal)
        self.horizon = horizon
        self.state_dim = 2
        self.action_dim = 5
        self.observation_space = gym.spaces.Box(
            low=0, high=dim - 1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_dim)
        
        # Initialize state tracking
        self.current_step = 0
        self.state = None

    def sample_state(self):
        return self.np_random.integers(0, self.dim, 2) if hasattr(self, 'np_random') else np.random.randint(0, self.dim, 2)

    def sample_action(self):
        i = self.np_random.integers(0, 5) if hasattr(self, 'np_random') else np.random.randint(0, 5)
        a = np.zeros(self.action_space.n)
        a[i] = 1
        return a

    def _legacy_reset(self):
        self.current_step = 0
        self.state = np.array([0, 0], dtype=np.float32)
        return self.state

    def transit(self, state, action):
        action = np.argmax(action)
        assert action in np.arange(self.action_space.n)
        state = np.array(state, dtype=np.float32)
        if action == 0:
            state[0] += 1
        elif action == 1:
            state[0] -= 1
        elif action == 2:
            state[1] += 1
        elif action == 3:
            state[1] -= 1
        state = np.clip(state, 0, self.dim - 1)

        if np.all(state == self.goal):
            reward = 1.0
        else:
            reward = 0.0
        return state, reward

    def _legacy_step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.horizon)
        return self.state.copy(), r, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        if state[0] < self.goal[0]:
            action = 0
        elif state[0] > self.goal[0]:
            action = 1
        elif state[1] < self.goal[1]:
            action = 2
        elif state[1] > self.goal[1]:
            action = 3
        else:
            action = 4
        zeros = np.zeros(self.action_space.n)
        zeros[action] = 1
        return zeros


class DarkroomEnvPermuted(DarkroomEnv):
    """
    Darkroom environment with permuted actions. The goal is always the bottom right corner.
    """

    def __init__(self, dim, perm_index, H):
        goal = np.array([dim - 1, dim - 1])
        super().__init__(dim, goal, H)

        self.perm_index = perm_index
        assert perm_index < 120     # 5! permutations in darkroom
        actions = np.arange(self.action_space.n)
        permutations = list(itertools.permutations(actions))
        self.perm = permutations[perm_index]

    def transit(self, state, action):
        perm_action = np.zeros(self.action_space.n)
        perm_action[self.perm[np.argmax(action)]] = 1
        return super().transit(state, perm_action)

    def opt_action(self, state):
        action = super().opt_action(state)
        action = np.argmax(action)
        perm_action = np.where(self.perm == action)[0][0]
        zeros = np.zeros(self.action_space.n)
        zeros[perm_action] = 1
        return zeros


class DarkroomEnvVec(BaseEnv):
    """
    Vectorized Darkroom environment.
    """

    def __init__(self, envs):
        super().__init__()
        self._envs = envs
        self._num_envs = len(envs)

    def _legacy_reset(self):
        return [env.reset()[0] if isinstance(env.reset(), tuple) else env.reset() for env in self._envs]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if seed is not None:
            seeds = [seed + i for i in range(self._num_envs)]
            return [env.reset(seed=s)[0] for env, s in zip(self._envs, seeds)], {}
        else:
            return [env.reset()[0] if isinstance(env.reset(), tuple) else env.reset() for env in self._envs], {}

    def step(self, actions) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], Dict[str, Any]]:
        next_obs, rews, terminateds, truncateds = [], [], [], []
        for action, env in zip(actions, self._envs):
            result = env.step(action)
            if len(result) == 5:  # New API
                next_ob, rew, terminated, truncated, _ = result
            else:  # Legacy API
                next_ob, rew, done, _ = result
                terminated = done
                truncated = False
            
            next_obs.append(next_ob)
            rews.append(rew)
            terminateds.append(terminated)
            truncateds.append(truncated)
        return next_obs, rews, terminateds, truncateds, {}

    def _legacy_step(self, actions):
        # For backward compatibility
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            # Handle both old and new step API
            result = env.step(action)
            if len(result) == 5:  # New API
                next_ob, rew, terminated, truncated, _ = result
                done = terminated or truncated
            else:  # Legacy API
                next_ob, rew, done, _ = result
            
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    @property
    def state_dim(self):
        return self._envs[0].state_dim

    @property
    def action_dim(self):
        return self._envs[0].action_dim

    def deploy(self, ctrl):
        ob, _ = self.reset()  # Unpack new reset format
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            result = self.step(act)
            if len(result) == 5:  # New API
                ob, rew, terminateds, truncateds, _ = result
                done = all(terminateds) or all(truncateds)
            else:  # Legacy API (fallback)
                ob, rew, dones, _ = result
                done = all(dones)

            rews.append(rew)
            next_obs.append(ob)

        obs = np.stack(obs, axis=1)
        acts = np.stack(acts, axis=1)
        next_obs = np.stack(next_obs, axis=1)
        rews = np.stack(rews, axis=1)
        return obs, acts, next_obs, rews