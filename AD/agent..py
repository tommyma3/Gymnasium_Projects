import os
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import trange, tqdm
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv



Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

DATE_FORMAT = "%Y%m%d_%H%M%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



