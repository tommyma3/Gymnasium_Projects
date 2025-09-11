from policy import Policy
from environment import MNISTEnv

import torch
from torch import nn
import itertools
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


RUNS_DIR = "runs"
os.makedirs(RUNS, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

