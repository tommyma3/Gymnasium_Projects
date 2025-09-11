from policy import Policy
from environment import MNISTEnv

import torch
from torch import nn
import itertools
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import datetime
from datetime import datetime, timedelta

DATE_FORMAT = "%Y%m%d-%H%M%S"


RUNS_DIR = "runs"
os.makedirs(RUNS, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as f:
            all_hyperparameter_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

            self.hyperparameter_set = hyperparameter_set

            self.n_envs = hyperparameters["n_envs"]
            self.learning_rate = hyperparameters["learning_rate"]
            self.entropy_coef = hyperparameters["entropy_coef"]
            self.eval_interval = hyperparameters["eval_interval"]
            self.target_accuracy = hyperparameters["target_accuracy"]
            self.patience = hyperparameters["patience"]
            self.seed = hyperparameters["seed"]
            self.hidden_dim = hyperparameters["hidden_dim"]

            self.loss_fn = None
            self.optimizer = None

            self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
            self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
            self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')



    def run(self, is_training=True, render=False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time 

            log_message = f"{start_time.strftime(DATE_FORMAT)} Starting training with hyperparameter set '{self.hyperparameter_set}'"
            print(log_message)
            with open(self.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')

        env = MNISTEnv(split=("train" if is_training else "test"), seed=self.seed, device=device)

        num_actions = env.action_space.n

        policy = Policy(hidden_dim=self.hidden_dim)


    