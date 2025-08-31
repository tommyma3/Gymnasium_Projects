import datetime
from random import random
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from torch import nn
from experience_replay import ReplayMemory
import itertools
import yaml

import os

import numpy as np
import argparse

import matplotlib
import matplotlib.pyplot as plt

DATE_FORMAT = "%Y%m%d-%H%M%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as f:
            all_hyperparameter_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set

        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.stop_on_reward = hyperparameters["stop_on_reward"]
        self.fc1_nodes = hyperparameters["fc1_nodes"]

        self.loss_fn = nn.MSELoss()
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

        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            best_reward = -float('inf')
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:    
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device) 

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1
                    
                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)} New best reward {episode_reward} in episode {episode}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > datetime.timedelta(seconds=10) :
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time



                

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0


    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) 
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122) 
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        '''
        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                target_q = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(new_state).max()

            current_q = policy_dqn(state)

            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        '''

        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)