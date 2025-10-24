import gymnasium as gym
import numpy as np
import yaml
import random
import os
import pickle
import torch

from envs import darkroom_env
from ppo_teacher import PPOAgent



DATASET_DIR = "history_set"
os.makedirs(DATASET_DIR, exist_ok=True)



device = "cpu"


class History_Generator:

    def __init__(self):
        with open("hyperparameters.yml", "r") as file:
            np.random.seed(0)
            random.seed(0)

            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets['config']
            ppo_config = all_hyperparameter_sets['ppo']

            self.env = hyperparameters['env']
            self.n_envs = hyperparameters['envs']
            self.n_hists = hyperparameters['hists']
            self.n_samples = hyperparameters['samples']
            self.horizon = hyperparameters['H']
            self.dim = hyperparameters['dim']
            self.var = hyperparameters['var']
            self.cov = hyperparameters['cov']
            self.env_id_start = hyperparameters['env_id_start']
            self.env_id_end = hyperparameters['env_id_end']
            self.lin_d = hyperparameters['lin_d']

            self.total_timesteps = ppo_config['total_timesteps']
            self.timesteps_per_batch = ppo_config['timesteps_per_batch']
            self.k_epochs = ppo_config['k_epochs']
            self.eps_clip = ppo_config['eps_clip']
            self.gamma = ppo_config['gamma']
            self.lr_actor = ppo_config['lr_actor']
            self.lr_critic = ppo_config['lr_critic']
            self.patience = ppo_config['patience']

            self.HISTORY_STATE_FILE = os.path.join(DATASET_DIR, 'history_state.pkl')
            self.HISTORY_ACTION_FILE = os.path.join(DATASET_DIR, 'history_action.pkl')
            self.HISTORY_REWARD_FILE = os.path.join(DATASET_DIR, 'history_reward.pkl')
            self.TRAIN_GOAL_FILE = os.path.join(DATASET_DIR, 'train_goal.pkl')
            self.TEST_GOAL_FILE = os.path.join(DATASET_DIR, 'test_goals.pkl')

    def run(self):
        
        goals = np.array([[(j, i) for i in range(self.dim)] for j in range(self.dim)]).reshape(-1, 2)
        np.random.RandomState(seed=0).shuffle(goals)
        train_test_split = int(.8 * len(goals))
        train_goals = goals[:train_test_split]
        test_goals = goals[train_test_split:]

        with open(self.TRAIN_GOAL_FILE, 'wb') as file:
            torch.save(train_goals, file, pickle_protocol=4)
        with open(self.TEST_GOAL_FILE, 'wb') as file:
            torch.save(test_goals, file, pickle_protocol=4)

        
        print(f'Training on {device}...')
        print(f'Darkroom Horizon: {self.horizon}')


        history_state, history_action, history_reward = {}, {}, {}
        
        for task_id, goal in enumerate(train_goals):
            print(f"Training PPO teacher on task {task_id} with goal {goal}")

            train_env = darkroom_env.DarkroomEnv(self.dim, goal, self.horizon)
            state_dim = train_env.observation_space.shape[0]
            action_dim = train_env.action_space.n

            agent = PPOAgent(state_dim, action_dim, self.lr_actor, self.lr_critic, self.gamma, self.k_epochs, self.eps_clip)

            time_step = 0
            i_episode = 0
            correct_shot = 0
            
            all_rewards = []
            task_state, task_action, task_reward = [], [], []


            while time_step <= self.total_timesteps:
                state, _ = train_env.reset()
                current_ep_reward = 0



                for t in range(1, self.horizon + 1):
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = train_env.step(action)

                    task_state.append(state)
                    task_action.append(action)
                    task_reward.append(reward)

                    state = next_state

                    done = terminated or truncated

                    agent.buffer.rewards.append(reward)
                    agent.buffer.is_terminals.append(done)

                    time_step += 1
                    current_ep_reward += reward

                    if len(agent.buffer.rewards) == self.timesteps_per_batch:
                        agent.update()


                    if done:
                        break
                
                all_rewards.append(current_ep_reward)

                if i_episode % 50 == 0:
                    avg_reward = np.mean(all_rewards[-50:]) if len(all_rewards) > 0 else 0.0
                    print(f"Episode {i_episode}\tTimestep: {time_step}\tAverage Reward: {avg_reward:.2f}")

                    if avg_reward >= 0.98:
                        correct_shot += 1
                    else:
                        correct_shot = 0

                    if correct_shot >= self.patience:
                        print(f"Task {task_id} converged at episode {i_episode}")
                        break
                
                i_episode += 1
            
            history_state[task_id] = task_state
            history_action[task_id] = task_action
            history_reward[task_id] = task_reward      


        with open(self.HISTORY_STATE_FILE, 'wb') as file:
            torch.save(history_state, file, pickle_protocol=4)
        with open(self.HISTORY_ACTION_FILE, 'wb') as file:
            torch.save(history_action, file, pickle_protocol=4)
        with open(self.HISTORY_REWARD_FILE, 'wb') as file:
            torch.save(history_reward, file, pickle_protocol=4)

        print("\nHistory saved.")
        

                
                    
if __name__ == '__main__':

    generator = History_Generator()
    generator.run()