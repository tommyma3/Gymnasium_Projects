import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import yaml
import random
import matplotlib.pyplot as plt

from envs import darkroom_env
from network import ADTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./models/ad_final.pt"
TEST_GOAL = "./history_set/test_goals.pkl"

if __name__ == "__main__":
    
    with open("hyperparameters.yml", "r") as file:
        hp_all = yaml.safe_load(file)
        hp = hp_all["config"]

    dim = hp["dim"]
    horizon = hp["H"]
    num_epochs = hp["num_epochs"]
    n_embd = hp["embd"]
    n_layer = hp["layer"]
    n_head = hp["head"]
    dropout = hp["dropout"]
    seed = hp["seed"]
    max_seq_len = hp["max_seq_len"]


    state_dim = 2
    action_dim = 5
    model = ADTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    test_goals = torch.load(TEST_GOAL, weights_only=False)
    goal = random.choice(test_goals)
    

    print(f"\nEvaluating goal {goal}")

    env = darkroom_env.DarkroomEnv(dim, goal, horizon)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    time_step = 0
    i_episode = 0

    all_states = []
    all_actions = []
    all_rewards = []
    episode_rewards = []

    while time_step <= 50000:
        state, _ = env.reset()
        all_states.append(state)
        current_ep_reward = 0
        
        '''
        if time_step == 0:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            action_one_hot = np.zeros(action_dim)
            action_one_hot[action] = 1

            all_states.append(next_state)
            all_actions.append(action_one_hot)
            all_rewards.append(reward)

            state = next_state
            done = terminated or truncated
            
            time_step += 1

            if done:
                continue
        '''

        for t in range(1, horizon + 1):

            max_timesteps = (max_seq_len - 1) // 3  

            available_timesteps = len(all_actions)  
            use_timesteps = min(max_timesteps, available_timesteps)


            if use_timesteps <= 0:
                # No actions yet; use the most recent state only and sample random action
                states_truncated = all_states[-1:]
                actions_truncated = []
                rewards_truncated = []
            else:
                states_truncated = all_states[-(use_timesteps + 1):]
                actions_truncated = all_actions[-use_timesteps:]
                rewards_truncated = all_rewards[-use_timesteps:]

            state_tensor = torch.from_numpy(np.array(states_truncated, dtype=np.float32)).unsqueeze(0).to(device)
            if len(actions_truncated) == 0:
                action_tensor = torch.zeros((1, 0, action_dim), dtype=torch.float32, device=device)
                reward_tensor = torch.zeros((1, 0), dtype=torch.float32, device=device)
            else:
                action_tensor = torch.from_numpy(np.array(actions_truncated, dtype=np.float32)).unsqueeze(0).to(device)
                reward_tensor = torch.from_numpy(np.array(rewards_truncated, dtype=np.float32)).unsqueeze(0).to(device)

            pred_action_onehot = model(state_tensor, action_tensor, reward_tensor)
            action = torch.argmax(pred_action_onehot[0, -1]).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)

            action_one_hot = np.zeros(action_dim)
            action_one_hot[action] = 1

            all_states.append(next_state)
            all_actions.append(action_one_hot)
            all_rewards.append(reward)

            state = next_state
            done = terminated or truncated
            
            time_step += 1
            current_ep_reward += reward

            if done:
                break

        episode_rewards.append(current_ep_reward)

        if i_episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 0 else 0.0
            print(f"Episode {i_episode}\tTimestep: {time_step}\tAverage Reward: {avg_reward:.2f}")
        
        i_episode += 1

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    if len(episode_rewards) > 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100) / 100, mode='valid')
        plt.plot(moving_avg, linewidth=3, label="Moving Average (100 episodes)")
    plt.title(f'Total Reward per Episode with goal {goal}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f"figs/eval_loss_goal{goal}")
    plt.legend()
                
